/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#include "hipSYCL/compiler/llvm-to-backend/host/HoistDivergentSubgroupCollectivesPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

#include <functional>

namespace hipsycl::compiler {
namespace {

constexpr llvm::StringLiteral CollectivePrefix = "__acpp_sscp_sub_group_";

// same order as __acpp_sscp_algorithm_op
enum AlgorithmOp { Plus, Multiply, Min, Max, BitAnd, BitOr, BitXor, LogicalAnd, LogicalOr };

// identity element of a reduction/scan operation for type Ty (Unsigned: an unsigned integer)
llvm::Constant *getIdentity(int64_t Op, llvm::Type *Ty, bool Unsigned) {
  if (Ty->isFloatingPointTy()) {
    switch (Op) {
    case Plus:     return llvm::ConstantFP::getNegativeZero(Ty);
    case Multiply: return llvm::ConstantFP::get(Ty, 1.0);
    case Min:      return llvm::ConstantFP::getInfinity(Ty, false);
    case Max:      return llvm::ConstantFP::getInfinity(Ty, true);
    default:       return nullptr;
    }
  }
  if (!Ty->isIntegerTy())
    return nullptr;
  const unsigned W = Ty->getIntegerBitWidth();
  switch (Op) {
  case Plus: case BitOr: case BitXor: case LogicalOr:
    return llvm::ConstantInt::get(Ty, 0);
  case Multiply: case LogicalAnd:
    return llvm::ConstantInt::get(Ty, 1);
  case BitAnd:
    return llvm::ConstantInt::getAllOnesValue(Ty);
  case Min:
    return llvm::ConstantInt::get(Ty, Unsigned ? llvm::APInt::getMaxValue(W)
                                               : llvm::APInt::getSignedMaxValue(W));
  case Max:
    return llvm::ConstantInt::get(Ty, Unsigned ? llvm::APInt::getMinValue(W)
                                               : llvm::APInt::getSignedMinValue(W));
  default:
    return nullptr;
  }
}

// Lanes that now call C without having reached it must not change the other lanes' result:
// the operand to replace by the identity for them (Identity == nullptr: none, e.g. a shuffle,
// whose value from such a lane is never trusted). Returns false if no identity is known.
bool getMask(llvm::CallInst &C, unsigned &ArgNo, llvm::Constant *&Identity) {
  auto Name = C.getCalledFunction()->getName().drop_front(CollectivePrefix.size());
  Identity = nullptr;
  if (Name.starts_with("reduce_") || Name.starts_with("inclusive_scan_") ||
      Name.starts_with("exclusive_scan_")) {
    auto *Op = llvm::dyn_cast<llvm::ConstantInt>(C.getArgOperand(0));
    ArgNo = 1;
    Identity = Op ? getIdentity(Op->getSExtValue(), C.getArgOperand(1)->getType(),
                                Name.rsplit('_').second.starts_with("u"))
                  : nullptr;
    return Identity != nullptr;
  }
  if (Name == "any" || Name == "all" || Name == "none") {
    ArgNo = 0;
    Identity = llvm::ConstantInt::get(C.getArgOperand(0)->getType(), Name == "all" ? 1 : 0);
  }
  return true;
}

// may C be moved above I? Only if I touches no memory other lanes or C could see
bool touchesOnlyPrivateMemory(const llvm::Instruction &I) {
  if (!I.mayReadOrWriteMemory())
    return true;
  if (auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(&I))
    return II->isLifetimeStartOrEnd() || llvm::isa<llvm::DbgInfoIntrinsic>(II) ||
           II->getIntrinsicID() == llvm::Intrinsic::assume;
  const llvm::Value *Ptr = llvm::getLoadStorePointerOperand(&I);
  return Ptr && llvm::isa<llvm::AllocaInst>(llvm::getUnderlyingObject(Ptr));
}

class Hoister {
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;
  llvm::LoopInfo LI;

  static bool keep(llvm::CallInst &C, const char *Why) {
    HIPSYCL_DEBUG_INFO << "[HoistDivergentSubgroupCollectives] kept " << C << ": " << Why << "\n";
    return false;
  }

public:
  explicit Hoister(llvm::Function &F) : DT{F}, PDT{F}, LI{DT} {}

  // moves C above its nearest controlling branch; false if C has none or can't be moved
  bool hoistOnce(llvm::CallInst &C) {
    auto *B = C.getParent();
    // the controlling branch: nearest dominator that B doesn't post-dominate
    llvm::BasicBlock *G = nullptr;
    for (auto *N = DT.getNode(B)->getIDom(); N && !G; N = N->getIDom())
      if (!PDT.dominates(B, N->getBlock()))
        G = N->getBlock();
    if (!G)
      return false; // every lane that runs the function reaches C
    auto *Br = llvm::dyn_cast<llvm::BranchInst>(G->getTerminator());
    if (!Br || !Br->isConditional())
      return keep(C, "controlled by a non-branch terminator");
    // the edge every lane reaching C takes, and no other way into its target
    llvm::BasicBlock *S = DT.dominates(Br->getSuccessor(0), B) ? Br->getSuccessor(0)
                          : DT.dominates(Br->getSuccessor(1), B) ? Br->getSuccessor(1)
                                                                 : nullptr;
    if (!S || S->getSinglePredecessor() != G)
      return keep(C, "branch target has other predecessors");
    // same loop, and not a loop exit: C keeps running once per iteration
    auto *L = LI.getLoopFor(G);
    if (LI.getLoopFor(B) != L ||
        (L && (!L->contains(Br->getSuccessor(0)) || !L->contains(Br->getSuccessor(1)))))
      return keep(C, "branch is a loop exit, or C is in a deeper loop");

    unsigned ArgNo = 0;
    llvm::Constant *Identity = nullptr;
    if (!getMask(C, ArgNo, Identity))
      return keep(C, "no identity element for this operation");

    // C's inputs computed after the branch move along: pure arithmetic only (operands first)
    auto *GTerm = G->getTerminator();
    llvm::SmallSetVector<llvm::Instruction *, 8> Slice;
    std::function<bool(llvm::Value *)> Collect = [&](llvm::Value *V) {
      auto *I = llvm::dyn_cast<llvm::Instruction>(V);
      if (!I || DT.dominates(I, GTerm) || Slice.contains(I))
        return true;
      if (llvm::isa<llvm::PHINode>(I) || I->mayReadOrWriteMemory() ||
          !llvm::isSafeToSpeculativelyExecute(I))
        return false;
      for (auto *Op : I->operand_values())
        if (!Collect(Op))
          return false;
      Slice.insert(I);
      return true;
    };
    for (llvm::Value *Op : C.args())
      if (!Collect(Op))
        return keep(C, "an input is not plain arithmetic");

    // C now runs before everything between the branch and itself: private memory only there
    llvm::SmallVector<llvm::BasicBlock *, 8> Work{B};
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> Region{B};
    while (!Work.empty()) {
      auto *X = Work.pop_back_val();
      for (auto *P : llvm::predecessors(X))
        if (DT.dominates(S, P) && Region.insert(P).second)
          Work.push_back(P);
    }
    for (auto *X : Region)
      for (auto &I : *X) {
        if (&I == &C)
          break;
        if (!Slice.contains(&I) && !touchesOnlyPrivateMemory(I))
          return keep(C, "memory or calls between the branch and the collective");
      }

    for (auto *I : Slice)
      I->moveBefore(GTerm);
    C.moveBefore(GTerm);
    if (Identity) {
      llvm::IRBuilder<> Builder{&C};
      llvm::Value *Taken = Br->getCondition();
      if (S != Br->getSuccessor(0))
        Taken = Builder.CreateNot(Taken);
      C.setArgOperand(ArgNo, Builder.CreateSelect(Taken, C.getArgOperand(ArgNo), Identity));
    }
    HIPSYCL_DEBUG_INFO << "[HoistDivergentSubgroupCollectives] hoisted " << C << " into "
                       << G->getName() << "\n";
    return true;
  }
};

} // namespace

llvm::PreservedAnalyses HoistDivergentSubgroupCollectivesPass::run(llvm::Module &M,
                                                                   llvm::ModuleAnalysisManager &) {
  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // in program order, so a collective feeding another one moves first
    llvm::SmallVector<llvm::CallInst *, 16> Calls;
    for (auto *BB : llvm::ReversePostOrderTraversal<llvm::Function *>(&F))
      for (auto &I : *BB)
        if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction();
              Callee && Callee->getName().starts_with(CollectivePrefix))
            Calls.push_back(CI);
    if (Calls.empty())
      continue;
    Hoister H{F};
    for (auto *C : Calls)
      while (H.hoistOnce(*C))
        Changed = true;
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
