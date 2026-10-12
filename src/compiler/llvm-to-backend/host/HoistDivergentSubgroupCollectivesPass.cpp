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

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Demangle/Demangle.h>

#include <algorithm>
#include <functional>
#include <optional>

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

// Does a value differ between lanes of one sub-group? Decided by evaluating it for every lane of
// every sub-group of one work-group (size known at JIT time). Inputs that don't depend on the lane
// (kernel arguments, group ids, loads from one address, ...) get the same pseudo-random value in
// all lanes, so two lanes of a sub-group getting different values prove the value lane-dependent.
// Whatever can't be evaluated (PHIs, floating point, private memory, calls) gives Unknown: valid
// code is never reported, some divergent code is missed.
class LaneVariance {
public:
  enum Result { Uniform, Varying, Unknown };

  LaneVariance(const llvm::DataLayout &DL, std::array<int, 3> GroupSize, int SubgroupSize)
      : DL{DL}, Size{uint64_t(std::max(GroupSize[0], 0)), uint64_t(std::max(GroupSize[1], 0)),
                     uint64_t(std::max(GroupSize[2], 0))},
        S{uint64_t(SubgroupSize)} {}

  Result classify(const llvm::Value *V) {
    const uint64_t W = Size[0] * Size[1] * Size[2];
    if (W == 0 || W > 4096)
      return Unknown;
    for (Seed = 1; Seed <= 2; ++Seed)
      for (uint64_t First = 0; First < W; First += S) {
        std::optional<llvm::APInt> Ref;
        for (Lane = First; Lane < std::min(First + S, W); ++Lane) {
          Memo.clear();
          auto X = eval(V);
          if (!X)
            return Unknown;
          if (!Ref)
            Ref = X;
          else if (*X != *Ref)
            return Varying;
        }
      }
    return Uniform;
  }

private:
  const llvm::DataLayout &DL;
  std::array<uint64_t, 3> Size;
  uint64_t S, Seed = 0, Lane = 0;
  llvm::DenseMap<const llvm::Value *, std::optional<llvm::APInt>> Memo;

  // the same pseudo-random value in every lane, for a lane-independent input
  llvm::APInt leaf(const llvm::Value *V, unsigned W, uint64_t Extra = 0) const {
    return llvm::APInt(64, size_t(llvm::hash_combine(Seed, V, Extra))).zextOrTrunc(W);
  }

  std::optional<llvm::APInt> eval(const llvm::Value *V) {
    if (auto It = Memo.find(V); It != Memo.end())
      return It->second;
    auto R = compute(V);
    Memo[V] = R;
    return R;
  }

  std::optional<llvm::APInt> compute(const llvm::Value *V) {
    using namespace llvm;
    Type *T = V->getType();
    if (!T->isIntegerTy() && !T->isPointerTy())
      return std::nullopt;
    const unsigned W = DL.getTypeSizeInBits(T);
    if (W > 64)
      return std::nullopt;
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI->getValue();
    if (isa<Constant>(V) || isa<Argument>(V))
      return leaf(V, W);

    if (auto *Call = dyn_cast<CallInst>(V)) {
      auto *Fn = Call->getCalledFunction();
      if (!Fn)
        return std::nullopt;
      StringRef N = Fn->getName();
      const uint64_t Id[3] = {Lane % Size[0], Lane / Size[0] % Size[1], Lane / (Size[0] * Size[1])};
      if (N.consume_front("__acpp_sscp_get_local_id_"))
        return APInt(W, Id[N[0] - 'x']);
      if (N.consume_front("__acpp_sscp_get_local_size_"))
        return APInt(W, Size[N[0] - 'x']);
      if (N == "__acpp_sscp_get_subgroup_local_id")
        return APInt(W, Lane % S);
      if (N == "__acpp_sscp_get_subgroup_id")
        return APInt(W, Lane / S);
      if (N == "__acpp_sscp_get_subgroup_size" || N == "__acpp_sscp_get_subgroup_max_size")
        return APInt(W, S);
      if (N.starts_with("__acpp_sscp_get_")) // group ids, group counts: the same in every lane
        return leaf(Fn, W);
      return std::nullopt;
    }
    if (auto *L = dyn_cast<LoadInst>(V)) {
      if (isa<AllocaInst>(getUnderlyingObject(L->getPointerOperand())))
        return std::nullopt; // private memory: content per lane unknown
      auto A = eval(L->getPointerOperand());
      if (!A)
        return std::nullopt;
      return leaf(L, W, A->getZExtValue()); // the same address gives the same value
    }
    if (auto *G = dyn_cast<GEPOperator>(V)) {
      auto P = eval(G->getPointerOperand());
      SmallMapVector<Value *, APInt, 4> VarOffsets;
      APInt Offset(W, 0);
      if (!P || !G->collectOffset(DL, W, VarOffsets, Offset))
        return std::nullopt;
      APInt R = *P + Offset;
      for (auto &[Idx, Scale] : VarOffsets) {
        auto X = eval(Idx);
        if (!X)
          return std::nullopt;
        R += X->sextOrTrunc(W) * Scale;
      }
      return R;
    }
    if (auto *C = dyn_cast<CastInst>(V)) {
      auto X = eval(C->getOperand(0));
      if (!X)
        return std::nullopt;
      switch (C->getOpcode()) {
      case Instruction::SExt:
        return X->sext(W);
      case Instruction::Trunc: case Instruction::ZExt: case Instruction::PtrToInt:
      case Instruction::IntToPtr: case Instruction::BitCast: case Instruction::AddrSpaceCast:
        return X->zextOrTrunc(W);
      default:
        return std::nullopt;
      }
    }
    if (auto *B = dyn_cast<BinaryOperator>(V)) {
      auto X = eval(B->getOperand(0)), Y = eval(B->getOperand(1));
      if (!X || !Y)
        return std::nullopt;
      switch (B->getOpcode()) {
      case Instruction::Add: return *X + *Y;
      case Instruction::Sub: return *X - *Y;
      case Instruction::Mul: return *X * *Y;
      case Instruction::And: return *X & *Y;
      case Instruction::Or:  return *X | *Y;
      case Instruction::Xor: return *X ^ *Y;
      case Instruction::Shl: case Instruction::LShr: case Instruction::AShr: {
        if (Y->uge(W))
          return std::nullopt;
        const unsigned Sh = Y->getZExtValue();
        return B->getOpcode() == Instruction::Shl    ? X->shl(Sh)
               : B->getOpcode() == Instruction::LShr ? X->lshr(Sh)
                                                     : X->ashr(Sh);
      }
      case Instruction::UDiv: case Instruction::URem: case Instruction::SDiv: case Instruction::SRem:
        if (Y->isZero())
          return std::nullopt;
        return B->getOpcode() == Instruction::UDiv   ? X->udiv(*Y)
               : B->getOpcode() == Instruction::URem ? X->urem(*Y)
               : B->getOpcode() == Instruction::SDiv ? X->sdiv(*Y)
                                                     : X->srem(*Y);
      default:
        return std::nullopt;
      }
    }
    if (auto *Cmp = dyn_cast<ICmpInst>(V)) {
      auto X = eval(Cmp->getOperand(0)), Y = eval(Cmp->getOperand(1));
      if (!X || !Y)
        return std::nullopt;
      return APInt(1, ICmpInst::compare(*X, *Y, Cmp->getPredicate()));
    }
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      auto C = eval(Sel->getCondition());
      if (!C)
        return std::nullopt;
      return eval(C->isOne() ? Sel->getTrueValue() : Sel->getFalseValue());
    }
    if (auto *F = dyn_cast<FreezeInst>(V))
      return eval(F->getOperand(0));
    return std::nullopt;
  }
};

class Hoister {
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;
  llvm::LoopInfo LI;

  static bool keep(llvm::CallInst &C, const char *Why) {
    HIPSYCL_DEBUG_INFO << "[HoistDivergentSubgroupCollectives] kept " << C << ": " << Why << "\n";
    return false;
  }

  // the block whose branch decides whether B runs: nearest dominator B doesn't post-dominate
  llvm::BasicBlock *controllingBlock(llvm::BasicBlock *B) const {
    for (auto *N = DT.getNode(B)->getIDom(); N; N = N->getIDom())
      if (!PDT.dominates(B, N->getBlock()))
        return N->getBlock();
    return nullptr;
  }

public:
  explicit Hoister(llvm::Function &F) : DT{F}, PDT{F}, LI{DT} {}

  // moves C above its nearest controlling branch; false if C has none or can't be moved
  bool hoistOnce(llvm::CallInst &C) {
    auto *B = C.getParent();
    llvm::BasicBlock *G = controllingBlock(B);
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

  // C (wherever hoisting left it) still runs under a branch that splits a sub-group: warn
  void warnIfDivergent(llvm::CallInst &C, LaneVariance &LV) const {
    for (auto *G = controllingBlock(C.getParent()); G; G = controllingBlock(G)) {
      const llvm::Value *Cond = nullptr;
      if (auto *Br = llvm::dyn_cast<llvm::BranchInst>(G->getTerminator()); Br && Br->isConditional())
        Cond = Br->getCondition();
      else if (auto *Sw = llvm::dyn_cast<llvm::SwitchInst>(G->getTerminator()))
        Cond = Sw->getCondition();
      if (!Cond || LV.classify(Cond) != LaneVariance::Varying)
        continue;
      std::string Kernel = llvm::demangle(C.getFunction()->getName().str());
      if (Kernel.size() > 160)
        Kernel = Kernel.substr(0, 157) + "...";
      // source location (with -g): the innermost inlined frame outside AdaptiveCpp's headers
      std::string Where;
      for (const llvm::DILocation *L = C.getDebugLoc().get(); L; L = L->getInlinedAt()) {
        llvm::StringRef File = L->getFilename();
        if (!File.contains("/hipSYCL/") && !File.contains("/include/AdaptiveCpp/")) {
          Where = " at " + File.str() + ":" + std::to_string(L->getLine());
          break;
        }
      }
      HIPSYCL_DEBUG_WARNING
          << "Sub-group collective "
          << C.getCalledFunction()->getName().drop_front(CollectivePrefix.size()) << Where
          << " is reached by only part of a sub-group (lane-dependent branch) in " << Kernel
          << ". This is undefined in SYCL and gives wrong results on this backend; call it from"
             " every lane of the sub-group (e.g. move it out of the branch).\n";
      return;
    }
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
    LaneVariance LV{M.getDataLayout(), GroupSize, SubgroupSize};
    for (auto *C : Calls)
      H.warnIfDivergent(*C, LV);
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
