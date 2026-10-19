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

#include "hipSYCL/compiler/llvm-to-backend/host/MaskDivergentSubgroupCollectivesPass.hpp"
#include "hipSYCL/compiler/cbs/SubgroupOps.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/SSAUpdater.h>

namespace hipsycl::compiler {
namespace {

constexpr llvm::StringLiteral BuiltinPrefix = "__acpp_sscp_";
constexpr llvm::StringLiteral CollectivePrefix = "__acpp_sscp_sub_group_";

llvm::CallInst *asCollective(llvm::Instruction *I) {
  auto *Call = llvm::dyn_cast<llvm::CallInst>(I);
  auto *Callee = Call ? Call->getCalledFunction() : nullptr;
  return Callee && Callee->getName().starts_with(CollectivePrefix) ? Call : nullptr;
}

// Lanes that now call C without having reached it must not change the other lanes' result:
// the operand to replace by the identity for them (Identity == nullptr: none, e.g. a shuffle,
// whose value from such a lane is never trusted). Returns false if no identity is known.
bool getMask(llvm::CallInst &C, unsigned &ArgNo, llvm::Constant *&Identity) {
  using Kind = sgops::Collective::Kind;
  Identity = nullptr;
  auto Coll = sgops::parse(C.getCalledFunction()->getName().drop_front(BuiltinPrefix.size()));
  if (!Coll)
    return true; // e.g. the sub-group barrier: nothing to combine
  if (Coll->K == Kind::Any || Coll->K == Kind::All || Coll->K == Kind::None) {
    ArgNo = 0;
    Identity = llvm::ConstantInt::get(C.getArgOperand(0)->getType(), Coll->K == Kind::All);
  } else if (Coll->K == Kind::Reduce || Coll->K == Kind::InclusiveScan ||
             Coll->K == Kind::ExclusiveScan) {
    auto *Op = llvm::dyn_cast<llvm::ConstantInt>(C.getArgOperand(0));
    ArgNo = 1;
    Identity = Op ? sgops::identity(Op->getZExtValue(), Coll->Class, C.getArgOperand(1)->getType())
                  : nullptr;
    return Identity != nullptr;
  }
  return true;
}

class Masker {
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;
  llvm::LoopInfo LI;

  static bool keep(llvm::CallInst &C, const char *Why) {
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] kept " << C << ": " << Why << "\n";
    return false;
  }

  // the block whose branch decides whether B runs: nearest dominator B doesn't post-dominate
  llvm::BasicBlock *controllingBlock(llvm::BasicBlock *B) const {
    for (auto *N = DT.getNode(B)->getIDom(); N; N = N->getIDom())
      if (!PDT.dominates(B, N->getBlock()))
        return N->getBlock();
    return nullptr;
  }

  // the nearest branch deciding whether C runs, if it can be handled (null: none, or C is kept):
  // a two-way br in the same loop as C, not a loop exit (C keeps running once per iteration),
  // whose successor S on the way to C has no other predecessor
  llvm::BranchInst *controllingBranch(llvm::CallInst &C, llvm::BasicBlock *&S) {
    auto *B = C.getParent();
    llvm::BasicBlock *G = controllingBlock(B);
    if (!G)
      return nullptr; // every lane that runs the function reaches C
    auto *Br = llvm::dyn_cast<llvm::BranchInst>(G->getTerminator());
    if (!Br || !Br->isConditional())
      return keep(C, "controlled by a non-branch terminator"), nullptr;
    S = DT.dominates(Br->getSuccessor(0), B)   ? Br->getSuccessor(0)
        : DT.dominates(Br->getSuccessor(1), B) ? Br->getSuccessor(1)
                                               : nullptr;
    if (!S || S->getSinglePredecessor() != G)
      return keep(C, "branch target has other predecessors"), nullptr;
    auto *L = LI.getLoopFor(G);
    if (LI.getLoopFor(B) != L ||
        (L && (!L->contains(Br->getSuccessor(0)) || !L->contains(Br->getSuccessor(1)))))
      return keep(C, "branch is a loop exit, or C is in a deeper loop"), nullptr;
    return Br;
  }

public:
  explicit Masker(llvm::Function &F) : DT{F}, PDT{F}, LI{DT} {}

  // Takes C out of its nearest controlling `if`; called again for the next `if` around it:
  //   G: br c, S, K   S … B: pre; C; post      →   G: br c, S, J   S … B: pre
  //                                                J: C(c ? x : identity); br c, post, K
  // J is the new block holding C, reached from the end of `pre` and, through G's skip edge, by
  // the lanes that skip the `if`; after C every lane goes where it was going (`post` or K).
  // C takes along the collectives right after it (only pure code between them), so they stay one
  // sub-group step. The skipping lanes don't have the values of S … pre: they get zero there (the
  // identity for reductions, scans and votes), and use none of the results. Invalidates the
  // analyses: false if C has no controlling branch or can't be handled.
  bool maskOnce(llvm::CallInst &C) {
    llvm::BasicBlock *S = nullptr;
    auto *Br = controllingBranch(C, S);
    unsigned ArgNo = 0;
    llvm::Constant *Identity = nullptr;
    if (!Br)
      return false;
    if (!getMask(C, ArgNo, Identity))
      return keep(C, "no identity element for this operation");
    llvm::SmallVector<llvm::CallInst *, 4> Run{&C};
    for (auto *I = C.getNextNode(); !I->isTerminator(); I = I->getNextNode())
      if (auto *Call = asCollective(I); Call && getMask(*Call, ArgNo, Identity))
        Run.push_back(Call);
      else if (Call || I->mayReadOrWriteMemory() || !llvm::isSafeToSpeculativelyExecute(I))
        break;
    auto *G = Br->getParent();
    auto *B = C.getParent();
    const unsigned Skip = Br->getSuccessor(0) == S;
    auto *K = Br->getSuccessor(Skip);
    llvm::SmallVector<llvm::BasicBlock *, 16> Region; // may define values the skip path lacks
    DT.getDescendants(S, Region);

    auto *J = B->splitBasicBlock(&C, B->getName() + ".masked");
    auto *Post = J->splitBasicBlock(Run.back()->getNextNode(), B->getName() + ".post");
    J->getTerminator()->eraseFromParent();
    llvm::IRBuilder<> Builder{&C};
    llvm::Value *Taken = Skip ? Br->getCondition() : Builder.CreateNot(Br->getCondition());
    for (auto *Call : Run)
      if (getMask(*Call, ArgNo, Identity); Identity) {
        Builder.SetInsertPoint(Call);
        Call->setArgOperand(ArgNo, Builder.CreateSelect(Taken, Call->getArgOperand(ArgNo), Identity));
      }
    Builder.SetInsertPoint(J);
    Builder.CreateCondBr(Taken, Post, K);
    K->replacePhiUsesWith(G, J);
    Br->setSuccessor(Skip, J);

    llvm::DominatorTree NewDT{*B->getParent()};
    for (auto *X : Region)
      for (auto &I : *X) {
        llvm::SmallVector<llvm::Use *, 4> Uses;
        for (auto &U : I.uses())
          if (!NewDT.dominates(&I, U))
            Uses.push_back(&U);
        if (Uses.empty())
          continue;
        llvm::SSAUpdater SSA;
        SSA.Initialize(I.getType(), I.getName());
        SSA.AddAvailableValue(X, &I);
        SSA.AddAvailableValue(G, llvm::Constant::getNullValue(I.getType()));
        for (auto *U : Uses)
          SSA.RewriteUse(*U);
      }
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] masked " << Run.size()
                       << " collective(s) from " << C << " at " << G->getName() << "\n";
    return true;
  }
};

} // namespace

llvm::PreservedAnalyses MaskDivergentSubgroupCollectivesPass::run(llvm::Module &M,
                                                                  llvm::ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration() ||
        llvm::none_of(llvm::instructions(F), [](auto &I) { return asCollective(&I); }))
      continue;
    // to SSA first, as the vectorizer does: here the code between two collectives still goes
    // through private memory, which lanes skipping them must not write
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    FPM.addPass(llvm::EarlyCSEPass());
    FPM.addPass(llvm::InstCombinePass(llvm::InstCombineOptions().setVerifyFixpoint(false)));
    FPM.addPass(llvm::SimplifyCFGPass());
    FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    FPM.run(F, FAM);
    Changed = true;
    // in program order, so a collective feeding another one is masked first
    llvm::SmallVector<llvm::CallInst *, 16> Calls;
    for (auto *BB : llvm::ReversePostOrderTraversal<llvm::Function *>(&F))
      for (auto &I : *BB)
        if (auto *C = asCollective(&I))
          Calls.push_back(C);
    for (auto *C : Calls)
      while (Masker{F}.maskOnce(*C)) // fresh analyses after every change of the CFG
        ;
    FAM.invalidate(F, llvm::PreservedAnalyses::none());
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
