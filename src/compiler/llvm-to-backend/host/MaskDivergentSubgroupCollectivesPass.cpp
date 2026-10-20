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
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/SSAUpdater.h>

namespace hipsycl::compiler {
namespace {

constexpr llvm::StringLiteral BuiltinPrefix = "__acpp_sscp_";
constexpr llvm::StringLiteral CollectivePrefix = "__acpp_sscp_sub_group_";
constexpr llvm::StringLiteral VoteName = "__acpp_sscp_sub_group_any";

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

// The lanes that skip G's `if` don't have the values defined in it (Region): they get zero there
// (the identity for reductions, scans and votes), and use none of the results.
void repairSSA(llvm::ArrayRef<llvm::Instruction *> Region, llvm::BasicBlock *G) {
  llvm::DominatorTree DT{*G->getParent()};
  for (auto *I : Region) {
    llvm::SmallVector<llvm::Use *, 4> Uses;
    for (auto &U : I->uses())
      if (!DT.dominates(I, U))
        Uses.push_back(&U);
    if (Uses.empty())
      continue;
    llvm::SSAUpdater SSA;
    SSA.Initialize(I->getType(), I->getName());
    SSA.AddAvailableValue(I->getParent(), I);
    SSA.AddAvailableValue(G, llvm::Constant::getNullValue(I->getType()));
    for (auto *U : Uses)
      SSA.RewriteUse(*U);
  }
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
  // a two-way br whose successor S on the way to C has no other predecessor. A loop around C
  // counts as one block (the lanes that enter it together leave it together): its exit test is
  // passed over, and Inner is the outermost loop around C that the branch is outside of
  llvm::BranchInst *controllingBranch(llvm::CallInst &C, llvm::BasicBlock *&S,
                                      llvm::Loop *&Inner) {
    Inner = nullptr;
    for (auto *B = C.getParent();;) {
      llvm::BasicBlock *G = controllingBlock(B);
      if (!G)
        return nullptr; // every lane that runs the function reaches C
      auto *Br = llvm::dyn_cast<llvm::BranchInst>(G->getTerminator());
      if (!Br || !Br->isConditional())
        return keep(C, "controlled by a non-branch terminator"), nullptr;
      auto *L = LI.getLoopFor(G);
      if (L && (!L->contains(Br->getSuccessor(0)) || !L->contains(Br->getSuccessor(1)))) {
        if (!L->contains(B))
          return keep(C, "C is after the loop of its branch"), nullptr;
        B = L->getHeader(); // G exits a loop around C: go on from that loop's entry
        continue;
      }
      for (auto *X = LI.getLoopFor(B); X != L; X = X->getParentLoop())
        Inner = X;
      S = DT.dominates(Br->getSuccessor(0), B)   ? Br->getSuccessor(0)
          : DT.dominates(Br->getSuccessor(1), B) ? Br->getSuccessor(1)
                                                 : nullptr;
      if (!Inner && (!S || S->getSinglePredecessor() != G))
        return keep(C, "branch target has other predecessors"), nullptr;
      return Br;
    }
  }

  // the instructions from S on, which the lanes skipping the `if` around S won't have run
  llvm::SmallVector<llvm::Instruction *, 64> regionOf(llvm::BasicBlock *S) const {
    llvm::SmallVector<llvm::BasicBlock *, 16> Blocks;
    DT.getDescendants(S, Blocks);
    llvm::SmallVector<llvm::Instruction *, 64> Insts;
    for (auto *X : Blocks)
      for (auto &I : *X)
        Insts.push_back(&I);
    return Insts;
  }

  // take C out of its nearest `if`: every lane calls it, the lanes without t with a neutral value:
  //   if (t) { pre; C(x); post }   →   if (t) pre;  C(t ? x : identity);  if (t) post
  // (called again for the next `if` around C; an `if` around a loop holding C goes to enterLoop)
  bool enterLoop(llvm::CallInst &C, llvm::BranchInst &Br, llvm::Loop &L) {
    auto *P = L.getLoopPreheader(), *H = L.getHeader(), *Latch = L.getLoopLatch();
    auto *X = L.getExitingBlock(), *E = L.getExitBlock(), *G = Br.getParent();
    auto *XBr = X ? llvm::dyn_cast<llvm::BranchInst>(X->getTerminator()) : nullptr;
    if (!P || !Latch || !E || !XBr || !XBr->isConditional())
      return keep(C, "its loop has several exits");
    const unsigned Skip = DT.dominates(Br.getSuccessor(0), P);
    auto *S = Br.getSuccessor(!Skip), *K = Br.getSuccessor(Skip);
    if (!DT.dominates(S, P) || S->getSinglePredecessor() != G || L.contains(K))
      return keep(C, "branch target has other predecessors");
    auto Region = regionOf(S);
    auto &Ctx = H->getContext();
    auto *F = H->getParent();
    llvm::IRBuilder<> B{&Br};
    llvm::Value *Tk = Skip ? Br.getCondition() : B.CreateNot(Br.getCondition());

    Br.setSuccessor(Skip, P->splitBasicBlock(P->getTerminator(), P->getName() + ".all"));
    auto *HW = H->splitBasicBlock(H->getFirstNonPHIIt(), H->getName() + ".work");
    X = X == H ? HW : X;
    Latch = Latch == H ? HW : Latch;
    auto *V = llvm::BasicBlock::Create(Ctx, H->getName() + ".vote", F, E);
    H->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(HW, V, Tk, H);

    const bool OutFirst = !L.contains(XBr->getSuccessor(0));
    auto *In = XBr->getSuccessor(OutFirst);
    B.SetInsertPoint(XBr);
    llvm::Value *Stay = OutFirst ? B.CreateNot(XBr->getCondition()) : XBr->getCondition();
    XBr->eraseFromParent();
    llvm::BranchInst::Create(V, X);
    E->replacePhiUsesWith(X, V);
    B.SetInsertPoint(V);
    auto *Votes = B.CreatePHI(B.getInt1Ty(), 2);
    Votes->addIncoming(Stay, X);
    Votes->addIncoming(B.getFalse(), H);
    auto *Any = B.CreateCall(
        F->getParent()->getOrInsertFunction(VoteName, B.getInt1Ty(), B.getInt1Ty()), Votes);
    if (In == H) {
      H->replacePhiUsesWith(X, V);
      B.CreateCondBr(Any, H, E);
    } else {
      auto *T = llvm::BasicBlock::Create(Ctx, H->getName() + ".guard", F, E);
      auto *L2 = llvm::BasicBlock::Create(Ctx, H->getName() + ".latch", F, E);
      B.CreateCondBr(Any, T, E);
      In->replacePhiUsesWith(X, T);
      llvm::BranchInst::Create(In, L2, Tk, T);
      Latch->getTerminator()->replaceSuccessorWith(H, L2);
      H->replacePhiUsesWith(Latch, L2);
      llvm::BranchInst::Create(H, L2);
    }
    auto *EW = E->splitBasicBlock(E->getFirstNonPHIIt(), E->getName() + ".work");
    E->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(EW, K, Tk, E);
    K->replacePhiUsesWith(G, E);
    repairSSA(Region, G);
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] every lane enters the loop at "
                       << H->getName() << " for " << C << "\n";
    return true;
  }

public:
  explicit Masker(llvm::Function &F) : DT{F}, PDT{F}, LI{DT} {}

  // Takes C out of its nearest controlling `if`; called again for the next `if` around it:
  //   G: br c, S, K   S … B: pre; C; post      →   G: br c, S, J   S … B: pre
  //                                                J: C(c ? x : identity); br c, post, K
  // J is the new block holding C, reached from the end of `pre` and, through G's skip edge, by
  // the lanes that skip the `if`; after C every lane goes where it was going (`post` or K).
  // An `if` around a loop holding C is moved into the loop first (enterLoop). Invalidates the
  // analyses: false if C has no controlling branch or can't be handled.
  bool maskOnce(llvm::CallInst &C) {
    llvm::BasicBlock *S = nullptr;
    llvm::Loop *Inner = nullptr;
    auto *Br = controllingBranch(C, S, Inner);
    unsigned ArgNo = 0;
    llvm::Constant *Identity = nullptr;
    if (!Br)
      return false;
    if (!getMask(C, ArgNo, Identity))
      return keep(C, "no identity element for this operation");
    if (Inner)
      return enterLoop(C, *Br, *Inner);
    auto *G = Br->getParent();
    auto *B = C.getParent();
    const unsigned Skip = Br->getSuccessor(0) == S;
    auto *K = Br->getSuccessor(Skip);
    auto Region = regionOf(S);

    auto *J = B->splitBasicBlock(&C, B->getName() + ".masked");
    auto *Post = J->splitBasicBlock(C.getNextNode(), B->getName() + ".post");
    J->getTerminator()->eraseFromParent();
    llvm::IRBuilder<> Builder{&C};
    llvm::Value *Taken = Skip ? Br->getCondition() : Builder.CreateNot(Br->getCondition());
    if (Identity)
      C.setArgOperand(ArgNo, Builder.CreateSelect(Taken, C.getArgOperand(ArgNo), Identity));
    Builder.SetInsertPoint(J);
    Builder.CreateCondBr(Taken, Post, K);
    K->replacePhiUsesWith(G, J);
    Br->setSuccessor(Skip, J);
    repairSSA(Region, G);
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] masked " << C << " at "
                       << G->getName() << "\n";
    return true;
  }
};

} // namespace

llvm::PreservedAnalyses MaskDivergentSubgroupCollectivesPass::run(llvm::Module &M,
                                                                  llvm::ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool Changed = false;
  for (auto &F : M) {
    // in program order, so a collective feeding another one is masked first
    auto collectives = [&F] {
      llvm::SmallVector<llvm::CallInst *, 16> Calls;
      for (auto *BB : llvm::ReversePostOrderTraversal<llvm::Function *>(&F))
        for (auto &I : *BB)
          if (auto *C = asCollective(&I))
            Calls.push_back(C);
      return Calls;
    };
    if (F.isDeclaration() || collectives().empty())
      continue;
    bool FChanged = false;
    { // enterLoop wants loops with a preheader, one latch and dedicated exits
      llvm::DominatorTree DT{F};
      llvm::LoopInfo LI{DT};
      for (auto *L : llvm::SmallVector<llvm::Loop *, 8>{LI.begin(), LI.end()})
        FChanged |= llvm::simplifyLoop(L, &DT, &LI, nullptr, nullptr, nullptr, false);
    }
    for (bool Again = true; Again;) { // again for the votes enterLoop adds
      Again = false;
      for (auto *C : collectives())
        while (Masker{F}.maskOnce(*C)) // fresh analyses after every change of the CFG
          Again = FChanged = true;
    }
    if (FChanged)
      FAM.invalidate(F, llvm::PreservedAnalyses::none());
    Changed |= FChanged;
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
