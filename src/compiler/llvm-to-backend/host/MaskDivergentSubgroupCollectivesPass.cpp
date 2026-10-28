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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/SSAUpdater.h>

#include <optional>

namespace hipsycl::compiler {
namespace {

constexpr llvm::StringLiteral BuiltinPrefix = "__acpp_sscp_";
constexpr llvm::StringLiteral CollectivePrefix = "__acpp_sscp_sub_group_";
constexpr llvm::StringLiteral VoteName = "__acpp_sscp_sub_group_any";

llvm::CallInst *asCollective(llvm::Instruction *I) {
  auto *Call = llvm::dyn_cast<llvm::CallInst>(I);
  auto *Callee = Call ? Call->getCalledFunction() : nullptr;
  return Callee && llvmutils::starts_with(Callee->getName(), CollectivePrefix) ? Call : nullptr;
}

// a supported builtin: __acpp_sscp_sub_group_<kind>_<type>, or any / all / none
struct Collective {
  enum Kind {
    Shl, Shr, Permute, Select, Broadcast, Reduce, InclusiveScan, ExclusiveScan, Any, All, None
  };
  Kind K;
  char Class; // 'i' signed, 'u' unsigned, 'f' floating point, 'b' bool
};

std::optional<Collective> parseCollective(llvm::StringRef Name) { // e.g. "sub_group_shl_i32"
  if (!Name.consume_front("sub_group_"))
    return std::nullopt;
  if (Name == "any" || Name == "all" || Name == "none")
    return Collective{Name == "any" ? Collective::Any : Name == "all" ? Collective::All
                                                                      : Collective::None, 'b'};
  auto [Kind, Type] = Name.rsplit('_');
  if (Type.empty() || !llvm::StringRef("iuf").contains(Type[0]))
    return std::nullopt;
  auto K = llvm::StringSwitch<std::optional<Collective::Kind>>(Kind)
               .Case("shl", Collective::Shl)
               .Case("shr", Collective::Shr)
               .Case("permute", Collective::Permute)
               .Case("select", Collective::Select)
               .Case("broadcast", Collective::Broadcast)
               .Case("reduce", Collective::Reduce)
               .Case("inclusive_scan", Collective::InclusiveScan)
               .Case("exclusive_scan", Collective::ExclusiveScan)
               .Default(std::nullopt);
  if (!K)
    return std::nullopt;
  return Collective{*K, Type[0]};
}

// x op identity = x. Op: the __acpp_sscp_algorithm_op (plus, multiply, min, max, bit_and, bit_or,
// bit_xor, logical_and, logical_or). Null if there is none (bit operations on floating point).
llvm::Constant *getIdentity(unsigned Op, char Class, llvm::Type *T) {
  if (Class == 'f') {
    switch (Op) {
    case 0: return llvm::ConstantFP::getNegativeZero(T); // -0.0 + -0.0 = -0.0
    case 1:
    case 7: return llvm::ConstantFP::get(T, 1.0);
    case 2: return llvm::ConstantFP::getInfinity(T, /*Negative*/ false);
    case 3: return llvm::ConstantFP::getInfinity(T, /*Negative*/ true);
    case 8: return llvm::ConstantFP::get(T, 0.0);
    default: return nullptr;
    }
  }
  const unsigned W = T->getIntegerBitWidth();
  const bool Signed = Class == 'i';
  switch (Op) {
  case 1:
  case 7: return llvm::ConstantInt::get(T, 1); // multiply, logical and
  case 2:
    return llvm::ConstantInt::get(T, Signed ? llvm::APInt::getSignedMaxValue(W)
                                            : llvm::APInt::getMaxValue(W));
  case 3:
    return llvm::ConstantInt::get(T, Signed ? llvm::APInt::getSignedMinValue(W)
                                            : llvm::APInt::getMinValue(W));
  case 4: return llvm::ConstantInt::get(T, llvm::APInt::getAllOnes(W));
  default: return llvm::ConstantInt::get(T, 0); // plus, or, xor, logical or
  }
}

// The lanes that skipped C's `if` now call C too, and must not change the other lanes' result:
// ArgNo is the argument they pass as Identity (nullptr: none, e.g. a shuffle, whose value from
// such a lane is undefined in SYCL anyway). False if no identity is known.
bool getMask(llvm::CallInst &C, unsigned &ArgNo, llvm::Constant *&Identity) {
  using Kind = Collective::Kind;
  Identity = nullptr;
  auto Coll = parseCollective(C.getCalledFunction()->getName().drop_front(BuiltinPrefix.size()));
  if (!Coll)
    return true; // e.g. the sub-group barrier: nothing to combine
  if (Coll->K == Kind::Any || Coll->K == Kind::All || Coll->K == Kind::None) {
    ArgNo = 0;
    Identity = llvm::ConstantInt::get(C.getArgOperand(0)->getType(), Coll->K == Kind::All);
  } else if (Coll->K == Kind::Reduce || Coll->K == Kind::InclusiveScan ||
             Coll->K == Kind::ExclusiveScan) {
    auto *Op = llvm::dyn_cast<llvm::ConstantInt>(C.getArgOperand(0));
    ArgNo = 1;
    Identity = Op ? getIdentity(Op->getZExtValue(), Coll->Class, C.getArgOperand(1)->getType())
                  : nullptr;
    return Identity != nullptr;
  }
  return true;
}

// The lanes that skip BR's `if` now pass code that uses values defined in it (Region), which they
// don't have: they get zero, and use none of the results (C's argument is Identity for them).
void repairSSA(llvm::ArrayRef<llvm::Instruction *> Region, llvm::BasicBlock *BR) {
  llvm::DominatorTree DT{*BR->getParent()};
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
    SSA.AddAvailableValue(BR, llvm::Constant::getNullValue(I->getType()));
    for (auto *U : Uses)
      SSA.RewriteUse(*U);
  }
}

// Block names: BR ends with Branch, the `if` around C, which sends the lanes with Taken to UB (on
// the way to C) and the others to SKIP. CB holds C; maskOnce splits it at C: CB keeps the code
// before C (PRE), COLL holds C alone, POST the rest.
class Masker {
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;
  llvm::LoopInfo LI;

  static bool keep(llvm::CallInst &C, const char *Why) {
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] kept " << C << ": " << Why << "\n";
    return false;
  }

  // the block whose branch decides whether CB runs: nearest dominator CB doesn't post-dominate
  llvm::BasicBlock *controllingBlock(llvm::BasicBlock *CB) const {
    for (auto *N = DT.getNode(CB)->getIDom(); N; N = N->getIDom())
      if (!PDT.dominates(CB, N->getBlock()))
        return N->getBlock();
    return nullptr;
  }

  // the nearest branch deciding whether C runs, if it can be handled (null: none, or C is kept):
  // a two-way br whose successor UB on the way to C has no other predecessor. A loop around C
  // counts as one block (the lanes that enter it together leave it together): its exit test is
  // passed over, and Inner is the outermost loop around C that the branch is outside of
  llvm::BranchInst *controllingBranch(llvm::CallInst &C, llvm::BasicBlock *&UB,
                                      llvm::Loop *&Inner) {
    Inner = nullptr;
    for (auto *CB = C.getParent();;) {
      llvm::BasicBlock *BR = controllingBlock(CB);
      if (!BR)
        return nullptr; // every lane that runs the function reaches C
      auto *Branch = llvm::dyn_cast<llvm::BranchInst>(BR->getTerminator());
      if (!Branch || !Branch->isConditional())
        return keep(C, "controlled by a non-branch terminator"), nullptr;
      auto *L = LI.getLoopFor(BR);
      if (L && (!L->contains(Branch->getSuccessor(0)) || !L->contains(Branch->getSuccessor(1)))) {
        if (!L->contains(CB))
          return keep(C, "C is after the loop of its branch"), nullptr;
        CB = L->getHeader(); // BR exits a loop around C: go on from that loop's entry
        continue;
      }
      for (auto *X = LI.getLoopFor(CB); X != L; X = X->getParentLoop())
        Inner = X;
      UB = DT.dominates(Branch->getSuccessor(0), CB)   ? Branch->getSuccessor(0)
           : DT.dominates(Branch->getSuccessor(1), CB) ? Branch->getSuccessor(1)
                                                       : nullptr;
      if (!Inner && (!UB || UB->getSinglePredecessor() != BR))
        return keep(C, "branch target has other predecessors"), nullptr;
      return Branch;
    }
  }

  // the values defined from UB on, which the lanes skipping the `if` around UB won't have; no
  // instructions without a value: enterLoop erases branches in the region before repairSSA
  llvm::SmallVector<llvm::Instruction *, 64> regionOf(llvm::BasicBlock *UB) const {
    llvm::SmallVector<llvm::BasicBlock *, 16> Blocks;
    DT.getDescendants(UB, Blocks);
    llvm::SmallVector<llvm::Instruction *, 64> Insts;
    for (auto *X : Blocks)
      for (auto &I : *X)
        if (!I.getType()->isVoidTy())
          Insts.push_back(&I);
    return Insts;
  }

  // C sits in L, a loop inside Branch's `if`: if (Taken) { for (...) { C } }. Moves the `if` into
  // L: every lane enters L, which runs while any lane votes to stay; a lane runs an iteration only
  // while active (Act). C is then under `if (Act)`, which maskOnce masks next.
  bool enterLoop(llvm::CallInst &C, llvm::BranchInst &Branch, llvm::Loop &L) {
    auto *P = L.getLoopPreheader(), *H = L.getHeader(), *Latch = L.getLoopLatch();
    auto *X = L.getExitingBlock(), *E = L.getExitBlock(), *BR = Branch.getParent();
    auto *XBr = X ? llvm::dyn_cast<llvm::BranchInst>(X->getTerminator()) : nullptr;
    if (!P || !Latch || !E || !XBr || !XBr->isConditional() || !DT.dominates(X, Latch))
      return keep(C, "its loop has several exits, or an exit test not on every iteration");
    const unsigned SkipIdx = DT.dominates(Branch.getSuccessor(0), P);
    auto *UB = Branch.getSuccessor(!SkipIdx), *SKIP = Branch.getSuccessor(SkipIdx);
    if (!DT.dominates(UB, P) || UB->getSinglePredecessor() != BR || L.contains(SKIP))
      return keep(C, "branch target has other predecessors");
    auto Region = regionOf(UB);
    auto &Ctx = H->getContext();
    auto *F = H->getParent();
    llvm::IRBuilder<> B{&Branch};
    llvm::Value *Taken = SkipIdx ? Branch.getCondition() : B.CreateNot(Branch.getCondition());

    // entry: the lanes without Taken enter L too (BR → P.all, was BR → SKIP)
    auto *PA = P->splitBasicBlock(P->getTerminator(), P->getName() + ".all");
    Branch.setSuccessor(SkipIdx, PA);
    // header: H: br Act, H.work, V; Act is Taken on entry, then the lane's own vote
    auto *HW = H->splitBasicBlock(H->getFirstNonPHI(), H->getName() + ".work");
    X = X == H ? HW : X;
    Latch = Latch == H ? HW : Latch;
    auto *V = llvm::BasicBlock::Create(Ctx, H->getName() + ".vote", F, E);
    B.SetInsertPoint(H->getTerminator());
    auto *Act = B.CreatePHI(B.getInt1Ty(), 2);
    Act->addIncoming(Taken, PA);
    H->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(HW, V, Act, H);

    // exit test: X: br V; V: Any = vote(Votes), Votes: Stay for active lanes, else false
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
    if (In == H) { // exit test at the end of the body: V: br Any, H, E
      H->replacePhiUsesWith(X, V);
      Act->addIncoming(Votes, V);
      B.CreateCondBr(Any, H, E);
    } else { // in front of it: V: br Any, T, E; T: br Votes, In, L2 (the body if this lane stays)
      auto *T = llvm::BasicBlock::Create(Ctx, H->getName() + ".guard", F, E);
      auto *L2 = llvm::BasicBlock::Create(Ctx, H->getName() + ".latch", F, E);
      B.CreateCondBr(Any, T, E);
      In->replacePhiUsesWith(X, T);
      llvm::BranchInst::Create(In, L2, Votes, T);
      Latch->getTerminator()->replaceSuccessorWith(H, L2);
      H->replacePhiUsesWith(Latch, L2);
      Act->addIncoming(Votes, L2);
      llvm::BranchInst::Create(H, L2);
    }
    // exit: E: br Taken, E.work, SKIP
    auto *EW = E->splitBasicBlock(E->getFirstNonPHI(), E->getName() + ".work");
    E->getTerminator()->eraseFromParent();
    llvm::BranchInst::Create(EW, SKIP, Taken, E);
    SKIP->replacePhiUsesWith(BR, E);
    repairSSA(Region, BR);
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] every lane enters the loop at "
                       << H->getName() << " for " << C << "\n";
    return true;
  }

public:
  explicit Masker(llvm::Function &F) : DT{F}, PDT{F}, LI{DT} {}

  // Takes C out of its nearest `if`, so every lane calls it; called again for the next `if`.
  // A loop between the `if` and C goes to enterLoop. False: C has no `if`, or is kept.
  bool maskOnce(llvm::CallInst &C) {
    llvm::BasicBlock *UB = nullptr;
    llvm::Loop *Inner = nullptr;
    auto *Branch = controllingBranch(C, UB, Inner);
    unsigned ArgNo = 0;
    llvm::Constant *Identity = nullptr;
    if (!Branch)
      return false;
    if (!getMask(C, ArgNo, Identity))
      return keep(C, "no identity element for this operation");
    if (Inner)
      return enterLoop(C, *Branch, *Inner);
    auto *BR = Branch->getParent();
    auto *CB = C.getParent();
    const unsigned SkipIdx = Branch->getSuccessor(0) == UB;
    auto *SKIP = Branch->getSuccessor(SkipIdx);
    auto Region = regionOf(UB);

    // split CB at C: CB keeps PRE, COLL holds C, POST the rest
    auto *COLL = CB->splitBasicBlock(&C, CB->getName() + ".masked");
    auto *POST = COLL->splitBasicBlock(C.getNextNode(), CB->getName() + ".post");
    COLL->getTerminator()->eraseFromParent();
    // COLL: C with argument ArgNo = Taken ? its value : Identity; br Taken, POST, SKIP
    llvm::IRBuilder<> Builder{&C};
    llvm::Value *Taken =
        SkipIdx ? Branch->getCondition() : Builder.CreateNot(Branch->getCondition());
    if (Identity)
      C.setArgOperand(ArgNo, Builder.CreateSelect(Taken, C.getArgOperand(ArgNo), Identity));
    Builder.SetInsertPoint(COLL);
    Builder.CreateCondBr(Taken, POST, SKIP);
    // the lanes without Taken: BR → COLL → SKIP (was BR → SKIP)
    SKIP->replacePhiUsesWith(BR, COLL);
    Branch->setSuccessor(SkipIdx, COLL);
    repairSSA(Region, BR);
    HIPSYCL_DEBUG_INFO << "[MaskDivergentSubgroupCollectives] masked " << C << " at "
                       << BR->getName() << "\n";
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
    // only two-way branches are masked: switches (which optimizers build from compares of one
    // value, differently per LLVM version) become chains of them
    bool FChanged = !llvm::LowerSwitchPass{}.run(F, FAM).areAllPreserved();
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
