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
#include "hipSYCL/compiler/cbs/SubgroupSteps.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/SubgroupOps.hpp"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <cstring>
#include <optional>

namespace hipsycl::compiler {
using namespace llvm;

using namespace sgops;

namespace {
// steps 2 (group, expandBlock) and 3 (expand) for one kernel
class StepBuilder {
  Function &F;
  SplitterAnnotationInfo &SAA;
  std::size_t S;

  // flattened local id, x fastest: the index of CBS's lane loops (CBS maps the id loads to each
  // lane's ids). Loaded here, not in the entry block: barrier canonicalization moves that code.
  Value *linearId(IRBuilder<> &B) {
    auto &M = *F.getParent();
    auto *SizeT = M.getDataLayout().getLargestLegalIntType(F.getContext());
    auto Load = [&](const char *Name) { return B.CreateLoad(SizeT, M.getOrInsertGlobal(Name, SizeT)); };
    Value *Id[3], *Size[2];
    for (int D = 0; D < 3; ++D)
      Id[D] = Load(cbs::LocalIdGlobalNames[D]);
    for (int D = 0; D < 2; ++D)
      Size[D] = Load(cbs::LocalSizeGlobalNames[D]);
    return B.CreateAdd(Id[0], B.CreateMul(Size[0], B.CreateAdd(Id[1], B.CreateMul(Size[1], Id[2]))),
                       "sg.lin");
  }

  // one slot per lane: a sub-group runs the span before the barrier, then its step and the span
  // after, with no other sub-group in between (both are sub-group spans), so one sub-group at a
  // time uses the array. Marked arrayified: CBS doesn't arrayify it again; as sub-group shared,
  // not loop state: CBS doesn't keep a value loaded from it there across later barriers.
  AllocaInst *array(Type *T) {
    if (T->isIntegerTy(1)) // bools as bytes: a vector of i1 in memory would be bit-packed
      T = Type::getInt8Ty(F.getContext());
    IRBuilder<> EB(F.getEntryBlock().getTerminator());
    auto *A = EB.CreateAlloca(T, EB.getInt64(S), "sg.array");
    A->setAlignment(Align(DefaultAlignment));
    auto &Ctx = F.getContext();
    A->setMetadata(MDKind::Arrayified, MDNode::get(Ctx, {MDString::get(Ctx, MDKind::SubgroupShared)}));
    return A;
  }

  // step 3, expand. Group: collectives of one block in order, none using another's result. Each
  // lane stores its inputs to arrays (slot = lane), then ONE sub-group barrier, a step marker per
  // collective, and each lane loads its result from out[lane]; the calls are erased.
  void expand(ArrayRef<CallInst *> Group) {
    CallInst *Last = Group.back();
    IRBuilder<> B(Last);
    auto Lane = [&] { // lane in the sub-group: flattened local id mod S
      Value *Lin = linearId(B);
      return B.CreateURem(Lin, ConstantInt::get(Lin->getType(), S), "sg.lane");
    };
    Value *L = Lane();
    SmallVector<std::pair<AllocaInst *, SmallVector<Value *, 4>>, 4> Steps; // result array, marker args
    for (auto *C : Group) {
      auto *Out = array(C->getType());
      SmallVector<Value *, 4> Args{Out};
      for (Value *A : C->args()) {
        if (isa<Constant>(A)) {
          Args.push_back(A);
          continue;
        }
        auto *In = array(A->getType());
        B.CreateStore(A->getType()->isIntegerTy(1) ? B.CreateZExt(A, In->getAllocatedType()) : A,
                      B.CreateInBoundsGEP(In->getAllocatedType(), In, L));
        Args.push_back(In);
      }
      Steps.push_back({Out, Args});
    }
    utils::createBarrier(Last, SAA, GroupTier::SubGroup); // before Last, after the stores
    B.SetInsertPoint(Last);
    for (size_t I = 0; I < Group.size(); ++I) {
      auto Name = Group[I]->getCalledFunction()->getName().drop_front(std::strlen(KeptCollectivePrefix));
      SmallVector<Type *, 4> Types;
      for (auto *A : Steps[I].second)
        Types.push_back(A->getType());
      B.CreateCall(F.getParent()->getOrInsertFunction((StepMarkerPrefix + Name).str(),
                                                      FunctionType::get(B.getVoidTy(), Types, false)),
                   Steps[I].second);
    }
    L = Lane(); // recomputed: nothing per lane crosses the barrier
    for (size_t I = 0; I < Group.size(); ++I) {
      auto *Out = Steps[I].first;
      Value *Y = B.CreateLoad(Out->getAllocatedType(),
                              B.CreateInBoundsGEP(Out->getAllocatedType(), Out, L));
      if (Group[I]->getType()->isIntegerTy(1))
        Y = B.CreateICmpNE(Y, ConstantInt::get(Out->getAllocatedType(), 0));
      Group[I]->replaceAllUsesWith(Y);
      Group[I]->eraseFromParent();
    }
  }

public:
  StepBuilder(Function &F, SplitterAnnotationInfo &SAA, std::size_t S) : F{F}, SAA{SAA}, S{S} {}

  // step 2, group. Calls: the kept collectives of one block in order. A collective joins the
  // current group unless it uses a member's result. Pure lane code that uses a result moves after
  // the last member (no collective touches memory); a dependent load, store or call, or a
  // barrier, ends the group.
  void expandBlock(ArrayRef<CallInst *> Calls) {
    for (size_t I = 0; I < Calls.size();) {
      SmallVector<CallInst *, 4> Group{Calls[I]};
      SmallPtrSet<Instruction *, 16> Dep{Calls[I]};
      SmallVector<Instruction *, 16> Sink;
      auto UsesDep = [&](Instruction *X) {
        return any_of(X->operands(), [&](Value *Op) {
          auto *OI = dyn_cast<Instruction>(Op);
          return OI && Dep.contains(OI);
        });
      };
      size_t J = I + 1;
      for (auto *X = Calls[I]->getNextNode(); X && !X->isTerminator(); X = X->getNextNode()) {
        if (J < Calls.size() && X == Calls[J]) {
          if (UsesDep(X))
            break;
          Group.push_back(Calls[J++]);
          Dep.insert(X);
        } else if (utils::isBarrier(X, SAA)) {
          break;
        } else if (UsesDep(X)) {
          if (X->mayHaveSideEffects() || X->mayReadFromMemory())
            break;
          Dep.insert(X);
          Sink.push_back(X);
        }
      }
      auto *AfterLast = Group.back()->getNextNode();
      for (auto *X : Sink)
        if (X->comesBefore(Group.back()))
          X->moveBefore(AfterLast);
      HIPSYCL_DEBUG_INFO << "[SubgroupSteps] " << F.getName() << ": " << Group.size()
                         << " collective(s), one sub-group step\n";
      expand(Group);
      I = J;
    }
  }
};
} // namespace

// step 1, keep: supported collectives in kernels call __acpp_cbs_sg.<builtin>, never defined, so
// linking doesn't inline the builtins' per-lane bodies; convergent: O1 doesn't move them under
// more branches
bool keepSubgroupCollectives(Module &M, ArrayRef<std::string> KernelNames) {
  SmallVector<CallInst *, 16> Calls;
  for (const auto &Name : KernelNames)
    if (auto *K = M.getFunction(Name))
      for (auto &I : instructions(*K))
        if (auto *C = dyn_cast<CallInst>(&I); C && C->getCalledFunction())
          Calls.push_back(C);
  bool Changed = false;
  for (auto *C : Calls) {
    StringRef Name = C->getCalledFunction()->getName();
    if (!Name.consume_front("__acpp_sscp_"))
      continue;
    auto Coll = parse(Name);
    if (!Coll || !isSupported(*Coll, *C))
      continue;
    auto Kept = M.getOrInsertFunction((KeptCollectivePrefix + Name).str(), C->getFunctionType());
    auto *F = cast<Function>(Kept.getCallee());
    F->addFnAttr(Attribute::Convergent);
    F->addFnAttr(Attribute::NoUnwind);
    C->setCalledFunction(Kept);
    Changed = true;
  }
  return Changed;
}

// steps 2 and 3 for each block with kept collectives
PreservedAnalyses SubgroupStepsPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F))
    return PreservedAnalyses::all();
  MapVector<BasicBlock *, SmallVector<CallInst *, 4>> PerBlock;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I); C && C->getCalledFunction() &&
                                            C->getCalledFunction()->getName().starts_with(KeptCollectivePrefix))
        PerBlock[&BB].push_back(C);
  if (PerBlock.empty())
    return PreservedAnalyses::all();
  StepBuilder SB{F, *SAA, S};
  for (auto &[BB, Calls] : PerBlock)
    SB.expandBlock(Calls);
  PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}

// step 4, lower: masked vector loads of the input arrays, the operation, a masked store
void emitSubgroupStep(CallInst &Marker, Instruction *InsertBefore, Value *Begin, Value *End,
                      unsigned S) {
  IRBuilder<> B(InsertBefore);
  const auto C = *parse(Marker.getCalledFunction()->getName().drop_front(std::strlen(StepMarkerPrefix)));
  Value *N = B.CreateSub(End, Begin, "sg.n"); // lanes of this sub-group (< S: the last, partial)
  Value *Active = B.CreateICmpULT(laneIds(N->getType(), S), B.CreateVectorSplat(S, N), "sg.active");

  // marker arguments: the result array, then the builtin's, each per-lane one as its array
  auto Arg = [&](unsigned I) -> Value * {
    Value *A = Marker.getArgOperand(I + 1);
    auto *Arr = dyn_cast<AllocaInst>(A);
    if (!Arr)
      return A; // a constant
    auto *VecT = FixedVectorType::get(Arr->getAllocatedType(), S);
    Value *V = B.CreateMaskedLoad(VecT, Arr, Arr->getAlign(), Active, PoisonValue::get(VecT));
    return C.Class == 'b' ? B.CreateICmpNE(V, Constant::getNullValue(VecT)) : V;
  };
  auto Vec = [&](Value *V) { return V->getType()->isVectorTy() ? V : B.CreateVectorSplat(S, V); };
  auto Op = [&] { return unsigned(cast<ConstantInt>(Marker.getArgOperand(1))->getZExtValue()); };

  Value *R = nullptr;
  switch (C.K) {
  case Collective::Shl:
  case Collective::Shr:
  case Collective::Permute:
  case Collective::Select: R = shuffle(B, C.K, Vec(Arg(0)), Arg(1), S); break;
  case Collective::Broadcast: R = shuffle(B, Collective::Select, Vec(Arg(1)), Arg(0), S); break;
  case Collective::Reduce: R = reduce(B, Op(), C.Class, Vec(Arg(1)), N, S); break;
  case Collective::InclusiveScan: R = inclusiveScan(B, Op(), C.Class, Vec(Arg(1)), S); break;
  case Collective::ExclusiveScan: R = exclusiveScan(B, Op(), C.Class, Vec(Arg(1)), Arg(2), S); break;
  default: { // any / all / none: only active lanes count
    Value *P = Vec(Arg(0));
    Value *Any = B.CreateOrReduce(B.CreateSelect(Active, P, ConstantInt::getFalse(P->getType())));
    Value *All = B.CreateAndReduce(B.CreateSelect(Active, P, ConstantInt::getTrue(P->getType())));
    R = B.CreateVectorSplat(S, C.K == Collective::Any   ? Any
                               : C.K == Collective::All ? All
                                                        : B.CreateNot(Any));
  }
  }

  auto *Out = cast<AllocaInst>(Marker.getArgOperand(0));
  if (R->getType()->getScalarType()->isIntegerTy(1))
    R = B.CreateZExt(R, FixedVectorType::get(Out->getAllocatedType(), S));
  B.CreateMaskedStore(R, Out, Out->getAlign(), Active);
  Marker.eraseFromParent();
}
} // namespace hipsycl::compiler
