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
#include "hipSYCL/compiler/llvm-to-backend/host/SubgroupVectorizerPass.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SubgroupOps.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"

#include <rv/PlatformInfo.h>
#include <rv/config.h>
#include <rv/passes/loopExitCanonicalizer.h>
#include <rv/region/FunctionRegion.h>
#include <rv/region/Region.h>
#include <rv/rv.h>
#include <rv/utils.h>
#include <rv/vectorMapping.h>
#include <rv/vectorizationInfo.h>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Verifier.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/Scalarizer.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/BreakCriticalEdges.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <functional>
#include <memory>
#include <optional>

namespace hipsycl::compiler {
namespace {
using namespace llvm;

// arguments the per-lane function gets after the kernel's: sg_first (linear local id of lane 0),
// group ids x y z, numbers of groups x y z
constexpr unsigned NumIdArgs = 7;
constexpr char CollectivePrefix[] = "acpp_rv_sg."; // mapped collectives (KernelVectorizer::collective)

class KernelVectorizer {
public:
  KernelVectorizer(Function &K, unsigned S, std::array<uint64_t, 3> L)
      : K{K}, M{*K.getParent()}, S{S}, L{L}, W{L[0] * L[1] * L[2]} {}

  // false, with Why, if K stays scalar
  bool run(FunctionAnalysisManager &FAM);
  std::string Why;

private:
  Function &K;
  Module &M;
  const unsigned S;
  const std::array<uint64_t, 3> L; // work-group size x, y, z
  const uint64_t W;                // work-group size; the last sub-group has W - (W/S)·S lanes if not 0
  std::vector<rv::VectorMapping> Mappings; // of the collectives' vector functions
  SmallVector<CallInst *, 8> UniformCalls; // with side effects, still once per sub-group

  Value *collective(IRBuilder<> &B, const sgops::Collective &Coll, CallInst &C);
  std::array<Value *, 3> localIds(IRBuilder<> &B, Value *SgFirst);
  bool rewriteBuiltins(Function &Lane);
  void guardMissingLanes(Function &Lane);
  void alignLaneCopies(Function &Lane);
  SmallVector<Instruction *, 16> laneInvariantValues(Function &Lane);
  unsigned splitJoins(Function &Lane);
  unsigned lowerPrivateMemsets(Function &Lane);
  std::array<unsigned, 4> windowAccesses(Function &V);
  void replaceBody(Function &V);
  void erase(Function *F, FunctionAnalysisManager &FAM) {
    FAM.clear(*F, F->getName());
    F->eraseFromParent();
  }
};

// A sub-group collective C as a call of `T acpp_rv_sg.<builtin>[.<op> | .c<constant>](per-lane args)`,
// which RV maps to a masked vector function built with SubgroupOps.hpp: for
// reduce and scans inactive lanes contribute the identity, then the builtins' tree order (results
// as on the scalar path); a constant shuffle operand is part of the function, so it is one
// shufflevector. RV maps every resolved call result as a vector, so reduce returns a splat.
Value *KernelVectorizer::collective(IRBuilder<> &B, const sgops::Collective &Coll, CallInst &C) {
  using Kind = sgops::Collective::Kind;
  const bool HasOp = Coll.K == Kind::Reduce || Coll.K == Kind::InclusiveScan || Coll.K == Kind::ExclusiveScan;
  const unsigned Op = HasOp ? cast<ConstantInt>(C.getArgOperand(0))->getZExtValue() : 0;
  std::string Name = (CollectivePrefix + C.getCalledFunction()->getName().drop_front(12)).str();
  SmallVector<Value *, 2> Args; // per lane
  ConstantInt *Const = nullptr;
  if (HasOp) { // (op, x) or (op, x, init)
    Name += "." + std::to_string(Op);
    Args.append(C.arg_begin() + 1, C.arg_end());
  } else { // (x, delta / mask / id)
    Args.push_back(C.getArgOperand(0));
    if ((Const = dyn_cast<ConstantInt>(C.getArgOperand(1))))
      Name += ".c" + std::to_string(Const->getSExtValue());
    else
      Args.push_back(C.getArgOperand(1));
  }
  auto *F = M.getFunction(Name);
  if (!F) {
    SmallVector<Type *, 3> Tys, VecTys;
    for (auto *A : Args) {
      Tys.push_back(A->getType());
      VecTys.push_back(FixedVectorType::get(A->getType(), S));
    }
    VecTys.push_back(FixedVectorType::get(B.getInt1Ty(), S)); // the mask, last
    F = Function::Create(FunctionType::get(C.getType(), Tys, false), Function::ExternalLinkage, Name, M);
    auto *V = Function::Create(FunctionType::get(FixedVectorType::get(C.getType(), S), VecTys, false),
                               Function::InternalLinkage, Name + "_v", M);
    IRBuilder<> VB(BasicBlock::Create(M.getContext(), "entry", V));
    Value *X = V->getArg(0), *R;
    if (HasOp) {
      X = VB.CreateSelect(V->getArg(Args.size()), X,
                          ConstantVector::getSplat(ElementCount::getFixed(S),
                                                   sgops::identity(Op, Coll.Class, C.getType())));
      R = Coll.K == Kind::Reduce          ? sgops::reduce(VB, Op, Coll.Class, X, S)
          : Coll.K == Kind::InclusiveScan ? sgops::inclusiveScan(VB, Op, Coll.Class, X, S)
                                          : sgops::exclusiveScan(VB, Op, Coll.Class, X, V->getArg(1), S);
    } else
      R = sgops::shuffle(VB, Coll.K, X, Const ? static_cast<Value *>(Const) : V->getArg(1), S);
    VB.CreateRet(R);
    Mappings.emplace_back(F, V, S, /*maskPos*/ int(Args.size()), rv::VectorShape::varying(),
                          rv::VectorShapeVec(Args.size(), rv::VectorShape::varying()),
                          rv::CallPredicateMode::PredicateArg);
  }
  return B.CreateCall(F, Args);
}

// Local ids from sg_first + lane. A sub-group inside one x row (LX % S == 0) has x contiguous and
// y, z uniform; one inside an xy plane has z uniform; otherwise all are computed per lane.
std::array<Value *, 3> KernelVectorizer::localIds(IRBuilder<> &B, Value *SgFirst) {
  auto C = [&](uint64_t V) { return B.getInt64(V); };
  Value *Lane = B.CreateZExt(B.CreateCall(M.getOrInsertFunction("rv_lane_id", B.getInt32Ty())),
                             B.getInt64Ty(), "lane");
  const uint64_t LX = L[0], LY = L[1], LXY = L[0] * L[1];
  if (LX % S == 0)
    return {B.CreateAdd(B.CreateURem(SgFirst, C(LX)), Lane, "lid.x"),
            B.CreateURem(B.CreateUDiv(SgFirst, C(LX)), C(LY), "lid.y"),
            B.CreateUDiv(SgFirst, C(LXY), "lid.z")};
  const bool InPlane = LXY % S == 0;
  Value *P = B.CreateAdd(InPlane ? B.CreateURem(SgFirst, C(LXY)) : SgFirst, Lane);
  return {B.CreateURem(P, C(LX), "lid.x"), B.CreateURem(B.CreateUDiv(P, C(LX)), C(LY), "lid.y"),
          B.CreateUDiv(InPlane ? SgFirst : P, C(LXY), "lid.z")};
}

// Replaces the SSCP builtins in Lane, whose last NumIdArgs arguments are the ids. False if one is
// not supported (work-group collectives, sub-group ones other than below, unknown queries).
bool KernelVectorizer::rewriteBuiltins(Function &Lane) {
  Value *Arg[NumIdArgs];
  for (unsigned I = 0; I < NumIdArgs; ++I)
    Arg[I] = Lane.getArg(Lane.arg_size() - NumIdArgs + I);
  IRBuilder<> EB(&*Lane.getEntryBlock().getFirstInsertionPt());
  const auto Lid = localIds(EB, Arg[0]);

  // InstCombine turns `a + b` with disjoint bits into `or disjoint`, which RV's shape analysis
  // treats as varying (gathers instead of vector loads): back to add
  for (auto &I : make_early_inc_range(instructions(Lane)))
    if (auto *Or = dyn_cast<PossiblyDisjointInst>(&I); Or && Or->isDisjoint()) {
      auto *Add = BinaryOperator::CreateAdd(Or->getOperand(0), Or->getOperand(1), "",
                                            Or->getIterator());
      Add->takeName(Or);
      Or->replaceAllUsesWith(Add);
      Or->eraseFromParent();
    }

  SmallVector<CallInst *, 32> Calls;
  for (auto &I : instructions(Lane))
    if (auto *C = dyn_cast<CallInst>(&I); C && C->getCalledFunction())
      Calls.push_back(C);
  for (auto *C : Calls) {
    if (isa<AssumeInst>(C)) { // hints only; their conditions would be vectorized too
      C->eraseFromParent();
      continue;
    }
    StringRef N = C->getCalledFunction()->getName();
    if (!N.consume_front("__acpp_sscp_"))
      continue;
    IRBuilder<> B(C);
    Type *T = C->getType();
    const int Dim = N.ends_with("_x") ? 0 : N.ends_with("_y") ? 1 : 2;
    auto RV = [&](StringRef Name, ArrayRef<Value *> Args) {
      SmallVector<Type *, 2> Tys;
      for (auto *A : Args)
        Tys.push_back(A->getType());
      return B.CreateCall(M.getOrInsertFunction(Name, FunctionType::get(T, Tys, false)), Args);
    };
    Value *New = nullptr;
    if (N.starts_with("get_local_id_")) New = Lid[Dim];
    else if (N.starts_with("get_local_size_")) New = B.getInt64(L[Dim]);
    else if (N.starts_with("get_group_id_")) New = Arg[1 + Dim];
    else if (N.starts_with("get_num_groups_")) New = Arg[4 + Dim];
    else if (N == "get_subgroup_local_id") New = RV("rv_lane_id", {});
    else if (N == "get_subgroup_max_size") New = ConstantInt::get(T, S);
    else if (N == "get_subgroup_size") // the last sub-group may be partial: min(S, W - sg_first)
      New = W % S ? B.CreateTrunc(B.CreateBinaryIntrinsic(Intrinsic::umin, B.CreateSub(B.getInt64(W), Arg[0]),
                                                          B.getInt64(S)), T)
                  : ConstantInt::get(T, S);
    else if (N == "get_subgroup_id") New = B.CreateTrunc(B.CreateUDiv(Arg[0], B.getInt64(S)), T);
    else if (N == "get_num_subgroups") New = ConstantInt::get(T, (W + S - 1) / S);
    else if (N == "hint_subgroup_size" || N == "sub_group_barrier") { // a sub-group is one vector
      C->eraseFromParent();
      continue;
    } else if (N.consume_front("sub_group_broadcast_")) // (sender, x)
      New = RV(("rv_extract." + N).str(), {C->getArgOperand(1), C->getArgOperand(0)});
    else if (N == "sub_group_any") New = RV("rv_any", {C->getArgOperand(0)});
    else if (N == "sub_group_all") New = RV("rv_all", {C->getArgOperand(0)});
    else if (N == "sub_group_none") New = B.CreateNot(RV("rv_any", {C->getArgOperand(0)}));
    else if (auto Coll = sgops::parse(N); Coll && sgops::isSupported(*Coll, *C)) // shuffles, reduce, scans
      New = collective(B, *Coll, *C);
    else if (N == "rsqrt_f32" || N == "rsqrt_f64") // as the host libkernel: 1 / sqrt(x)
      New = B.CreateFDiv(ConstantFP::get(T, 1.0),
                         B.CreateUnaryIntrinsic(Intrinsic::sqrt, C->getArgOperand(0)));
    else if (N == "popcount_u32" || N == "popcount_u64")
      New = B.CreateUnaryIntrinsic(Intrinsic::ctpop, C->getArgOperand(0));
    else if (N == "work_group_barrier" || N == "get_dynamic_local_memory" ||
             N == "host_get_internal_local_memory") {
      UniformCalls.push_back(C);
      continue;
    } else if (!N.starts_with("work_group_") && !N.starts_with("sub_group_") &&
               !N.starts_with("get_"))
      continue; // math, atomics, ...: once per lane
    if (!New) {
      Why = (C->getCalledFunction()->getName() + " is not supported").str();
      return false;
    }
    C->replaceAllUsesWith(New);
    C->eraseFromParent();
  }
  return true;
}

// RV gives each lane its own copy of a private array, the S copies back to back, but keeps the
// accesses' alignment: a lane's copy is aligned only to gcd(alignment, size). Lower the alloca's
// and its accesses' alignment to that (else `int r[6] = {-1, ..}`, a 24-byte memset `align 16`,
// becomes aligned 16-byte stores on 8-byte aligned lanes: a fault).
void KernelVectorizer::alignLaneCopies(Function &Lane) {
  const auto &DL = M.getDataLayout();
  for (auto &I : Lane.getEntryBlock()) {
    auto *A = dyn_cast<AllocaInst>(&I);
    auto Size = A ? A->getAllocationSize(DL) : std::nullopt;
    if (!Size || Size->isScalable())
      continue;
    const Align LaneAlign = commonAlignment(A->getAlign(), Size->getFixedValue());
    if (LaneAlign == A->getAlign())
      continue;
    A->setAlignment(LaneAlign);
    SmallVector<Value *, 8> Ptrs{A}; // A and pointers derived from it
    while (!Ptrs.empty())
      for (auto *U : Ptrs.pop_back_val()->users())
        if (isa<GetElementPtrInst, CastInst>(U))
          Ptrs.push_back(U);
        else if (auto *L = dyn_cast<LoadInst>(U))
          L->setAlignment(std::min(L->getAlign(), LaneAlign));
        else if (auto *St = dyn_cast<StoreInst>(U))
          St->setAlignment(std::min(St->getAlign(), LaneAlign)); // also if A is the stored value: harmless
        else if (auto *MI = dyn_cast<MemIntrinsic>(U)) {
          MI->setDestAlignment(std::min(MI->getDestAlign().valueOrOne(), LaneAlign));
          if (auto *MT = dyn_cast<MemTransferInst>(MI))
            MT->setSourceAlignment(std::min(MT->getSourceAlign().valueOrOne(), LaneAlign));
        }
  }
}

// The values of an integer expression of only sg_first, the lane id and constants in every
// (sub-group k, lane l), at index k·S + l: W and S are known, sg_first = k·S. The lane id is
// rv_lane_id in Lane; in RV's output, constant vectors and splats. Arguments, loads, phis and
// other calls are not evaluated.
class LaneValues {
public:
  struct Values {
    SmallVector<APInt, 0> At; // per k·S + l
    bool Lane;                // depends on the lane id
  };
  LaneValues(Value *SgFirst, unsigned S, uint64_t W)
      : SgFirst{SgFirst}, S{S}, N{unsigned((W + S - 1) / S * S)} {}
  unsigned size() const { return N; }
  const Values *get(Value *X);

private:
  Value *SgFirst;
  const unsigned S, N;
  DenseMap<Value *, std::unique_ptr<Values>> Memo; // stable storage: callers keep pointers
};

const LaneValues::Values *LaneValues::get(Value *X) {
  if (auto It = Memo.find(X); It != Memo.end())
    return It->second.get();
  Memo[X] = nullptr; // also breaks cycles
  auto *VT = dyn_cast<FixedVectorType>(X->getType());
  auto *T = dyn_cast<IntegerType>(X->getType()->getScalarType());
  if (!T || T->getBitWidth() > 64 || (VT && VT->getNumElements() != S))
    return nullptr;
  const unsigned BW = T->getBitWidth();
  Values R{{}, false};
  auto *C = dyn_cast<CallInst>(X);
  if (auto *Const = dyn_cast<Constant>(X)) { // scalar, or element l for lane l
    for (unsigned J = 0; J < N; ++J) {
      auto *E = VT ? Const->getAggregateElement(J % S) : Const;
      if (isa_and_nonnull<UndefValue>(E)) // poison / undef: any value
        R.At.push_back(APInt(BW, 0));
      else if (auto *CI = dyn_cast_or_null<ConstantInt>(E))
        R.At.push_back(CI->getValue());
      else
        return nullptr;
    }
    R.Lane = VT && !Const->getSplatValue();
  } else if (X == SgFirst)
    for (unsigned J = 0; J < N; ++J)
      R.At.push_back(APInt(BW, J / S * S));
  else if (C && C->getCalledFunction() && C->getCalledFunction()->getName() == "rv_lane_id") {
    for (unsigned J = 0; J < N; ++J)
      R.At.push_back(APInt(BW, J % S));
    R.Lane = true;
  } else if (auto *Sh = dyn_cast<ShuffleVectorInst>(X)) { // lane l = lane Mask[l] of a ++ b
    auto *A = get(Sh->getOperand(0)), *B = get(Sh->getOperand(1));
    if (!A || !B)
      return nullptr;
    for (unsigned J = 0; J < N; ++J) {
      const int M = Sh->getMaskValue(J % S);
      R.At.push_back(M < 0 ? APInt(BW, 0) : (M < int(S) ? A : B)->At[J / S * S + M % S]);
    }
    R.Lane = A->Lane || B->Lane;
  } else if (auto *IE = dyn_cast<InsertElementInst>(X)) {
    auto *A = get(IE->getOperand(0)), *E = get(IE->getOperand(1));
    auto *Idx = dyn_cast<ConstantInt>(IE->getOperand(2));
    if (!A || !E || !Idx)
      return nullptr;
    for (unsigned J = 0; J < N; ++J)
      R.At.push_back(J % S == Idx->getZExtValue() ? E->At[J] : A->At[J]);
    R.Lane = A->Lane || E->Lane;
  } else if (auto *I = dyn_cast<Instruction>(X)) {
    SmallVector<const Values *, 3> Ops;
    for (Value *O : I->operands()) {
      auto *E = get(O);
      if (!E)
        return nullptr;
      Ops.push_back(E);
      R.Lane |= E->Lane;
    }
    for (unsigned J = 0; J < N; ++J) {
      auto A = [&](unsigned K) -> const APInt & { return Ops[K]->At[J]; };
      std::optional<APInt> V;
      if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        const bool Div = BO->isIntDivRem(), Shift = BO->isShift();
        if ((Div && A(1).isZero()) || (Shift && A(1).uge(BW)))
          return nullptr;
        switch (BO->getOpcode()) {
        case Instruction::Add: V = A(0) + A(1); break;
        case Instruction::Sub: V = A(0) - A(1); break;
        case Instruction::Mul: V = A(0) * A(1); break;
        case Instruction::UDiv: V = A(0).udiv(A(1)); break;
        case Instruction::SDiv: V = A(0).sdiv(A(1)); break;
        case Instruction::URem: V = A(0).urem(A(1)); break;
        case Instruction::SRem: V = A(0).srem(A(1)); break;
        case Instruction::Shl: V = A(0).shl(A(1)); break;
        case Instruction::LShr: V = A(0).lshr(A(1)); break;
        case Instruction::AShr: V = A(0).ashr(A(1)); break;
        case Instruction::And: V = A(0) & A(1); break;
        case Instruction::Or: V = A(0) | A(1); break;
        case Instruction::Xor: V = A(0) ^ A(1); break;
        default: return nullptr;
        }
      } else if (auto *Cast = dyn_cast<CastInst>(I)) {
        if (Cast->getOpcode() == Instruction::ZExt) V = A(0).zext(BW);
        else if (Cast->getOpcode() == Instruction::SExt) V = A(0).sext(BW);
        else if (Cast->getOpcode() == Instruction::Trunc) V = A(0).trunc(BW);
        else return nullptr;
      } else if (auto *Cmp = dyn_cast<ICmpInst>(I))
        V = APInt(1, ICmpInst::compare(A(0), A(1), Cmp->getPredicate()));
      else if (isa<SelectInst>(I))
        V = A(0).getBoolValue() ? A(1) : A(2);
      else
        return nullptr;
      R.At.push_back(*V);
    }
  } else
    return nullptr;
  return (Memo[X] = std::make_unique<Values>(std::move(R))).get();
}

// values that depend on the lane but are equal in all lanes of every sub-group, e.g.
// GROMACS's `tidx / 8` at S = 8 in 4x4 work-groups: RV has no periodic shape (lid.x = lane % 4)
// and calls them varying. Their values in every (sub-group, lane) (LaneValues) equal in all S
// lanes of every sub-group (missing lanes of a partial one included, so the missing-lane guard
// never qualifies) means uniform. Pinned uniform, RV keeps branches on them and computes them
// once, from lane 0 of their operands.
SmallVector<Instruction *, 16> KernelVectorizer::laneInvariantValues(Function &Lane) {
  LaneValues Values(Lane.getArg(K.arg_size()), S, W);
  SmallVector<Instruction *, 16> Invariant;
  for (auto &I : instructions(Lane)) {
    auto *E = Values.get(&I);
    if (!E || !E->Lane)
      continue;
    bool Same = true;
    for (unsigned J = 0; J < Values.size() && Same; ++J)
      Same = E->At[J] == E->At[J / S * S];
    if (Same)
      Invariant.push_back(&I);
  }
  return Invariant;
}

// problem:
// latch:  m_next = phi [m,  skip-A],          ← A not taken
//                      [m', B-not-taken],
//                      [m', B-taken]
// latch is the join of both A (uniform) and B (divergent). RV makes a phi in the join of a divergent
// branch varying unless all incoming values are equal (m != m'), so m_next is varying, although all
// lanes arrive from skip-A together (A is uniform) and both B paths bring the same m'. Through the
// loop, m and A become varying too.
// Split: B's two paths get their own join; there m' comes from both, so it needs no phi. latch then
// joins A only: m_next = phi [m, skip-A], [m', B-join] is the join of a uniform branch, so uniform,
// and m and A stay uniform through the loop.
unsigned KernelVectorizer::splitJoins(Function &Lane) {
  DominatorTree DT(Lane);
  LoopInfo LI(DT);
  unsigned Splits = 0;
  SmallVector<BasicBlock *, 32> Work;
  for (auto &BB : Lane)
    Work.push_back(&BB);
  while (!Work.empty()) {
    auto *J = Work.pop_back_val();
    auto *Node = DT.getNode(J);
    auto *L = LI.getLoopFor(J);
    if (!Node || !Node->getIDom() || pred_size(J) < 3 || !isa<PHINode>(J->front()) ||
        (L && L->getHeader() == J))
      continue;
    auto *D = Node->getIDom()->getBlock();
    for (auto *Succ : successors(D)) { // critical edges are split: each predecessor has J only
      SmallVector<BasicBlock *, 4> Preds;
      for (auto *P : predecessors(J))
        if (LI.getLoopFor(P) == L && DT.dominates(BasicBlockEdge(D, Succ), P))
          Preds.push_back(P);
      if (Preds.size() < 2 || Preds.size() == pred_size(J))
        continue;
      if (auto *New = SplitBlockPredecessors(J, Preds, ".join", &DT, &LI, nullptr, true)) {
        Work.push_back(New);
        ++Splits;
      }
    }
  }
  return Splits;
}

// Private arrays element-major. RV keeps one copy of a private array per lane (lane-major):
//   lane0: acc[0] acc[1] ... acc[7] | lane1: acc[0] ... acc[7] | ... | lane7: ...
// so acc[i] is a gather to read and per-lane stores (AVX2) or masked scatter (AVX512, SVE). RV's StructOpt
// can store it element-major instead, when every access has a uniform index:
//   acc[0]: lane0..lane7 | acc[1]: lane0..lane7 | ... | acc[7]: lane0..lane7
// then acc[i] is one vector load / masked vector store. StructOpt rejects arrays with a memset user,
// e.g. the zeroing `float acc[8] = {}`: a constant memset of a whole small array without padding
// becomes one store per element, so StructOpt accepts the array.
unsigned KernelVectorizer::lowerPrivateMemsets(Function &Lane) {
  const auto &DL = M.getDataLayout();
  unsigned Lowered = 0;
  for (auto &I : make_early_inc_range(instructions(Lane))) {
    auto *MS = dyn_cast<MemSetInst>(&I);
    auto *A = MS ? dyn_cast<AllocaInst>(MS->getDest()) : nullptr;
    auto *Len = MS ? dyn_cast<ConstantInt>(MS->getLength()) : nullptr;
    auto *Byte = MS ? dyn_cast<ConstantInt>(MS->getValue()) : nullptr;
    if (!A || !Len || !Byte || MS->isVolatile() ||
        Len->getZExtValue() != DL.getTypeAllocSize(A->getAllocatedType()))
      continue;
    IRBuilder<> B(MS);
    SmallVector<std::pair<Type *, SmallVector<Value *, 4>>, 16> Leaves; // element type, GEP indices
    SmallVector<Value *, 4> Idx{B.getInt64(0)};
    std::function<bool(Type *)> Collect = [&](Type *T) {
      if (auto *AT = dyn_cast<ArrayType>(T)) {
        for (uint64_t K = 0; K < AT->getNumElements(); ++K) {
          Idx.push_back(B.getInt64(K));
          if (!Collect(AT->getElementType()))
            return false;
          Idx.pop_back();
        }
        return true;
      }
      if (auto *ST = dyn_cast<StructType>(T)) {
        for (unsigned K = 0; K < ST->getNumElements(); ++K) {
          Idx.push_back(B.getInt32(K));
          if (!Collect(ST->getElementType(K)))
            return false;
          Idx.pop_back();
        }
        return true;
      }
      Leaves.emplace_back(T, Idx);
      return (T->isIntegerTy() || T->isFloatingPointTy()) && T->getPrimitiveSizeInBits() % 8 == 0 &&
             Leaves.size() <= 64;
    };
    if (!Collect(A->getAllocatedType()))
      continue;
    uint64_t Bytes = 0;
    for (auto &L : Leaves)
      Bytes += DL.getTypeStoreSize(L.first);
    if (Bytes != Len->getZExtValue())
      continue;
    for (auto &[T, LeafIdx] : Leaves) {
      APInt Bits = APInt::getSplat(T->getPrimitiveSizeInBits(), Byte->getValue().trunc(8));
      Constant *C = T->isIntegerTy() ? static_cast<Constant *>(ConstantInt::get(T, Bits))
                                     : ConstantFP::get(T, APFloat(T->getFltSemantics(), Bits));
      const uint64_t Offset = DL.getIndexedOffsetInType(A->getAllocatedType(), LeafIdx);
      B.CreateAlignedStore(C, B.CreateInBoundsGEP(A->getAllocatedType(), A, LeafIdx),
                           commonAlignment(A->getAlign(), Offset));
    }
    MS->eraseFromParent();
    ++Lowered;
  }
  return Lowered;
}

// Periodic and contiguous accesses. Each lane reads element i*4 + lid.x of GROMACS's xqib, an
// array of float4 {x, y, z, q}. At S = 8 in 4x4 work-groups lid.x = <0,1,2,3,0,1,2,3>, which RV
// calls varying, so it emits one gather per field:
//   x = gather(&x8, &x9, &x10, &x11, &x8, &x9, &x10, &x11),  y = gather(&y8, ...),  z, q likewise
// The lanes' addresses differ from lane 0's by known constants within a small range, so one load
// of that range, shuffled into one vector per field, replaces the 4 gathers:
//   memory  x8 y8 z8 q8 | x9 y9 z9 q9 | x10 y10 z10 q10 | x11 y11 z11 q11   (floats 0..15)
//   W = load 16 floats at &x8
//   x = shuffle(W, <0,4,8,12,0,4,8,12>),  y = shuffle(W, <1,5,9,13,...>),  z, q likewise
// The window holds at most 2S elements. Gathers share one if they have the same base and mask, in
// one block with no write between them; under a mask it loads only the elements active lanes read.
// A scatter writing distinct elements becomes a shuffle and one masked store.
std::array<unsigned, 4> KernelVectorizer::windowAccesses(Function &V) {
  const auto &DL = M.getDataLayout();
  LaneValues Values(V.getArg(K.arg_size()), S, W);
  // lane l's value minus lane 0's, the same in every sub-group; ExactU / ExactS: D as a signed
  // number is the exact difference of the values read as unsigned / signed (for zext / sext)
  struct LaneDiffs {
    SmallVector<APInt, 16> D;
    bool ExactU, ExactS;
  };
  DenseMap<Value *, std::optional<LaneDiffs>> Memo;
  std::function<std::optional<LaneDiffs>(Value *)> Diffs = [&](Value *X) -> std::optional<LaneDiffs> {
    if (auto It = Memo.find(X); It != Memo.end())
      return It->second;
    Memo[X] = std::nullopt;
    auto *VT = dyn_cast<FixedVectorType>(X->getType());
    auto *IT = dyn_cast<IntegerType>(X->getType()->getScalarType());
    if (!VT || !IT || VT->getNumElements() != S || IT->getBitWidth() > 64)
      return std::nullopt;
    const unsigned BW = IT->getBitWidth();
    auto Wide = [&](const APInt &A, bool Signed) { return Signed ? A.sext(BW + 1) : A.zext(BW + 1); };
    std::optional<LaneDiffs> R;
    if (getSplatValue(X)) // uniform
      R = LaneDiffs{SmallVector<APInt, 16>(S, APInt(BW, 0)), true, true};
    else if (auto *E = Values.get(X)) { // of sg_first, lane ids and constants: evaluated
      R = LaneDiffs{{}, true, true};
      for (unsigned J = 0; J < Values.size(); ++J) {
        const APInt &A = E->At[J], &A0 = E->At[J / S * S], D = A - A0;
        if (J < S)
          R->D.push_back(D);
        else if (D != R->D[J % S])
          return std::nullopt;
        R->ExactU &= Wide(A, false) - Wide(A0, false) == D.sext(BW + 1);
        R->ExactS &= Wide(A, true) - Wide(A0, true) == D.sext(BW + 1);
      }
    } else if (auto *I = dyn_cast<Instruction>(X)) {
      const unsigned Op = I->getOpcode();
      auto A = I->getNumOperands() ? Diffs(I->getOperand(0)) : std::nullopt;
      auto *C = I->getNumOperands() > 1 ? dyn_cast_or_null<ConstantInt>(getSplatValue(I->getOperand(1)))
                                        : nullptr;
      if (!A)
        ;
      else if (Op == Instruction::Add || Op == Instruction::Sub ||
               (Op == Instruction::Or && cast<PossiblyDisjointInst>(I)->isDisjoint())) {
        auto B = Diffs(I->getOperand(1)); // or disjoint: an add without carries
        const bool Or = Op == Instruction::Or;
        if (B) {
          R = LaneDiffs{{}, A->ExactU && B->ExactU && (Or || I->hasNoUnsignedWrap()),
                        A->ExactS && B->ExactS && (Or || I->hasNoSignedWrap())};
          for (unsigned L = 0; L < S; ++L) {
            bool Ov;
            R->D.push_back(Op == Instruction::Sub ? A->D[L].ssub_ov(B->D[L], Ov) : A->D[L].sadd_ov(B->D[L], Ov));
            R->ExactU &= !Ov;
            R->ExactS &= !Ov;
          }
        }
      } else if ((Op == Instruction::Shl && C && C->getValue().ult(BW)) || (Op == Instruction::Mul && C)) {
        R = LaneDiffs{{}, A->ExactU && I->hasNoUnsignedWrap(), A->ExactS && I->hasNoSignedWrap()};
        for (unsigned L = 0; L < S; ++L) {
          bool Ov;
          R->D.push_back(Op == Instruction::Shl ? A->D[L].sshl_ov(C->getValue(), Ov)
                                                : A->D[L].smul_ov(C->getValue(), Ov));
          R->ExactU &= !Ov;
          R->ExactS &= !Ov;
        }
      } else if (Op == Instruction::ZExt && (A->ExactU || (I->hasNonNeg() && A->ExactS))) {
        R = LaneDiffs{{}, true, true}; // non-negative in the wider type
        for (auto &D : A->D)
          R->D.push_back(D.sext(BW));
      } else if (Op == Instruction::SExt && A->ExactS) {
        R = LaneDiffs{{}, false, true};
        for (auto &D : A->D)
          R->D.push_back(D.sext(BW));
      } else if (Op == Instruction::Trunc) {
        R = LaneDiffs{{}, A->ExactU && I->hasNoUnsignedWrap(), A->ExactS && I->hasNoSignedWrap()};
        for (auto &D : A->D) {
          R->D.push_back(D.trunc(BW));
          R->ExactU &= R->D.back().sext(D.getBitWidth()) == D;
          R->ExactS &= R->D.back().sext(D.getBitWidth()) == D;
        }
      }
    }
    return Memo[X] = R;
  };

  // a gather / scatter: lane l's address = Root + Σ Key (index · stride) + Const + B[l], B[0] = 0
  struct Access {
    IntrinsicInst *Call;
    bool Gather;
    Value *Ptr, *Mask, *Root = nullptr;
    Type *T;
    Align A;
    SmallVector<std::pair<Value *, int64_t>, 4> Key; // non-constant indices, strides
    int64_t Const = 0;
    SmallVector<int64_t, 16> B;
  };
  std::function<bool(Value *, Access &)> Walk = [&](Value *P, Access &A) {
    if (Value *Sp = P->getType()->isVectorTy() ? getSplatValue(P) : P) {
      A.Root = Sp;
      return true;
    }
    auto *G = dyn_cast<GEPOperator>(P);
    if (!G || !Walk(G->getPointerOperand(), A))
      return false;
    for (auto GTI = gep_type_begin(G), E = gep_type_end(G); GTI != E; ++GTI) {
      Value *Idx = GTI.getOperand(), *Uni = Idx->getType()->isVectorTy() ? getSplatValue(Idx) : Idx;
      auto *CI = dyn_cast_or_null<ConstantInt>(Uni);
      if (GTI.isStruct()) {
        if (!CI)
          return false;
        A.Const += DL.getStructLayout(GTI.getStructType())->getElementOffset(CI->getZExtValue());
        continue;
      }
      const int64_t Stride = GTI.getSequentialElementStride(DL).getFixedValue();
      if (CI)
        A.Const += CI->getSExtValue() * Stride;
      else
        A.Key.emplace_back(Idx, Stride); // uniform, or lane 0's value of Idx
      if (Uni)
        continue;
      auto D = Diffs(Idx); // indices narrower than the pointer are sign-extended
      if (!D || (Idx->getType()->getScalarSizeInBits() < 64 && !D->ExactS))
        return false;
      for (unsigned L = 0; L < S; ++L)
        A.B[L] = int64_t(uint64_t(A.B[L]) + uint64_t(D->D[L].getSExtValue()) * uint64_t(Stride));
    }
    return true;
  };
  auto Analyze = [&](IntrinsicInst *C) -> std::optional<Access> {
    const bool Gather = C->getIntrinsicID() == Intrinsic::masked_gather;
    Access A{C, Gather, C->getArgOperand(Gather ? 0 : 1), C->getArgOperand(Gather ? 2 : 3)};
    A.T = cast<VectorType>((Gather ? C : C->getArgOperand(0))->getType())->getElementType();
    A.A = cast<ConstantInt>(C->getArgOperand(Gather ? 1 : 2))->getAlignValue();
    A.B.assign(S, 0);
    if (DL.getTypeStoreSize(A.T) != DL.getTypeAllocSize(A.T) || !Walk(A.Ptr, A) ||
        isa<AllocaInst>(getUnderlyingObject(A.Root)))
      return std::nullopt;
    return A;
  };

  // element j of a window is read / written by lanes Lanes[j]: mask lane-wise or of them
  auto WindowMask = [&](IRBuilder<> &B, Value *Mask, ArrayRef<SmallVector<int, 4>> Lanes) -> Value * {
    auto *C = dyn_cast<Constant>(Mask);
    if (C && C->isAllOnesValue()) {
      SmallVector<Constant *, 32> E;
      for (auto &L : Lanes)
        E.push_back(B.getInt1(!L.empty()));
      return ConstantVector::get(E);
    }
    Value *R = nullptr;
    for (size_t Round = 0;; ++Round) {
      SmallVector<int, 32> Sel;
      bool Any = false;
      for (auto &L : Lanes) {
        Any |= Round < L.size();
        Sel.push_back(Round < L.size() ? L[Round] : int(S)); // S: a false lane
      }
      if (!Any)
        return R;
      Value *M = B.CreateShuffleVector(Mask, Constant::getNullValue(Mask->getType()), Sel);
      R = R ? B.CreateOr(R, M) : M;
    }
  };
  std::array<unsigned, 4> Count{};
  auto Emit = [&](SmallVector<Access, 4> &G) {
    const int64_t E = DL.getTypeStoreSize(G[0].T);
    int64_t Lo = INT64_MAX, Hi = INT64_MIN;
    for (auto &A : G)
      for (int64_t Off : A.B) {
        Lo = std::min(Lo, A.Const + Off);
        Hi = std::max(Hi, A.Const + Off + E);
      }
    const int64_t N = (Hi - Lo) / E, NV = N <= int64_t(S) ? S : 2 * S;
    auto Elem = [&](const Access &A, unsigned L) { return (A.Const + A.B[L] - Lo) / E; };
    if (N > 2 * int64_t(S) || any_of(G, [&](const Access &A) {
          return any_of(A.B, [&](int64_t Off) { return (A.Const + Off - Lo) % E != 0; });
        }))
      return;
    SmallVector<SmallVector<int, 4>, 32> Lanes(NV);
    for (auto &A : G)
      for (unsigned L = 0; L < S; ++L)
        if (!is_contained(Lanes[Elem(A, L)], int(L)))
          Lanes[Elem(A, L)].push_back(L);
    if (!G[0].Gather && any_of(Lanes, [](auto &L) { return L.size() > 1; }))
      return; // a scatter writing one element from two lanes
    IRBuilder<> B(G[0].Call);
    auto *VT = FixedVectorType::get(G[0].T, NV);
    Value *Start = B.CreateGEP(B.getInt8Ty(), B.CreateExtractElement(G[0].Ptr, uint64_t(0)),
                               B.getInt64(Lo - G[0].Const));
    Align Al = G[0].A;
    for (auto &A : G)
      Al = std::min(Al, A.A);
    Value *Mask = WindowMask(B, G[0].Mask, Lanes);
    auto *MC = dyn_cast<Constant>(Mask);
    const bool Full = MC && MC->isAllOnesValue(); // every element, all lanes on
    if (G[0].Gather) {
      Value *Window = Full ? static_cast<Value *>(B.CreateAlignedLoad(VT, Start, Al))
                           : B.CreateMaskedLoad(VT, Start, Al, Mask);
      for (auto &A : G) {
        IRBuilder<> BA(A.Call); // at the gather: its passthru may be defined after G[0]
        SmallVector<int, 16> Sel;
        for (unsigned L = 0; L < S; ++L)
          Sel.push_back(Elem(A, L));
        Value *X = BA.CreateShuffleVector(Window, Sel);
        auto *AllOn = dyn_cast<Constant>(A.Mask);
        if (!isa<UndefValue>(A.Call->getArgOperand(3)) && !(AllOn && AllOn->isAllOnesValue()))
          X = BA.CreateSelect(A.Mask, X, A.Call->getArgOperand(3));
        A.Call->replaceAllUsesWith(X);
        A.Call->eraseFromParent();
      }
      Count[0] += G.size();
    } else {
      SmallVector<int, 32> Sel;
      for (auto &L : Lanes)
        Sel.push_back(L.empty() ? -1 : L[0]);
      Value *X = B.CreateShuffleVector(G[0].Call->getArgOperand(0), Sel);
      if (Full)
        B.CreateAlignedStore(X, Start, Al);
      else
        B.CreateMaskedStore(X, Start, Al, Mask);
      G[0].Call->eraseFromParent();
      Count[2] += 1;
    }
  };

  SmallVector<SmallVector<Access, 4>, 16> Groups;
  for (auto &BB : V) {
    size_t Open = Groups.size(); // gather groups from here on may still take a gather
    for (auto &I : BB) {
      auto *C = dyn_cast<IntrinsicInst>(&I);
      const bool Gather = C && C->getIntrinsicID() == Intrinsic::masked_gather;
      const bool Scatter = C && C->getIntrinsicID() == Intrinsic::masked_scatter;
      Count[1] += Gather;
      Count[3] += Scatter;
      if (I.mayWriteToMemory())
        Open = Groups.size(); // no window load across a write
      auto A = Gather || Scatter ? Analyze(C) : std::nullopt;
      if (!A)
        continue;
      auto *G = Gather ? find_if(drop_begin(Groups, Open), [&](auto &G) {
        return G[0].Gather && G[0].Root == A->Root && G[0].Key == A->Key && G[0].Mask == A->Mask &&
               G[0].T == A->T;
      }) : Groups.end();
      if (G != Groups.end())
        G->push_back(std::move(*A));
      else
        Groups.push_back({std::move(*A)}); // a scatter alone: it closed the open groups above
    }
  }
  for (auto &G : Groups)
    Emit(G);
  return Count;
}

// W not a multiple of S: the last sub-group lacks lanes, which must not run. RV's region entry
// mask is all lanes, so Lane's body (after the allocas) becomes `if (sg_first + lane < W)`, which
// RV linearizes into a mask on every access, call and collective in it.
void KernelVectorizer::guardMissingLanes(Function &Lane) {
  auto &Ctx = M.getContext();
  auto &Entry = Lane.getEntryBlock();
  auto Split = Entry.begin();
  for (auto It = Entry.begin(); It != Entry.end(); ++It)
    if (isa<AllocaInst>(*It))
      Split = std::next(It);
  auto *Body = Entry.splitBasicBlock(Split, "lanes");
  auto *Missing = BasicBlock::Create(Ctx, "missing.lanes", &Lane);
  ReturnInst::Create(Ctx, Missing);
  IRBuilder<> B(Entry.getTerminator());
  Value *Lin = B.CreateAdd(Lane.getArg(K.arg_size()),
                           B.CreateZExt(B.CreateCall(M.getOrInsertFunction("rv_lane_id", B.getInt32Ty())),
                                        B.getInt64Ty()));
  B.CreateCondBr(B.CreateICmpULT(Lin, B.getInt64(W)), Body, Missing);
  Entry.getTerminator()->eraseFromParent();
}

// K becomes one call of the vectorized code V per sub-group: CBS runs K as a 1D kernel whose
// local id x is the sub-group index.
void KernelVectorizer::replaceBody(Function &V) {
  auto &Ctx = M.getContext();
  SmallVector<Value *, 16> Args;
  for (auto &A : K.args())
    Args.push_back(&A);
  auto *SP = K.getSubprogram();
  K.deleteBody();
  K.setSubprogram(SP);
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", &K));
  if (SP)
    B.SetCurrentDebugLocation(DILocation::get(Ctx, SP->getLine(), 0, SP));
  auto Builtin = [&](const Twine &N) {
    return B.CreateCall(M.getOrInsertFunction(("__acpp_sscp_" + N).str(), B.getInt64Ty()));
  };
  Args.push_back(B.CreateMul(Builtin("get_local_id_x"), B.getInt64(S), "sg_first"));
  for (const char *Q : {"get_group_id_", "get_num_groups_"})
    for (const char *D : {"x", "y", "z"})
      Args.push_back(Builtin(Twine(Q) + D));
  auto *Call = B.CreateCall(&V, Args);
  B.CreateRetVoid();
  InlineFunctionInfo IFI;
  InlineFunction(*Call, IFI);
  SmallVector<CallInst *, 4> Collectives; // their vector functions
  for (auto &I : instructions(K))
    if (auto *C = dyn_cast<CallInst>(&I);
        C && C->getCalledFunction() && C->getCalledFunction()->getName().starts_with(CollectivePrefix))
      Collectives.push_back(C);
  for (auto *C : Collectives)
    InlineFunction(*C, IFI);
  // CBS keeps the entry block out of its sub-CFGs and does not give an alloca with a user there
  // (as RV's per-lane pointers into its replicated allocas) a copy per work-item (sub-group):
  // the entry block keeps only the allocas
  auto &Entry = K.getEntryBlock();
  auto FirstNonAlloca = Entry.begin();
  while (isa<AllocaInst>(*FirstNonAlloca))
    ++FirstNonAlloca;
  Entry.splitBasicBlock(FirstNonAlloca, "vectorized");
  K.addFnAttr(cbs::SubgroupInstancesAttrName, std::to_string((W + S - 1) / S));
}

bool KernelVectorizer::run(FunctionAnalysisManager &FAM) {
  { // the IR is inlined but not optimized yet: to SSA
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(InstCombinePass(InstCombineOptions().setVerifyFixpoint(false)));
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    FPM.addPass(InstCombinePass(InstCombineOptions().setVerifyFixpoint(false)));
    FPM.addPass(ADCEPass());
    FPM.run(K, FAM);
  }

  // per-lane copy Lane(kernel args..., id args); everything per lane comes from rv_lane_id
  auto &Ctx = M.getContext();
  SmallVector<Type *, 16> Params(K.getFunctionType()->params());
  Params.append(NumIdArgs, Type::getInt64Ty(Ctx));
  auto *Lane = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), Params, false),
                                Function::InternalLinkage, K.getName() + ".lane", M);
  ValueToValueMapTy VMap;
  for (unsigned I = 0; I < K.arg_size(); ++I)
    VMap[K.getArg(I)] = Lane->getArg(I);
  SmallVector<ReturnInst *, 4> Rets;
  CloneFunctionInto(Lane, &K, VMap, CloneFunctionChangeType::LocalChangesOnly, Rets);
  Lane->setLinkage(Function::InternalLinkage); // again: cloning copied K's (not dso_local)
  Lane->setSubprogram(nullptr);

  Function *V = nullptr;
  auto Finish = [&](bool Ok) {
    for (auto *F : {Lane, V})
      if (F)
        erase(F, FAM);
    for (auto &F : make_early_inc_range(M)) // scalar collectives, RV intrinsics
      if (F.isDeclaration() && F.use_empty() &&
          (F.getName().starts_with(CollectivePrefix) || F.getName().starts_with("rv_")))
        F.eraseFromParent();
    for (auto &F : make_early_inc_range(M)) // vector functions not inlined (failure)
      if (F.getName().starts_with(CollectivePrefix) && F.use_empty())
        erase(&F, FAM);
    FAM.invalidate(K, PreservedAnalyses::none());
    return Ok;
  };
  const bool Pin = getEnvironmentVariableOrDefault<int>("VECTORIZE_PIN", 1); // P10, P10b
  if (!rewriteBuiltins(*Lane))
    return Finish(false);
  if (W % S)
    guardMissingLanes(*Lane);

  // RV widens scalar types only and copies a vector-typed value (e.g. a sycl::float3) once per
  // lane: split them, with their loads and stores, into scalars, so each component becomes one
  // <S x T>. Then RV's preconditions: simplified loops, canonical loop exits, no critical edges.
  unsigned Joins;
  {
    FunctionPassManager FPM;
    ScalarizerPassOptions ScalarizeAll;
    ScalarizeAll.ScalarizeLoadStore = true;
    FPM.addPass(ScalarizerPass(ScalarizeAll));
    FPM.addPass(InstCombinePass(InstCombineOptions().setVerifyFixpoint(false)));
    FPM.addPass(LoopSimplifyPass());
    FPM.addPass(BreakCriticalEdgesPass());
    // after the edge splits: they move loop exits, and RV keeps each lane's value at its exit in
    // the exit blocks' LCSSA phis (else a value used after a divergent loop becomes undef)
    FPM.addPass(LCSSAPass());
    FPM.run(*Lane, FAM);
    FAM.invalidate(*Lane, PreservedAnalyses::none());
    LoopExitCanonicalizer(FAM.getResult<LoopAnalysis>(*Lane)).canonicalize(*Lane);
    FAM.invalidate(*Lane, PreservedAnalyses::none());
    Joins = Pin ? splitJoins(*Lane) : 0;
    FAM.getResult<LoopAnalysis>(*Lane); // RV reads it via getCachedResult
  }
  alignLaneCopies(*Lane);
  // ACPP_S2_VECTORIZE_PRIVATE=0 keeps the memsets (A/B profiling)
  const unsigned Memsets =
      getEnvironmentVariableOrDefault<int>("VECTORIZE_PRIVATE", 1) ? lowerPrivateMemsets(*Lane) : 0;
  if (verifyFunction(*Lane, &errs())) {
    Why = "invalid per-lane function";
    return Finish(false);
  }
  SmallVector<Instruction *, 16> Pinned; // ACPP_S2_VECTORIZE_PIN=0 disables it and splitJoins
  if (Pin)
    Pinned = laneInvariantValues(*Lane);

  // RV, after the rv_* declarations exist (PlatformInfo registers them when created). All
  // arguments are uniform; sg_first is a multiple of S.
  bool Ok;
  unsigned Branches = 0, UniformBranches = 0; // conditional, after RV's analysis
  {
    rv::PlatformInfo PI(M, nullptr, nullptr); // RV uses neither TTI nor TLI
    for (auto &Mapping : Mappings)
      PI.addMapping(Mapping);
    rv::VectorShapeVec ArgShapes(Lane->arg_size(), rv::VectorShape::uni());
    ArgShapes[K.arg_size()] = rv::VectorShape::uni(S);
    V = rv::createVectorDeclaration(*Lane, rv::VectorShape::uni(), ArgShapes, S, -1);
    V->setLinkage(Function::InternalLinkage);
    rv::VectorMapping Mapping(Lane, V, S, -1, rv::VectorShape::uni(), ArgShapes,
                              rv::CallPredicateMode::SafeWithoutPredicate);
    rv::FunctionRegion FR(*Lane);
    rv::Region R(FR);
    rv::VectorizationInfo VecInfo(R, Mapping);
    for (auto &I : instructions(*Lane)) // RV has no shape rule for these unless varying (asserts)
      if (isa<PtrToIntInst, IntToPtrInst>(I))
        VecInfo.setPinnedShape(I, rv::VectorShape::varying());
    for (auto *C : UniformCalls) // barriers have side effects: RV would call them once per lane
      VecInfo.setPinnedShape(*C, rv::VectorShape::uni());
    for (auto *I : Pinned) // lane-invariant 
      VecInfo.setPinnedShape(*I, rv::VectorShape::uni());
    rv::Config Cfg = rv::Config::createDefaultConfig();
    const auto Features = sys::getHostCPUFeatures();
    Cfg.useAVX = Cfg.useAVX2 = Features.lookup("avx2");
    Cfg.useAVX512 = Features.lookup("avx512f");
    rv::VectorizerInterface Vectorizer(PI, Cfg);
    Vectorizer.analyze(VecInfo, FAM);
    for (auto &BB : *Lane)
      if (auto *Br = dyn_cast<BranchInst>(BB.getTerminator()); Br && Br->isConditional()) {
        ++Branches;
        UniformBranches += VecInfo.getVectorShape(*Br).isUniform();
      }
    Ok = Vectorizer.linearize(VecInfo, FAM) && Vectorizer.vectorize(VecInfo, FAM, nullptr) &&
         !verifyFunction(*V, &errs());
  }
  if (!Ok) {
    Why = "RV failed";
    return Finish(false);
  }

  // ACPP_S2_VECTORIZE_WINDOWS=0 keeps periodic gathers / scatters (A/B profiling)
  std::array<unsigned, 4> Windows{};
  if (getEnvironmentVariableOrDefault<int>("VECTORIZE_WINDOWS", 1)) {
    Windows = windowAccesses(*V);
    if (verifyFunction(*V, &errs())) {
      Why = "invalid window accesses";
      return Finish(false);
    }
  }
  unsigned ElementMajor = 0; // private arrays RV's StructOpt laid out [N x <S x T>]
  for (auto &I : instructions(*V))
    if (auto *A = dyn_cast<AllocaInst>(&I))
      if (auto *AT = dyn_cast<ArrayType>(A->getAllocatedType()))
        ElementMajor += isa<FixedVectorType>(AT->getElementType());
  HIPSYCL_DEBUG_INFO << "[SubgroupVectorizer] " << K.getName() << ": " << Pinned.size()
                     << " lane-invariant value(s) pinned uniform, " << Joins << " join(s) split, "
                     << UniformBranches << " of " << Branches << " branch(es) uniform, " << Memsets
                     << " memset(s) lowered, " << ElementMajor << " private array(s) element-major, "
                     << Windows[0] << " of " << Windows[1] << " gather(s) and " << Windows[2] << " of "
                     << Windows[3] << " scatter(s) as windows\n";
  replaceBody(*V);
  return Finish(true);
}
} // namespace

PreservedAnalyses SubgroupVectorizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  const unsigned S = SubgroupSize;
  const std::array<uint64_t, 3> L{uint64_t(std::max(GroupSize[0], 0)),
                                  uint64_t(std::max(GroupSize[1], 0)),
                                  uint64_t(std::max(GroupSize[2], 0))};
  const uint64_t W = L[0] * L[1] * L[2];
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  // debugging: ACPP_S2_VECTORIZE_SKIP=a,b keeps kernels whose name contains a or b scalar
  SmallVector<StringRef, 4> Skip;
  const auto SkipList = getEnvironmentVariableOrDefault<std::string>("VECTORIZE_SKIP", "");
  StringRef(SkipList).split(Skip, ',', -1, false);
  bool Changed = false;
  for (const auto &Name : KernelNames) {
    auto *K = M.getFunction(Name);
    if (!K || K->isDeclaration())
      continue;
    KernelVectorizer KV{*K, S, L};
    if (any_of(Skip, [&](StringRef P) { return StringRef(Name).contains(P); }))
      KV.Why = "ACPP_S2_VECTORIZE_SKIP";
    else if (W == 0)
      KV.Why = "the work-group size is not known at JIT time";
    else if (KV.run(FAM)) {
      HIPSYCL_DEBUG_INFO << "[SubgroupVectorizer] " << Name << ": vectorized at width " << S
                         << ", " << (W + S - 1) / S << " sub-groups per work-group"
                         << (W % S ? " (the last partial)" : "") << "\n";
      Changed = true;
      continue;
    }
    HIPSYCL_DEBUG_WARNING << "[SubgroupVectorizer] " << Name << " stays scalar: " << KV.Why
                          << "\n";
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
