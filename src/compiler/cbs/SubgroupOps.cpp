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
#include "hipSYCL/compiler/cbs/SubgroupOps.hpp"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Instructions.h>

namespace hipsycl::compiler {
using namespace llvm;

namespace sgops {
std::optional<Collective> parse(StringRef Name) { // e.g. "sub_group_shl_i32"
  if (!Name.consume_front("sub_group_"))
    return std::nullopt;
  if (Name == "any" || Name == "all" || Name == "none")
    return Collective{Name == "any" ? Collective::Any : Name == "all" ? Collective::All
                                                                      : Collective::None, 'b'};
  auto [Kind, Type] = Name.rsplit('_');
  if (Type.empty() || !StringRef("iuf").contains(Type[0]))
    return std::nullopt;
  auto K = StringSwitch<std::optional<Collective::Kind>>(Kind)
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

// reduce and scans need a constant __acpp_sscp_algorithm_op the builtins support for the type
bool isSupported(const Collective &C, const CallInst &Call) {
  if (C.K != Collective::Reduce && C.K != Collective::InclusiveScan &&
      C.K != Collective::ExclusiveScan)
    return true;
  auto *Op = dyn_cast<ConstantInt>(Call.getArgOperand(0));
  return Op && Op->getZExtValue() <= (C.Class == 'f' ? 3u : 8u); // plus .. max / .. logical_or
}

// L op R as the builtins' binary operations (sscp/builtins/detail/utils.hpp), scalar or vector
Value *combine(IRBuilder<> &B, unsigned Op, char Class, Value *L, Value *R) {
  const bool FP = Class == 'f';
  auto Less = [&] {
    return FP ? B.CreateFCmpOLT(L, R) : Class == 'u' ? B.CreateICmpULT(L, R) : B.CreateICmpSLT(L, R);
  };
  auto NonZero = [&](Value *V) { return B.CreateICmpNE(V, Constant::getNullValue(V->getType())); };
  switch (Op) {
  case 0: return FP ? B.CreateFAdd(L, R) : B.CreateAdd(L, R);
  case 1: return FP ? B.CreateFMul(L, R) : B.CreateMul(L, R);
  case 2: return B.CreateSelect(Less(), L, R); // lhs < rhs ? lhs : rhs
  case 3: return B.CreateSelect(Less(), R, L);
  case 4: return B.CreateAnd(L, R);
  case 5: return B.CreateOr(L, R);
  case 6: return B.CreateXor(L, R);
  case 7: return B.CreateZExt(B.CreateAnd(NonZero(L), NonZero(R)), L->getType());
  default: return B.CreateZExt(B.CreateOr(NonZero(L), NonZero(R)), L->getType());
  }
}

Constant *identity(unsigned Op, char Class, Type *T) { // fp plus: -0.0, so that -0.0 + -0.0 = -0.0
  if (Class == 'f')
    return Op == 0   ? ConstantFP::getNegativeZero(T)
           : Op == 1 ? ConstantFP::get(T, 1.0)
                     : ConstantFP::getInfinity(T, /*Negative*/ Op == 3); // min: +inf, max: -inf
  const unsigned W = T->getIntegerBitWidth();
  const bool Signed = Class == 'i';
  switch (Op) {
  case 1:
  case 7: return ConstantInt::get(T, 1); // multiply, logical and
  case 2: return ConstantInt::get(T, Signed ? APInt::getSignedMaxValue(W) : APInt::getMaxValue(W));
  case 3: return ConstantInt::get(T, Signed ? APInt::getSignedMinValue(W) : APInt::getMinValue(W));
  case 4: return ConstantInt::get(T, APInt::getAllOnes(W));
  default: return ConstantInt::get(T, 0); // plus, or, xor, logical or
  }
}

Constant *laneIds(Type *T, unsigned S) { // <0, 1, ..., S-1>
  SmallVector<Constant *, 64> Ids;
  for (unsigned J = 0; J < S; ++J)
    Ids.push_back(ConstantInt::get(T, J));
  return ConstantVector::get(Ids);
}

// lane j gets lane Src(j) of V; Src(j) < 0: unused lane
static Value *permute(IRBuilder<> &B, Value *V, unsigned S, function_ref<int(int)> Src) {
  SmallVector<int, 64> Mask;
  for (unsigned J = 0; J < S; ++J)
    Mask.push_back(Src(J));
  return B.CreateShuffleVector(V, Mask);
}

// shifts, permute, select / broadcast: lane j reads lane src(j) mod S. P, the delta / mask / id,
// is a constant (one shufflevector) or, per lane, a vector (lane-by-lane gather)
Value *shuffle(IRBuilder<> &B, Collective::Kind K, Value *X, Value *P, unsigned S) {
  if (auto *PC = dyn_cast<ConstantInt>(P)) {
    const uint64_t D = PC->getZExtValue();
    return permute(B, X, S, [&](int J) -> int {
      switch (K) {
      case Collective::Shl: return (J + D) % S;
      case Collective::Shr: return (J + S - D % S) % S;
      case Collective::Permute: return (J ^ D) % S;
      default: return D % S;
      }
    });
  }
  auto *Lanes = laneIds(P->getType()->getScalarType(), S);
  Value *Src = K == Collective::Shl       ? B.CreateAdd(Lanes, P)
               : K == Collective::Shr     ? B.CreateSub(Lanes, P)
               : K == Collective::Permute ? B.CreateXor(Lanes, P)
                                          : P;
  Src = B.CreateURem(Src, ConstantVector::getSplat(ElementCount::getFixed(S),
                                                   ConstantInt::get(P->getType()->getScalarType(), S)));
  Value *R = PoisonValue::get(X->getType());
  for (unsigned J = 0; J < S; ++J)
    R = B.CreateInsertElement(R, B.CreateExtractElement(X, B.CreateExtractElement(Src, J)), J);
  return R;
}

// as sg_reduce_impl: for d = S/2 .. 1, lane j combines lane j + d if j + d < S (missing or
// inactive lanes hold the identity); lane 0's value goes to every lane
Value *reduce(IRBuilder<> &B, unsigned Op, char Class, Value *V, unsigned S) {
  auto *Lanes = laneIds(B.getInt32Ty(), S);
  for (unsigned D = S / 2; D > 0; D /= 2) {
    Value *Other = permute(B, V, S, [&](int J) { return (J + D) % S; });
    Value *Valid = B.CreateICmpULT(B.CreateAdd(Lanes, ConstantInt::get(Lanes->getType(), D)),
                                   ConstantInt::get(Lanes->getType(), S));
    V = B.CreateSelect(Valid, combine(B, Op, Class, V, Other), V);
  }
  return B.CreateVectorSplat(S, B.CreateExtractElement(V, uint64_t(0)));
}

// as sg_inclusive_scan: for d = 1, 2, 4 ..: lane j >= d combines lane j - d
Value *inclusiveScan(IRBuilder<> &B, unsigned Op, char Class, Value *V, unsigned S) {
  for (unsigned D = 1; D < S; D *= 2) {
    Value *Other = permute(B, V, S, [&](int J) { return J >= int(D) ? J - int(D) : -1; });
    SmallVector<Constant *, 64> Takes;
    for (unsigned J = 0; J < S; ++J)
      Takes.push_back(B.getInt1(J >= D));
    V = B.CreateSelect(ConstantVector::get(Takes), combine(B, Op, Class, V, Other), V);
  }
  return V;
}

// as sg_exclusive_scan: lane 0 starts from op(x0, init0); lane j gets the inclusive result of
// lane j - 1, lane 0 init0 (init of lane 0)
Value *exclusiveScan(IRBuilder<> &B, unsigned Op, char Class, Value *V, Value *Init, unsigned S) {
  Value *Init0 = Init->getType()->isVectorTy() ? B.CreateExtractElement(Init, uint64_t(0)) : Init;
  V = B.CreateInsertElement(
      V, combine(B, Op, Class, B.CreateExtractElement(V, uint64_t(0)), Init0), uint64_t(0));
  V = permute(B, inclusiveScan(B, Op, Class, V, S), S, [](int J) { return J - 1; });
  return B.CreateInsertElement(V, Init0, uint64_t(0));
}
} // namespace sgops
} // namespace hipsycl::compiler
