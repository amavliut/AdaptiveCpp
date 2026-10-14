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
#include <llvm/IR/DebugInfoMetadata.h>
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
#include <llvm/Transforms/Utils/BreakCriticalEdges.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <cmath>

namespace hipsycl::compiler {
namespace {
using namespace llvm;

// arguments the per-lane function gets after the kernel's: sg_first (linear local id of lane 0),
// group ids x y z, numbers of groups x y z
constexpr unsigned NumIdArgs = 7;
constexpr char ReducePrefix[] = "acpp_rv_reduce.";

// identity of __acpp_sscp_algorithm_op plus/multiply/min/max for type T (Signed: from the name)
Constant *identity(int Op, Type *T, bool Signed) {
  if (T->isFloatingPointTy()) {
    const double V[] = {0.0, 1.0, INFINITY, -INFINITY};
    return ConstantFP::get(T, V[Op]);
  }
  const unsigned W = T->getIntegerBitWidth();
  const APInt V[] = {APInt(W, 0), APInt(W, 1),
                     Signed ? APInt::getSignedMaxValue(W) : APInt::getMaxValue(W),
                     Signed ? APInt::getSignedMinValue(W) : APInt::getMinValue(W)};
  return ConstantInt::get(T, V[Op]);
}

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
  std::vector<rv::VectorMapping> ReduceMappings;
  SmallVector<CallInst *, 8> UniformCalls; // with side effects, still once per sub-group

  Function *reduceFunction(int Op, Type *T, StringRef TName);
  std::array<Value *, 3> localIds(IRBuilder<> &B, Value *SgFirst);
  bool rewriteBuiltins(Function &Lane);
  void guardMissingLanes(Function &Lane);
  void replaceBody(Function &V);
  void erase(Function *F, FunctionAnalysisManager &FAM) {
    FAM.clear(*F, F->getName());
    F->eraseFromParent();
  }
};

// T reduce(T x) for the per-lane code, with its masked vector version
// <S x T> reduce_v(<S x T> x, <S x i1> mask): inactive lanes contribute the identity. The result
// is splat: RV maps every resolved call result as a vector (NatBuilder::vectorizeCallInstruction).
Function *KernelVectorizer::reduceFunction(int Op, Type *T, StringRef TName) {
  const std::string Name = (ReducePrefix + Twine(Op) + "." + TName).str();
  if (auto *F = M.getFunction(Name))
    return F;
  auto *Scalar =
      Function::Create(FunctionType::get(T, {T}, false), Function::ExternalLinkage, Name, M);
  auto *VecT = FixedVectorType::get(T, S);
  auto *MaskT = FixedVectorType::get(Type::getInt1Ty(M.getContext()), S);
  auto *Vector = Function::Create(FunctionType::get(VecT, {VecT, MaskT}, false),
                                  Function::InternalLinkage, Name + "_v", M);
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", Vector));
  const bool Signed = TName.starts_with("i"), FP = T->isFloatingPointTy();
  Value *X = B.CreateSelect(
      Vector->getArg(1), Vector->getArg(0),
      ConstantVector::getSplat(ElementCount::getFixed(S), identity(Op, T, Signed)));
  FastMathFlags FMF;
  FMF.setAllowReassoc(); // any order is allowed for a group reduction; else LLVM adds in sequence
  B.setFastMathFlags(FMF);
  Value *R = Op == 0   ? (FP ? B.CreateFAddReduce(ConstantFP::get(T, -0.0), X) : B.CreateAddReduce(X))
             : Op == 1 ? (FP ? B.CreateFMulReduce(ConstantFP::get(T, 1.0), X) : B.CreateMulReduce(X))
             : Op == 2 ? (FP ? B.CreateFPMinReduce(X) : B.CreateIntMinReduce(X, Signed))
                       : (FP ? B.CreateFPMaxReduce(X) : B.CreateIntMaxReduce(X, Signed));
  B.CreateRet(B.CreateVectorSplat(S, R));
  ReduceMappings.emplace_back(Scalar, Vector, S, /*maskPos*/ 1, rv::VectorShape::varying(),
                              rv::VectorShapeVec{rv::VectorShape::varying()},
                              rv::CallPredicateMode::PredicateArg);
  return Scalar;
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
  auto &Ctx = M.getContext();
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
    auto Shuffle = [&](Value *Amount) -> Value * { // lane i gets x[i + Amount], constant only
      return isa<ConstantInt>(Amount) ? RV(("rv_shuffle." + N).str(), {C->getArgOperand(0), Amount})
                                      : nullptr;
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
    } else if (N.consume_front("sub_group_shl_")) New = Shuffle(C->getArgOperand(1));
    else if (N.consume_front("sub_group_shr_"))
      New = Shuffle(isa<Constant>(C->getArgOperand(1))
                        ? ConstantExpr::getNeg(cast<Constant>(C->getArgOperand(1))) : C->getArgOperand(1));
    else if (N.consume_front("sub_group_broadcast_")) // (sender, x)
      New = RV(("rv_extract." + N).str(), {C->getArgOperand(1), C->getArgOperand(0)});
    else if (N == "sub_group_any") New = RV("rv_any", {C->getArgOperand(0)});
    else if (N == "sub_group_all") New = RV("rv_all", {C->getArgOperand(0)});
    else if (N == "sub_group_none") New = B.CreateNot(RV("rv_any", {C->getArgOperand(0)}));
    else if (N.consume_front("sub_group_reduce_")) { // (op, x), op plus/multiply/min/max
      auto *Op = dyn_cast<ConstantInt>(C->getArgOperand(0));
      if (Op && Op->getZExtValue() <= 3)
        New = B.CreateCall(reduceFunction(Op->getZExtValue(), T, N), {C->getArgOperand(1)});
    } else if (N == "rsqrt_f32" || N == "rsqrt_f64") // as the host libkernel: 1 / sqrt(x)
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
  SmallVector<CallInst *, 4> Reductions; // masked vector reductions
  for (auto &I : instructions(K))
    if (auto *C = dyn_cast<CallInst>(&I);
        C && C->getCalledFunction() && C->getCalledFunction()->getName().starts_with(ReducePrefix))
      Reductions.push_back(C);
  for (auto *C : Reductions)
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
    for (auto &F : make_early_inc_range(M)) // scalar reductions, RV intrinsics
      if (F.isDeclaration() && F.use_empty() &&
          (F.getName().starts_with(ReducePrefix) || F.getName().starts_with("rv_")))
        F.eraseFromParent();
    for (auto &F : make_early_inc_range(M)) // vector reductions not inlined (failure)
      if (F.getName().starts_with(ReducePrefix) && F.use_empty())
        erase(&F, FAM);
    FAM.invalidate(K, PreservedAnalyses::none());
    return Ok;
  };
  if (!rewriteBuiltins(*Lane))
    return Finish(false);
  if (W % S)
    guardMissingLanes(*Lane);

  // RV widens scalar types only and copies a vector-typed value (e.g. a sycl::float3) once per
  // lane: split them, with their loads and stores, into scalars, so each component becomes one
  // <S x T>. Then RV's preconditions: simplified loops, canonical loop exits, no critical edges.
  {
    FunctionPassManager FPM;
    ScalarizerPassOptions ScalarizeAll;
    ScalarizeAll.ScalarizeLoadStore = true;
    FPM.addPass(ScalarizerPass(ScalarizeAll));
    FPM.addPass(InstCombinePass(InstCombineOptions().setVerifyFixpoint(false)));
    FPM.addPass(LoopSimplifyPass());
    FPM.addPass(LCSSAPass());
    FPM.addPass(BreakCriticalEdgesPass());
    FPM.run(*Lane, FAM);
    FAM.invalidate(*Lane, PreservedAnalyses::none());
    LoopExitCanonicalizer(FAM.getResult<LoopAnalysis>(*Lane)).canonicalize(*Lane);
    FAM.invalidate(*Lane, PreservedAnalyses::none());
    FAM.getResult<LoopAnalysis>(*Lane); // RV reads it via getCachedResult
  }
  if (verifyFunction(*Lane, &errs())) {
    Why = "invalid per-lane function";
    return Finish(false);
  }

  // RV, after the rv_* declarations exist (PlatformInfo registers them when created). All
  // arguments are uniform; sg_first is a multiple of S.
  bool Ok;
  {
    rv::PlatformInfo PI(M, nullptr, nullptr); // RV uses neither TTI nor TLI
    for (auto &Mapping : ReduceMappings)
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
    for (auto *C : UniformCalls) // barriers have side effects: RV would call them once per lane
      VecInfo.setPinnedShape(*C, rv::VectorShape::uni());
    rv::Config Cfg = rv::Config::createDefaultConfig();
    const auto Features = sys::getHostCPUFeatures();
    Cfg.useAVX = Cfg.useAVX2 = Features.lookup("avx2");
    Cfg.useAVX512 = Features.lookup("avx512f");
    rv::VectorizerInterface Vectorizer(PI, Cfg);
    Vectorizer.analyze(VecInfo, FAM);
    Ok = Vectorizer.linearize(VecInfo, FAM) && Vectorizer.vectorize(VecInfo, FAM, nullptr) &&
         !verifyFunction(*V, &errs());
  }
  if (!Ok) {
    Why = "RV failed";
    return Finish(false);
  }

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
