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
#ifndef HIPSYCL_SUBGROUP_STEPS_HPP
#define HIPSYCL_SUBGROUP_STEPS_HPP

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/PassManager.h>

#include <cstddef>
#include <string>

namespace llvm {
class CallInst;
class Instruction;
class Value;
} // namespace llvm

// In the CBS sub-group fission path every sub-group barrier splits the kernel into spans: values
// live across it are stored to per-lane arrays and reloaded. A builtin collective is written for
// one lane and costs 2 (shuffle) to 8 (reduce) such barriers. Instead, independent collectives
// share ONE barrier and each runs once per sub-group as vector code:
//  1. Keep (keepSubgroupCollectives, LLVMToHost, before the builtins are linked): supported
//     __acpp_sscp_sub_group_* calls in kernels call __acpp_cbs_sg.<builtin> instead, which the
//     builtin library doesn't define, so they reach CBS as calls (convergent: O1 doesn't move
//     them under more branches).
//  2. Group (SubgroupStepsPass, CBS, before CanonicalizeBarriers): per block, in order, a
//     collective joins the current group unless it uses a member's result. Pure lane code that
//     uses a result moves after the last member; a dependent load, store or call, or a barrier,
//     ends the group.
//  3. Expand (SubgroupStepsPass): per group, each lane stores its inputs to arrays of S slots
//     (slot = lane), then ONE sub-group barrier, a marker __acpp_cbs_sg_step.<builtin>(out, in..)
//     per collective, and each lane loads its result from out[lane]; the calls are erased.
//  4. Lower (emitSubgroupStep, SubCfgFormation, after the spans are built): a marker starts the
//     span after its barrier; it moves to that span's preheader, run once per sub-group before
//     the lanes, and becomes vector code: masked vector loads, the operation, a masked store.
// Reductions and scans combine lanes in the same order as the builtins, so results match.
namespace hipsycl::compiler {
static constexpr char KeptCollectivePrefix[] = "__acpp_cbs_sg.";
static constexpr char StepMarkerPrefix[] = "__acpp_cbs_sg_step.";

// step 1; returns whether a call was changed
bool keepSubgroupCollectives(llvm::Module &M, llvm::ArrayRef<std::string> KernelNames);

// steps 2 and 3; S: sub-group size
class SubgroupStepsPass : public llvm::PassInfoMixin<SubgroupStepsPass> {
  std::size_t S;

public:
  explicit SubgroupStepsPass(std::size_t S) : S{S} {}
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

// step 4: lowers Marker before InsertBefore for the sub-group whose flattened local ids are
// [Begin, End), S its full size, and erases Marker
void emitSubgroupStep(llvm::CallInst &Marker, llvm::Instruction *InsertBefore, llvm::Value *Begin,
                      llvm::Value *End, unsigned S);
} // namespace hipsycl::compiler

#endif
