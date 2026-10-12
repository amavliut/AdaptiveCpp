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
#ifndef ACPP_HOST_HOIST_DIVERGENT_SUBGROUP_COLLECTIVES_HPP
#define ACPP_HOST_HOIST_DIVERGENT_SUBGROUP_COLLECTIVES_HPP

#include <llvm/IR/PassManager.h>

#include <array>

namespace hipsycl {
namespace compiler {

// A sub-group collective (__acpp_sscp_sub_group_*) reached by only some lanes of a sub-group
// (e.g. under `if (tidxj < 3)`) is undefined in SYCL, but works on GPUs. On the host it splits
// the sub-group across different sub-group barriers. This pass moves such a call, with its pure
// input computation, above the branch, so every lane calls it; reductions, scans and votes get
// the identity element for lanes that didn't take the branch. Runs before the host builtins
// are linked, while the collectives are still calls. A collective that stays under a branch
// that provably splits a sub-group gets a warning (needs the work-group size known at JIT time).
class HoistDivergentSubgroupCollectivesPass
    : public llvm::PassInfoMixin<HoistDivergentSubgroupCollectivesPass> {
public:
  // GroupSize: x (fastest), y, z, 0 if unknown
  HoistDivergentSubgroupCollectivesPass(int SubgroupSize, std::array<int, 3> GroupSize)
      : SubgroupSize{SubgroupSize}, GroupSize{GroupSize} {}
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  int SubgroupSize;
  std::array<int, 3> GroupSize;
};

} // namespace compiler
} // namespace hipsycl

#endif
