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
#ifndef ACPP_HOST_MASK_DIVERGENT_SUBGROUP_COLLECTIVES_HPP
#define ACPP_HOST_MASK_DIVERGENT_SUBGROUP_COLLECTIVES_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// A sub-group collective (__acpp_sscp_sub_group_*) reached by only some lanes of a sub-group
// (e.g. under `if (tidxj < 3)`) is undefined in SYCL, but works on GPUs. On the host every lane
// must call it: the call holds sub-group barriers that all lanes of the sub-group must reach.
// This pass takes each collective out of the ifs around it (masking): code before the call stays
// under the original conditions, every lane calls it (lanes that would have skipped it pass the
// identity to reductions, scans and votes), and code after the call goes under the same
// conditions again. Runs before the host builtins are linked, while the collectives are still
// calls.
class MaskDivergentSubgroupCollectivesPass
    : public llvm::PassInfoMixin<MaskDivergentSubgroupCollectivesPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace compiler
} // namespace hipsycl

#endif
