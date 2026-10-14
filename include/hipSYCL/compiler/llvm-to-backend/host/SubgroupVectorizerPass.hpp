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
#ifndef ACPP_HOST_SUBGROUP_VECTORIZER_PASS_HPP
#define ACPP_HOST_SUBGROUP_VECTORIZER_PASS_HPP

#include <llvm/IR/PassManager.h>

#include <array>
#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {

// Experimental (WITH_RV builds, ACPP_OPENMP_VECTORIZE=1): vectorizes each kernel with RV at the
// sub-group width S, so that one SIMD instance is one sub-group (S consecutive linear local ids).
// Sub-group collectives become vector operations (masked: inactive lanes contribute the
// identity), sub-group barriers vanish. The kernel body becomes one call of the vectorized code
// per sub-group, and CBS runs the kernel as a 1D kernel over its sub-groups. Runs before the
// host builtins are linked, while ids and collectives are still calls. If S doesn't divide the
// work-group size, the last sub-group's missing lanes are masked off. A kernel stays scalar if
// the work-group size is unknown or it uses an unsupported builtin.
class SubgroupVectorizerPass : public llvm::PassInfoMixin<SubgroupVectorizerPass> {
public:
  // GroupSize: x (fastest), y, z, 0 if unknown
  SubgroupVectorizerPass(int SubgroupSize, std::array<int, 3> GroupSize,
                         std::vector<std::string> KernelNames)
      : SubgroupSize{SubgroupSize}, GroupSize{GroupSize}, KernelNames{std::move(KernelNames)} {}
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  int SubgroupSize;
  std::array<int, 3> GroupSize;
  std::vector<std::string> KernelNames;
};

} // namespace compiler
} // namespace hipsycl

#endif
