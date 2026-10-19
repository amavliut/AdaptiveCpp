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
#ifndef HIPSYCL_SUBGROUP_OPS_HPP
#define HIPSYCL_SUBGROUP_OPS_HPP

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>

#include <optional>

namespace hipsycl::compiler {
// The sub-group collectives as vector operations on the S lanes of a sub-group, in the builtins'
// order of operations (results as the per-lane builtins). Used by the RV path
// (SubgroupVectorizerPass); MaskDivergentSubgroupCollectivesPass takes parse() and identity().
// Op: the __acpp_sscp_algorithm_op; Class: from the builtin's name.
namespace sgops {
struct Collective { // a supported builtin: __acpp_sscp_sub_group_<kind>_<type>, or any / all / none
  enum Kind { Shl, Shr, Permute, Select, Broadcast, Reduce, InclusiveScan, ExclusiveScan, Any, All, None };
  Kind K;
  char Class; // 'i' signed, 'u' unsigned, 'f' floating point, 'b' bool
};
std::optional<Collective> parse(llvm::StringRef Name); // e.g. "sub_group_shl_i32"
bool isSupported(const Collective &C, const llvm::CallInst &Call);
llvm::Value *combine(llvm::IRBuilder<> &B, unsigned Op, char Class, llvm::Value *L, llvm::Value *R);
llvm::Constant *identity(unsigned Op, char Class, llvm::Type *T); // x op identity = x
llvm::Constant *laneIds(llvm::Type *T, unsigned S);
llvm::Value *shuffle(llvm::IRBuilder<> &B, Collective::Kind K, llvm::Value *X, llvm::Value *P, unsigned S);
llvm::Value *reduce(llvm::IRBuilder<> &B, unsigned Op, char Class, llvm::Value *V, llvm::Value *N,
                    unsigned S);
llvm::Value *inclusiveScan(llvm::IRBuilder<> &B, unsigned Op, char Class, llvm::Value *V, unsigned S);
llvm::Value *exclusiveScan(llvm::IRBuilder<> &B, unsigned Op, char Class, llvm::Value *V,
                           llvm::Value *Init, unsigned S);
} // namespace sgops
} // namespace hipsycl::compiler

#endif
