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

#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

HIPSYCL_SSCP_BUILTIN void *__acpp_sscp_host_get_internal_local_memory();

namespace {
// One slot per lane, after the 8K used by work-group algorithms. Must match
// the internal block layout in omp_queue.cpp and StaticLocalMemoryPass.cpp.
constexpr __acpp_uint64 sg_scratch_offset = 1024 * sizeof(__acpp_uint64);

void sg_barrier() {
  __acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope::sub_group,
                                __acpp_sscp_memory_order::relaxed);
}

// Value of lane `src` (modulo the sub-group size) of the caller's sub-group.
// All host sub-group collectives (broadcast, reduce, scan, any/all) build on this.
template <class T> T sg_shuffle(T value, __acpp_uint32 src) {
  const __acpp_uint32 s = __acpp_sscp_get_subgroup_max_size();
  if (s == 1) // only lane 0 exists
    return src == 0 ? value : T{};

  T *scratch = reinterpret_cast<T *>(
      static_cast<char *>(__acpp_sscp_host_get_internal_local_memory()) + sg_scratch_offset);
  const __acpp_uint32 base = __acpp_sscp_get_subgroup_id() * s;
  scratch[base + __acpp_sscp_get_subgroup_local_id()] = value;
  sg_barrier(); // all lanes have written
  const T result = scratch[base + src % __acpp_sscp_get_subgroup_size()];
  sg_barrier(); // all lanes have read before the next shuffle reuses scratch
  return result;
}
} // namespace

// in one macro generate shuffle, shift right, permute, and select
#define ACPP_HOST_SUBGROUP_SHUFFLES(N)                                                             \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##N __acpp_sscp_sub_group_shl_i##N(__acpp_int##N value, __acpp_uint32 delta) {         \
    return sg_shuffle(value, __acpp_sscp_get_subgroup_local_id() + delta);                         \
  }                                                                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##N __acpp_sscp_sub_group_shr_i##N(__acpp_int##N value, __acpp_uint32 delta) {         \
    return sg_shuffle(value, __acpp_sscp_get_subgroup_local_id() - delta);                         \
  }                                                                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##N __acpp_sscp_sub_group_permute_i##N(__acpp_int##N value, __acpp_int32 mask) {       \
    return sg_shuffle(value, __acpp_sscp_get_subgroup_local_id() ^ mask);                          \
  }                                                                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##N __acpp_sscp_sub_group_select_i##N(__acpp_int##N value, __acpp_int32 id) {          \
    return sg_shuffle(value, id);                                                                  \
  }

ACPP_HOST_SUBGROUP_SHUFFLES(8)
ACPP_HOST_SUBGROUP_SHUFFLES(16)
ACPP_HOST_SUBGROUP_SHUFFLES(32)
ACPP_HOST_SUBGROUP_SHUFFLES(64)
