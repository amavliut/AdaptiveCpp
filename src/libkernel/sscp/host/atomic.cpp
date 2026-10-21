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
#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"


inline constexpr int builtin_memory_order(__acpp_sscp_memory_order o) noexcept {
  switch(o){
    case __acpp_sscp_memory_order::relaxed:
      return __ATOMIC_RELAXED;
    case __acpp_sscp_memory_order::acquire:
      return __ATOMIC_ACQUIRE;
    case __acpp_sscp_memory_order::release:
      return __ATOMIC_RELEASE;
    case __acpp_sscp_memory_order::acq_rel:
      return __ATOMIC_ACQ_REL;
    case __acpp_sscp_memory_order::seq_cst:
      return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_RELAXED;
}

// One thread runs a whole work-group on the host: an atomic that only the work-group can observe
// (local or private memory, or scope work-item / sub-group / work-group) has no concurrent access,
// so a plain read-modify-write is enough. as and scope are constants once inlined: the test folds.
inline constexpr bool wg_only(__acpp_sscp_address_space as, __acpp_sscp_memory_scope scope) noexcept {
  return as == __acpp_sscp_address_space::local_space ||
         as == __acpp_sscp_address_space::private_space ||
         scope <= __acpp_sscp_memory_scope::work_group;
}

// + and - that wrap like the atomic builtins (signed overflow is defined there)
template <class T> T wrap_add(T a, T b) { T r; __builtin_add_overflow(a, b, &r); return r; }
template <class T> T wrap_sub(T a, T b) { T r; __builtin_sub_overflow(a, b, &r); return r; }
inline __acpp_f32 wrap_add(__acpp_f32 a, __acpp_f32 b) { return a + b; }
inline __acpp_f64 wrap_add(__acpp_f64 a, __acpp_f64 b) { return a + b; }
inline __acpp_f32 wrap_sub(__acpp_f32 a, __acpp_f32 b) { return a - b; }
inline __acpp_f64 wrap_sub(__acpp_f64 a, __acpp_f64 b) { return a - b; }

// the work-group-only case of a read-modify-write: *ptr = new_value, return the old value
#define ACPP_WG_ONLY_RMW(new_value)                                                                \
  if (wg_only(as, scope)) {                                                                        \
    auto old = *ptr;                                                                               \
    *ptr = new_value;                                                                              \
    return old;                                                                                    \
  }
#define ACPP_WG_ONLY_CMP_EXCH                                                                      \
  if (wg_only(as, scope)) {                                                                        \
    if (*ptr == *expected) {                                                                       \
      *ptr = desired;                                                                              \
      return true;                                                                                 \
    }                                                                                              \
    *expected = *ptr;                                                                              \
    return false;                                                                                  \
  }


// ********************** atomic store ***************************

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  if (wg_only(as, scope)) { *ptr = x; return; }
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  if (wg_only(as, scope)) { *ptr = x; return; }
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (wg_only(as, scope)) { *ptr = x; return; }
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (wg_only(as, scope)) { *ptr = x; return; }
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}


// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_load_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr) {
  if (wg_only(as, scope)) return *ptr;
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_load_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr) {
  if (wg_only(as, scope)) return *ptr;
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr) {
  if (wg_only(as, scope)) return *ptr;
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_load_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr) {
  if (wg_only(as, scope)) return *ptr;
  return __atomic_load_n(ptr, builtin_memory_order(order));
}


// ********************** atomic exchange ***************************

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_exchange_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(x)
  return __atomic_exchange_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_exchange_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr,
    __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(x)
    return __atomic_exchange_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_exchange_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr,
    __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(x)
    return __atomic_exchange_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_exchange_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr,
    __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(x)
    return __atomic_exchange_n(ptr, x, builtin_memory_order(order));
}

// ********************** atomic compare exchange weak **********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int8 *ptr, __acpp_int8 *expected, __acpp_int8 desired) {
  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, true,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int16 *ptr, __acpp_int16 *expected, __acpp_int16 desired) {
  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, true,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {
  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, true,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {
  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, true,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

// ********************* atomic compare exchange strong  *********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int8 *ptr, __acpp_int8 *expected, __acpp_int8 desired) {

  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int16 *ptr, __acpp_int16 *expected, __acpp_int16 desired) {

  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {

  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {

  ACPP_WG_ONLY_CMP_EXCH
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     builtin_memory_order(success), builtin_memory_order(failure));
}

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_and_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {

  ACPP_WG_ONLY_RMW(old & x)
  return __atomic_fetch_and(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_and_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {

  ACPP_WG_ONLY_RMW(old & x)
  return __atomic_fetch_and(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_and_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {

  ACPP_WG_ONLY_RMW(old & x)
  return __atomic_fetch_and(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_and_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {

  ACPP_WG_ONLY_RMW(old & x)
  return __atomic_fetch_and(ptr, x, builtin_memory_order(order));
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_or_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(old | x)
  return __atomic_fetch_or(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_or_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(old | x)
  return __atomic_fetch_or(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_or_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(old | x)
  return __atomic_fetch_or(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_or_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(old | x)
  return __atomic_fetch_or(ptr, x, builtin_memory_order(order));
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_xor_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(old ^ x)
  return __atomic_fetch_xor(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_xor_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(old ^ x)
  return __atomic_fetch_xor(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_xor_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(old ^ x)
  return __atomic_fetch_xor(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_xor_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(old ^ x)
  return __atomic_fetch_xor(ptr, x, builtin_memory_order(order));
}



// TODO: It seems that at least on gfx90a, there are additional unsafe atomic
// operations that can yield a performance improvement. Should we expose those?
HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_add_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_add_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_add_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_add_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_add_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_add_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_add_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_add_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  ACPP_WG_ONLY_RMW(wrap_add(old, x))
  return __atomic_fetch_add(ptr, x, builtin_memory_order(order));
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_sub_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_sub_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_sub_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_sub_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_sub_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_sub_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_sub_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_sub_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_sub_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_sub_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  ACPP_WG_ONLY_RMW(wrap_sub(old, x))
  return __atomic_fetch_sub(ptr, x, builtin_memory_order(order));
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_min_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_min_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_min_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_min_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_min_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_min_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_min_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_min_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  ACPP_WG_ONLY_RMW(x < old ? x : old)
  return __atomic_fetch_min(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_min_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  __acpp_int32 old_i = __acpp_sscp_atomic_load_i32(as, order, scope, (__acpp_int32*)ptr);
  __acpp_f32 old = *(__acpp_f32*)&old_i;
  do{
    if (old < x) return old;
  } while (!__acpp_sscp_cmp_exch_strong_i32(as, order, order, scope, (__acpp_int32*)ptr, (__acpp_int32*)&old, *(__acpp_int32*)&x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_min_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  __acpp_int64 old_i = __acpp_sscp_atomic_load_i64(as, order, scope, (__acpp_int64*)ptr);
  __acpp_f64 old = *(__acpp_f64*)&old_i;
  do{
    if (old < x) return old;
  } while (!__acpp_sscp_cmp_exch_strong_i64(as, order, order, scope, (__acpp_int64*)ptr, (__acpp_int64*)&old, *(__acpp_int64*)&x));
  return x;
}


HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_max_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_max_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_max_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_max_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_max_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_max_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_max_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_max_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  ACPP_WG_ONLY_RMW(x > old ? x : old)
  return __atomic_fetch_max(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_max_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  __acpp_int32 old_i = __acpp_sscp_atomic_load_i32(as, order, scope, (__acpp_int32*)ptr);
  __acpp_f32 old = *(__acpp_f32*)&old_i;
  do{
    if (old > x) return old;
  } while (!__acpp_sscp_cmp_exch_strong_i32(as, order, order, scope, (__acpp_int32*)ptr, (__acpp_int32*)&old, *(__acpp_int32*)&x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_max_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  __acpp_int64 old_i = __acpp_sscp_atomic_load_i64(as, order, scope, (__acpp_int64*)ptr);
  __acpp_f64 old = *(__acpp_f64*)&old_i;
  do{
    if (old > x) return old;
  } while (!__acpp_sscp_cmp_exch_strong_i64(as, order, order, scope, (__acpp_int64*)ptr, (__acpp_int64*)&old, *(__acpp_int64*)&x));
  return x;
}

