// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=1 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK
// RUN: %if rv %{ rm -rf %t.db %}
// RUN: %if rv %{ env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db ACPP_DEBUG_LEVEL=3 %t 2>&1 | FileCheck %s --check-prefixes=VEC,OK %}
// RUN: %if rv %{ rm -rf %t.db %}
// RUN: %if rv %{ env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=16 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK %}

// Atomics by address pattern across the lanes of a sub-group (overview, Part C): one address in
// all lanes, neighbours (a float3 array at a 4-byte offset, so 8-byte words straddle elements),
// duplicates, local memory, partial masks and partial sub-groups (work-groups of 20). Results are
// order-independent: exact sums of small integers, returned values a permutation, untouched
// neighbours (-0.0, a NaN payload) bit for bit. Prefix VEC checks that the kernels run vectorized.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include <sycl/sycl.hpp>

constexpr int WG = 20, N = 7 * WG;

template <class T, sycl::memory_scope Scope = sycl::memory_scope::device,
          sycl::access::address_space Space = sycl::access::address_space::global_space>
using aref = sycl::atomic_ref<T, sycl::memory_order::relaxed, Scope, Space>;

void report(const char *Name, bool Ok) {
  std::printf("%s: %s\n", Name, Ok ? "ok" : "FAIL");
  std::fflush(stdout);
}

uint32_t bits(float F) {
  uint32_t B;
  std::memcpy(&B, &F, 4);
  return B;
}

// VEC: [SubgroupVectorizer] {{.*}}7counterIiE{{.*}}: vectorized at width 8
// VEC: [SubgroupVectorizer] {{.*}}7counterIfE{{.*}}: vectorized at width 8
// OK: at_counter: ok
// one address in every lane; the returned values are 0 .. N-1 in some order
template <class T> bool counter(sycl::queue &q) {
  T *C = sycl::malloc_shared<T>(1, q);
  T *Old = sycl::malloc_shared<T>(N, q);
  *C = 0;
  q.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    Old[i] = aref<T>(*C).fetch_add(T(1));
  }).wait();
  std::vector<T> V(Old, Old + N);
  std::sort(V.begin(), V.end());
  bool Ok = *C == T(N);
  for (int i = 0; i < N; ++i)
    Ok &= V[i] == T(i);
  sycl::free(C, q);
  sycl::free(Old, q);
  return Ok;
}
void at_counter(sycl::queue &q) { report("at_counter", counter<int>(q) && counter<float>(q)); }

// VEC: [SubgroupVectorizer] {{.*}}at_maskedRN{{.*}}: vectorized at width 8
// OK: at_masked: ok
// a partial mask: lanes with i % 3 != 0 add i (int) and i (float) to one address each
void at_masked(sycl::queue &q) {
  int *SI = sycl::malloc_shared<int>(1, q);
  float *SF = sycl::malloc_shared<float>(1, q);
  *SI = 0;
  *SF = 0.f;
  q.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i % 3) {
      aref<int>(*SI).fetch_add(i);
      aref<float>(*SF).fetch_add(float(i));
    }
  }).wait();
  int E = 0;
  for (int i = 0; i < N; ++i)
    E += i % 3 ? i : 0;
  report("at_masked", *SI == E && *SF == float(E));
  sycl::free(SI, q);
  sycl::free(SF, q);
}

// VEC: [SubgroupVectorizer] {{.*}}at_scatter3RN{{.*}}: vectorized at width 8
// OK: at_scatter3: ok
// GROMACS's j-force shape: lanes 0-2 of every 4 add x, y, z of atom k = i / 4, three times;
// the float3 array starts 4 bytes after an 8-byte boundary; slot 0 (-0.0) and the last slot (a
// NaN payload) are never written and must keep their bits
void at_scatter3(sycl::queue &q) {
  constexpr int M = N / 4, Len = 1 + 3 * M + 1;
  float *F = sycl::malloc_shared<float>(Len + 1, q) + 1; // 8-byte aligned + 4
  const float NaN = sycl::bit_cast<float>(0x7fc12345u);
  for (int s = 0; s < Len; ++s)
    F[s] = 0.f;
  F[0] = -0.f;
  F[Len - 1] = NaN;
  q.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0), c = i % 4, k = i / 4;
    for (int r = 0; r < 3; ++r)
      if (c < 3)
        aref<float>(F[1 + 3 * k + c]).fetch_add(float(3 * k + c + 1));
  }).wait();
  bool Ok = bits(F[0]) == bits(-0.f) && bits(F[Len - 1]) == bits(NaN);
  for (int k = 0; k < M; ++k)
    for (int c = 0; c < 3; ++c)
      Ok &= F[1 + 3 * k + c] == 3.f * float(3 * k + c + 1);
  report("at_scatter3", Ok);
  sycl::free(F - 1, q);
}

// VEC: [SubgroupVectorizer] {{.*}}at_histogramRN{{.*}}: vectorized at width 8
// OK: at_histogram: ok
// duplicates: 8 int bins hit by i * 7 % 8 (+2, then -1), 3 float bins by i % 3
void at_histogram(sycl::queue &q) {
  int *BI = sycl::malloc_shared<int>(8, q);
  float *BF = sycl::malloc_shared<float>(3, q);
  std::fill(BI, BI + 8, 0);
  std::fill(BF, BF + 3, 0.f);
  q.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    aref<int>(BI[i * 7 % 8]).fetch_add(2);
    aref<int>(BI[i * 7 % 8]).fetch_sub(1);
    aref<float>(BF[i % 3]).fetch_add(1.f);
  }).wait();
  bool Ok = true;
  for (int b = 0; b < 8; ++b) {
    int E = 0;
    for (int i = 0; i < N; ++i)
      E += i * 7 % 8 == b;
    Ok &= BI[b] == E;
  }
  for (int b = 0; b < 3; ++b)
    Ok &= BF[b] == float((N - b + 2) / 3);
  report("at_histogram", Ok);
  sycl::free(BI, q);
  sycl::free(BF, q);
}

// VEC: [SubgroupVectorizer] {{.*}}at_minmaxRN{{.*}}: vectorized at width 8
// OK: at_minmax: ok
void at_minmax(sycl::queue &q) {
  int *R = sycl::malloc_shared<int>(2, q);
  R[0] = 1 << 30;
  R[1] = -(1 << 30);
  q.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
    const int v = int(it.get_global_id(0)) * 37 % 101 - 50;
    aref<int>(R[0]).fetch_min(v);
    aref<int>(R[1]).fetch_max(v);
  }).wait();
  int Lo = 1 << 30, Hi = -(1 << 30);
  for (int i = 0; i < N; ++i)
    Lo = std::min(Lo, i * 37 % 101 - 50), Hi = std::max(Hi, i * 37 % 101 - 50);
  report("at_minmax", R[0] == Lo && R[1] == Hi);
  sycl::free(R, q);
}

// VEC: [SubgroupVectorizer] {{.*}}at_local_histRN{{.*}}: vectorized at width 8
// OK: at_local_hist: ok
// local memory, work-group scope: 16 bins per work-group, then one device-scope add per bin
void at_local_hist(sycl::queue &q) {
  int *Out = sycl::malloc_shared<int>(16, q);
  std::fill(Out, Out + 16, 0);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int> Bins{16, h};
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const int l = it.get_local_id(0), i = it.get_global_id(0);
      if (l < 16)
        Bins[l] = 0;
      sycl::group_barrier(it.get_group());
      aref<int, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(
          Bins[i % 16])
          .fetch_add(1);
      sycl::group_barrier(it.get_group());
      if (l < 16)
        aref<int>(Out[l]).fetch_add(Bins[l]);
    });
  }).wait();
  bool Ok = true;
  for (int b = 0; b < 16; ++b) {
    int E = 0;
    for (int i = 0; i < N; ++i)
      E += i % 16 == b;
    Ok &= Out[b] == E;
  }
  report("at_local_hist", Ok);
  sycl::free(Out, q);
}

int main() {
  sycl::queue q;
  at_counter(q);
  at_masked(q);
  at_scatter3(q);
  at_histogram(q);
  at_minmax(q);
  at_local_hist(q);
}
