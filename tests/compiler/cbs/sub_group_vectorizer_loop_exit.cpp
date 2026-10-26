// REQUIRES: rv
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db ACPP_DEBUG_LEVEL=3 %t 2>&1 | FileCheck %s --check-prefixes=RV,OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=0 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK

// A lane-dependent loop with an early exit, then a load with the loop's value: each lane needs the
// value it had at its own exit, which RV keeps in LCSSA phis of the loop's exit blocks. Splitting
// the critical exit edges after LCSSA moved those phis out of the exit blocks, and the value became
// undef (sycl_tests usm_tests/shared_btree_search, shared_trie_search). Built with -O3: the shape
// comes from the optimized stage-1 IR.

#include <cstdio>

#include <sycl/sycl.hpp>

int walk(const int *next, const int *val, const int *p, int n) {
  int cur = 0;
  for (int k = 0; k < n; ++k) {
    cur = next[cur * 2 + (p[k] & 1)];
    if (cur < 0)
      return -1;
  }
  return val[cur];
}

// RV: [SubgroupVectorizer] {{.*}}: vectorized at width 8
// OK: loop_exit: ok
int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  constexpr int Nodes = 7, N = 16, L = 4; // a binary tree: node c's children 2c+1, 2c+2
  int *next = sycl::malloc_shared<int>(2 * Nodes, q), *val = sycl::malloc_shared<int>(Nodes, q);
  int *p = sycl::malloc_shared<int>(N * L, q), *n = sycl::malloc_shared<int>(N, q);
  int *o = sycl::malloc_shared<int>(N, q);
  for (int c = 0; c < 2 * Nodes; ++c)
    next[c] = c + 1 < Nodes ? c + 1 : -1;
  for (int c = 0; c < Nodes; ++c)
    val[c] = 100 + c;
  for (int i = 0; i < N; ++i) {
    n[i] = i % 4;
    o[i] = -99;
    for (int k = 0; k < L; ++k)
      p[i * L + k] = (i >> (k + 1)) & 1;
  }
  q.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    o[i] = walk(next, val, p + i * L, n[i]);
  }).wait();
  int Bad = 0;
  for (int i = 0; i < N; ++i)
    if (o[i] != walk(next, val, p + i * L, n[i]) && Bad++ < 5)
      std::printf("loop_exit: item %d: %d, expected %d\n", i, o[i], walk(next, val, p + i * L, n[i]));
  std::printf("loop_exit: %s\n", Bad ? "FAIL" : "ok");
}
