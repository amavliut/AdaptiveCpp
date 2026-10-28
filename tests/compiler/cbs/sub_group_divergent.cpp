// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=4 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=64 %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s

// Sub-group collectives called by only some lanes of a sub-group (undefined in SYCL, works on
// GPUs, used by GROMACS), which MaskDivergentSubgroupCollectivesPass makes every lane call.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

constexpr int num_groups = 4;

// f(item, out) per work-item, `fields` ints each, in global linear order
template <int D, class F>
std::vector<int> run(sycl::queue &q, sycl::range<D> wg, int fields, F f) {
  sycl::range<D> global = wg;
  global[0] *= num_groups;
  const std::size_t n = global.size() * fields;
  int *out = sycl::malloc_shared<int>(n, q);
  q.parallel_for(sycl::nd_range<D>{global, wg}, [=](sycl::nd_item<D> it) {
     f(it, out + it.get_global_linear_id() * fields);
   }).wait();
  std::vector<int> v(out, out + n);
  sycl::free(out, q);
  return v;
}

// sum of x = global id + 1 over sub-group s of the work-group starting at global id g0
int sg_sum(int g0, int s, int S, int W) {
  const int first = g0 + s * S + 1, n = std::min(S, W - s * S);
  return n * first + n * (n - 1) / 2;
}

int report(const char *name, int bad) {
  std::printf("%s: %s\n", name, bad ? "FAIL" : "ok");
  return bad;
}

// reductions, a scan and votes under `if (lane < 3)`: they cover exactly lanes 0-2
int collectives(sycl::queue &q, int S) {
  const int W = 25;
  auto v = run(q, sycl::range<1>{W}, 6, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int x = it.get_global_linear_id() + 1, l = sg.get_local_linear_id();
    int r[6] = {-1, -1, -1, -1, -1, -1};
    if (l < 3) {
      r[0] = sycl::reduce_over_group(sg, x, sycl::plus<int>{});
      r[1] = sycl::reduce_over_group(sg, x, sycl::minimum<int>{});
      r[2] = sycl::reduce_over_group(sg, x, sycl::maximum<int>{});
      r[3] = sycl::inclusive_scan_over_group(sg, x, sycl::plus<int>{});
      r[4] = sycl::any_of_group(sg, l == 3); // only a lane that skipped the branch would say yes
      r[5] = sycl::all_of_group(sg, l < 3);
    }
    for (int k = 0; k < 6; ++k)
      o[k] = r[k];
  });
  int bad = 0;
  for (std::size_t i = 0; i < v.size() / 6; ++i) {
    const int lid = i % W, l = lid % S, n = std::min(S, W - lid / S * S), p = std::min(3, n);
    const int b = i - l + 1, *o = &v[6 * i]; // b: x of lane 0 of this sub-group
    const int e[6] = {p * b + p * (p - 1) / 2, b, b + p - 1, (l + 1) * b + l * (l + 1) / 2, 0, 1};
    for (int k = 0; k < 6; ++k)
      bad += o[k] != (l < 3 ? e[k] : -1);
  }
  return report("collectives", bad);
}

// GROMACS's shift-force shape: two shuffles under a row-dependent `if` (rows 0-2 of a 5x5
// work-group, so a sub-group can straddle rows 2 and 3), then a collective by all lanes, which
// must still see the whole sub-group
int shuffles(sycl::queue &q, int S) {
  const int W = 25;
  auto v = run(q, sycl::range<2>{5, 5}, 2, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int x = it.get_global_linear_id() + 1;
    int f = x;
    if (it.get_local_id(0) < 3) {
      f += sycl::shift_group_left(sg, f, 1);
      f += sycl::shift_group_left(sg, f, 2);
    }
    o[0] = f;
    o[1] = sycl::reduce_over_group(sg, x, sycl::plus<int>{});
  });
  int bad = 0;
  for (std::size_t i = 0; i < v.size() / 2; ++i) {
    const int lid = i % W, s = lid / S, l = lid % S, n = std::min(S, W - s * S), x = i + 1;
    // defined only if lanes l..l+3 all took the branch and are in this sub-group
    if (lid >= 15)
      bad += v[2 * i] != x;
    else if (lid + 3 < 15 && l + 3 < n)
      bad += v[2 * i] != 4 * x + 6;
    bad += v[2 * i + 1] != sg_sum(i - lid, s, S, W);
  }
  return report("shuffles", bad);
}

// GROMACS's nonbonded shape at cluster size 8 (S = 64): a lane-dependent `continue` skips a whole
// inner loop holding shuffles (here per quad of lanes), with another `continue` inside. The first
// lane of a quad sums its quad, whatever the other quads do.
int loop(sycl::queue &q, int S) {
  const int W = 32, nk = S > 0 ? 2 : 3; // inner trip count unknown at compile time
  const auto quad = [](int j, int lid, int g) { return (j * 5 + lid / 4 * 3 + g) % 4; };
  auto v = run(q, sycl::range<1>{W}, 1, [=](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int lid = it.get_local_linear_id(), g = it.get_group_linear_id();
    int acc = 0;
    for (int j = 0; j < 6; ++j) {
      const int m = quad(j, lid, g);
      if (!m)
        continue;
      for (int k = 0; k < nk; ++k) {
        if (!((m >> k) & 1))
          continue;
        int f = (j + 1) * (k + 1) + lid;
        f += sycl::shift_group_left(sg, f, 1);
        f += sycl::shift_group_left(sg, f, 2);
        acc += f;
      }
    }
    o[0] = acc;
  });
  int bad = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const int lid = i % W, g = i / W;
    if (lid % 4 || S < 4) // defined for the first lane of a quad inside one sub-group
      continue;
    int e = 0;
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < nk; ++k)
        if (quad(j, lid, g) >> k & 1)
          e += 4 * (j + 1) * (k + 1) + 4 * lid + 6;
    bad += v[i] != e;
  }
  return report("loop", bad);
}

// loops with a lane-dependent trip count under a lane-dependent `if`, which lanes 3, 7, ... skip:
// lane l runs l % 4 iterations of a loop testing in front, l % 4 + 1 of one testing at the end;
// each iteration counts the lanes in it
int tripcount(sycl::queue &q, int S) {
  const int W = 32;
  auto v = run(q, sycl::range<1>{W}, 2, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int t = sg.get_local_linear_id() % 4;
    int r[2] = {-1, -1};
    if (t < 3) {
      r[0] = 0;
      for (int k = 0; k < t; ++k)
        r[0] += sycl::reduce_over_group(sg, 1, sycl::plus<int>{});
    }
    if (t < 3) {
      int k = 0;
      r[1] = 0;
      do
        r[1] += sycl::reduce_over_group(sg, 1, sycl::plus<int>{});
      while (k++ < t);
    }
    o[0] = r[0];
    o[1] = r[1];
  });
  int bad = 0;
  for (std::size_t i = 0; i < v.size() / 2; ++i) {
    const int lid = i % W, t = lid % S % 4, n = std::min(S, W - lid / S * S);
    for (int f = 0; f < 2; ++f) { // f = 1: the loop testing at the end, one more iteration
      int e = t < 3 ? 0 : -1;
      for (int k = 0; t < 3 && k < t + f; ++k)
        for (int m = 0; m < n; ++m) // lane m is in iteration k
          e += k < m % 4 + f && m % 4 < 3;
      bad += v[2 * i + f] != e;
    }
  }
  return report("tripcount", bad);
}

int main() {
  sycl::queue q;
  const int S = q.get_device().get_info<sycl::info::device::sub_group_sizes>().at(0);
  return (collectives(q, S) + shuffles(q, S) + loop(q, S) + tripcount(q, S)) ? 1 : 0;
}

// CHECK: collectives: ok
// CHECK: shuffles: ok
// CHECK: loop: ok
// CHECK: tripcount: ok
