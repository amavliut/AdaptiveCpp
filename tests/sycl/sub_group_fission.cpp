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

// Regression cases for sub-group fission on the SSCP host backend. Run with
// ACPP_OPENMP_SUBGROUP_SIZE unset and > 1; S always comes from the device query.
#include "sycl_test_suite.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

BOOST_FIXTURE_TEST_SUITE(sub_group_fission_tests, reset_device_fixture)

namespace {
constexpr std::size_t num_groups = 4;

int sg_size(sycl::queue &q) {
  return q.get_device().get_info<sycl::info::device::sub_group_sizes>()[0];
}

// Runs f(item, out) per work-item; `fields` ints per work-item, in global
// linear order (work-groups are contiguous: groups are stacked along dim 0).
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

// sub-groups partition the linearized work-group; last may be partial.
template <int D> void check_geometry(sycl::range<D> wg) {
  sycl::queue q;
  const int S = sg_size(q), W = wg.size();
  auto v = run(q, wg, 5, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    o[0] = sg.get_max_local_range()[0];
    o[1] = sg.get_group_linear_id();
    o[2] = sg.get_local_linear_id();
    o[3] = sg.get_local_linear_range();
    o[4] = sg.get_group_linear_range();
  });
  for (std::size_t i = 0; i < v.size() / 5; ++i) {
    const int lid = i % W, *o = &v[5 * i];
    BOOST_TEST(o[0] == S);
    BOOST_TEST(o[1] == lid / S);
    BOOST_TEST(o[2] == lid % S);
    BOOST_TEST(o[3] == std::min(S, W - lid / S * S));
    BOOST_TEST(o[4] == (W + S - 1) / S);
  }
}

// collectives called by the whole work-group (uniform control flow).
template <int D> void check_collectives(sycl::range<D> wg) {
  sycl::queue q;
  const int S = sg_size(q), W = wg.size();
  auto v = run(q, wg, 5, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int x = it.get_global_linear_id() + 1;
    const int l = sg.get_local_linear_id(), n = sg.get_local_linear_range();
    o[0] = sycl::reduce_over_group(sg, x, sycl::plus<int>{});
    o[1] = sycl::group_broadcast(sg, x, 0);
    o[2] = sycl::select_from_group(sg, x, (l + 1) % n);
    o[3] = sycl::shift_group_left(sg, x, 1);
    o[4] = sycl::inclusive_scan_over_group(sg, x, sycl::plus<int>{});
  });
  for (std::size_t i = 0; i < v.size() / 5; ++i) {
    // lane k of this sub-group holds x = base + k + 1
    const int lid = i % W, l = lid % S, base = i - l, n = std::min(S, W - lid / S * S);
    const int *o = &v[5 * i];
    BOOST_TEST(o[0] == n * (base + 1) + n * (n - 1) / 2);
    BOOST_TEST(o[1] == base + 1);
    BOOST_TEST(o[2] == base + (l + 1) % n + 1);
    if (l + 1 < n)
      BOOST_TEST(o[3] == base + l + 2);
    BOOST_TEST(o[4] == (l + 1) * (base + 1) + l * (l + 1) / 2);
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(sg_size_follows_env) {
  sycl::queue q;
  if (q.get_device().get_backend() != sycl::backend::omp)
    return;
  const char *env = std::getenv("ACPP_OPENMP_SUBGROUP_SIZE");
  BOOST_TEST(sg_size(q) == (env ? std::stoi(env) : 1));
}

// 25 and 5x5 are not multiples of S > 1: the last sub-group is partial.
BOOST_AUTO_TEST_CASE(sg_geometry) {
  check_geometry(sycl::range<1>{25});
  check_geometry(sycl::range<2>{5, 5});
}

BOOST_AUTO_TEST_CASE(sg_collectives_uniform) {
  check_collectives(sycl::range<1>{25});
  check_collectives(sycl::range<2>{5, 5});
}

// `group_barrier(wg); group_barrier(sg);` back to back must keep the
// work-group barrier (each lane reads a value written by another sub-group).
BOOST_AUTO_TEST_CASE(sg_wg_then_sg_barrier) {
  sycl::queue q;
  const int W = 25;
  auto v = run(q, sycl::range<1>{W}, 2, [](auto it, int *o) {
    const int l = it.get_local_linear_id(), w = it.get_local_range().size();
    o[0] = it.get_global_linear_id() + 1;
    sycl::group_barrier(it.get_group());
    sycl::group_barrier(it.get_sub_group());
    o[1] = o[2 * ((l + 1) % w - l)]; // next lane's o[0], wrapping within the group
  });
  for (std::size_t i = 0; i < v.size() / 2; ++i) {
    const int lid = i % W;
    BOOST_TEST(v[2 * i + 1] == int(i) - lid + (lid + 1) % W + 1);
  }
}

// after a sub-group barrier the span runs in the small loop; the
// per-dimension local ids rebuilt there must give back the linear local id.
template <int D> void check_ids_after_sg_barrier(sycl::range<D> wg) {
  sycl::queue q;
  const int W = wg.size();
  auto v = run(q, wg, 1, [](auto it, int *o) {
    sycl::group_barrier(it.get_sub_group());
    int lin = 0;
    for (int d = 0; d < D; ++d)
      lin = lin * it.get_local_range(d) + it.get_local_id(d);
    o[0] = lin;
  });
  for (std::size_t i = 0; i < v.size(); ++i)
    BOOST_TEST(v[i] == int(i % W));
}

BOOST_AUTO_TEST_CASE(sg_ids_after_sg_barrier) {
  check_ids_after_sg_barrier(sycl::range<2>{5, 5});
  check_ids_after_sg_barrier(sycl::range<3>{3, 3, 3});
}

// sum of x over sub-group s of the work-group starting at global id g0 (x = global id + 1)
int sg_sum(int g0, int s, int S, int W) {
  const int first = g0 + s * S + 1, n = std::min(S, W - s * S);
  return n * first + n * (n - 1) / 2;
}

// between two work-group barriers only even sub-groups run a
// sub-group collective; odd ones skip straight to the second barrier.
BOOST_AUTO_TEST_CASE(sg_divergent_collective) {
  sycl::queue q;
  const int S = sg_size(q), W = 25;
  auto v = run(q, sycl::range<1>{W}, 1, [](auto it, int *o) {
    auto sg = it.get_sub_group();
    int x = it.get_global_linear_id() + 1;
    sycl::group_barrier(it.get_group());
    if (sg.get_group_linear_id() % 2 == 0)
      x = sycl::reduce_over_group(sg, x, sycl::plus<int>{});
    sycl::group_barrier(it.get_group());
    o[0] = x;
  });
  for (std::size_t i = 0; i < v.size(); ++i) {
    const int s = (i % W) / S, g0 = i - i % W;
    BOOST_TEST(v[i] == (s % 2 == 0 ? sg_sum(g0, s, S, W) : int(i) + 1));
  }
}

// sub-group reduce, work-group barrier, sub-group reduce of the next
// sub-group's partial: every sub-group must finish the first reduce before
// any reads past the work-group barrier.
BOOST_AUTO_TEST_CASE(sg_reduce_wg_barrier_reduce) {
  sycl::queue q;
  const int S = sg_size(q), W = 25;
  int *partial = sycl::malloc_shared<int>(num_groups * W, q);
  auto v = run(q, sycl::range<1>{W}, 1, [partial, W](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int g = it.get_group_linear_id(), s = sg.get_group_linear_id();
    const int a = sycl::reduce_over_group(sg, int(it.get_global_linear_id()) + 1,
                                          sycl::plus<int>{});
    if (sg.leader())
      partial[g * W + s] = a;
    sycl::group_barrier(it.get_group());
    const int next = (s + 1) % sg.get_group_linear_range();
    o[0] = sycl::reduce_over_group(sg, partial[g * W + next], sycl::plus<int>{});
  });
  sycl::free(partial, q);
  const int ns = (W + S - 1) / S;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const int s = (i % W) / S, g0 = i - i % W, n = std::min(S, W - s * S);
    BOOST_TEST(v[i] == n * sg_sum(g0, (s + 1) % ns, S, W));
  }
}

// a uniform loop counter around a work-group barrier and a sub-group
// collective. Sub-groups run in turns, so the counter must not live in one
// shared slot (sub-group 0 would advance it before sub-group 1 reads it).
// The trip count is read from memory so the loop cannot be fully unrolled.
BOOST_AUTO_TEST_CASE(sg_uniform_counter_across_turns) {
  sycl::queue q;
  const int S = sg_size(q), W = 25;
  int *trips = sycl::malloc_shared<int>(1, q);
  *trips = 3;
  auto v = run(q, sycl::range<1>{W}, 1, [trips](auto it, int *o) {
    auto sg = it.get_sub_group();
    const int x = it.get_local_linear_id() + 1;
    int acc = 0;
    for (int k = 0; k < *trips; ++k) {
      sycl::group_barrier(it.get_group());
      acc += sycl::reduce_over_group(sg, (k + 1) * x, sycl::plus<int>{});
    }
    o[0] = acc;
  });
  sycl::free(trips, q);
  for (std::size_t i = 0; i < v.size(); ++i) {
    const int s = (i % W) / S;
    BOOST_TEST(v[i] == (1 + 2 + 3) * sg_sum(0, s, S, W)); // sum of x over the sub-group, k = 0..2
  }
}

// MaskDivergentSubgroupCollectivesPass: collectives called by only some lanes of a sub-group
// (undefined in SYCL, works on GPUs, used by GROMACS). Reductions, scans and votes must cover
// exactly the lanes that took the branch (here lanes 0-2 of each sub-group).
BOOST_AUTO_TEST_CASE(sg_lane_divergent_collectives) {
  sycl::queue q;
  if (q.get_device().get_backend() != sycl::backend::omp)
    return;
  const int S = sg_size(q), W = 25;
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
  for (std::size_t i = 0; i < v.size() / 6; ++i) {
    const int lid = i % W, l = lid % S, n = std::min(S, W - lid / S * S), p = std::min(3, n);
    const int b = i - l + 1, *o = &v[6 * i]; // b: x of lane 0 of this sub-group
    const int e[6] = {p * b + p * (p - 1) / 2, b, b + p - 1, (l + 1) * b + l * (l + 1) / 2, 0, 1};
    for (int k = 0; k < 6; ++k)
      BOOST_TEST(o[k] == (l < 3 ? e[k] : -1));
  }
}

// MaskDivergentSubgroupCollectivesPass, GROMACS's shift-force shape: two shuffles under a
// lane-dependent branch (rows 0-2 of a 5x5 work-group, so a sub-group can straddle rows 2 and 3),
// then a collective by all lanes, which must still see a sub-group in step.
BOOST_AUTO_TEST_CASE(sg_lane_divergent_shuffles) {
  sycl::queue q;
  if (q.get_device().get_backend() != sycl::backend::omp)
    return;
  const int S = sg_size(q), W = 25;
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
  for (std::size_t i = 0; i < v.size() / 2; ++i) {
    const int lid = i % W, s = lid / S, l = lid % S, n = std::min(S, W - s * S);
    const int x = i + 1;
    // defined only if lanes l..l+3 all took the branch and are in this sub-group
    if (lid >= 15)
      BOOST_TEST(v[2 * i] == x);
    else if (lid + 3 < 15 && l + 3 < n)
      BOOST_TEST(v[2 * i] == 4 * x + 6);
    BOOST_TEST(v[2 * i + 1] == sg_sum(i - lid, s, S, W));
  }
}

// MaskDivergentSubgroupCollectivesPass, GROMACS's nonbonded shape at S = 64: a lane-dependent
// `continue` skips a whole inner loop holding shuffles (here per quad of lanes), with another
// `continue` inside. The first lane of a quad sums its quad, whatever the other quads do.
BOOST_AUTO_TEST_CASE(sg_lane_divergent_loop) {
  sycl::queue q;
  if (q.get_device().get_backend() != sycl::backend::omp)
    return;
  const int S = sg_size(q), W = 32, nk = S > 0 ? 2 : 3; // inner trip count unknown at compile time
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
  for (std::size_t i = 0; i < v.size(); ++i) {
    const int lid = i % W, g = i / W;
    if (lid % 4 || S < 4) // defined for the first lane of a quad inside one sub-group
      continue;
    int e = 0;
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < nk; ++k)
        if (quad(j, lid, g) >> k & 1)
          e += 4 * (j + 1) * (k + 1) + 4 * lid + 6;
    BOOST_TEST(v[i] == e);
  }
}

BOOST_AUTO_TEST_SUITE_END()
