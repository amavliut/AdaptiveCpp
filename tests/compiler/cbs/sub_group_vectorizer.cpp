// REQUIRES: rv
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db ACPP_DEBUG_LEVEL=3 %t 2>&1 | FileCheck %s --check-prefixes=RV,OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=4 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=16 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=0 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK

// SubgroupVectorizerPass (SSCP host, built WITH_RV, ACPP_OPENMP_VECTORIZE=1): a kernel runs one
// sub-group per SIMD vector. Check prefixes: RV (the pass vectorized the kernel or kept it scalar,
// and why), OK (results right with and without vectorization; a collective reached by part of a
// sub-group is masked on both paths).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

#include <sycl/sycl.hpp>

constexpr size_t N = 64, WG = 32; // 1-D; 2-D: 8 x 8, work-groups of 4 x 4

struct In {
  const float *x, *y;
  const sycl::float2 *xy;
};

// K(handler, out) submits the kernel; ref(i, S): out[i], nullopt if unspecified
template <class K, class R> void check(sycl::queue &q, const char *Name, K Kernel, R ref) {
  const size_t S = q.get_device().get_info<sycl::info::device::sub_group_sizes>()[0];
  float *o = sycl::malloc_shared<float>(N, q);
  std::fill(o, o + N, -1.f);
  q.submit([&](sycl::handler &h) { Kernel(h, o); }).wait();
  int Bad = 0;
  for (size_t i = 0; i < N; ++i)
    if (auto e = ref(i, S); e && std::abs(o[i] - *e) > 1e-4f * (1 + std::abs(*e)) && Bad++ < 5)
      std::printf("%s: item %zu: %g, expected %g\n", Name, i, o[i], *e);
  std::printf("%s: %s\n", Name, Bad ? "FAIL" : "ok");
  std::fflush(stdout);
  sycl::free(o, q);
}

using Opt = std::optional<float>;
size_t lane(size_t i, size_t S) { return i % WG % S; }

// RV: [SubgroupVectorizer] {{.*}}vec_arithRN{{.*}}: vectorized at width 8, 4 sub-groups per work-group
// OK: vec_arith: ok
void vec_arith(sycl::queue &q, In in) {
  check(q, "vec_arith", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      o[i] = 3.f * in.x[i] + in.y[i];
    });
  }, [=](size_t i, size_t) -> Opt { return 3.f * in.x[i] + in.y[i]; });
}

// RV: [SubgroupVectorizer] {{.*}}vec_shift_reduceRN{{.*}}: vectorized at width 8
// OK: vec_shift_reduce: ok
void vec_shift_reduce(sycl::queue &q, In in) {
  check(q, "vec_shift_reduce", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      auto sg = it.get_sub_group();
      o[i] = sycl::shift_group_left(sg, in.x[i], 1) + sycl::reduce_over_group(sg, in.x[i], sycl::plus<float>{});
    });
  }, [=](size_t i, size_t S) -> Opt {
    const size_t l = lane(i, S);
    if (l + 1 >= S)
      return std::nullopt;
    float Sum = 0;
    for (size_t k = 0; k < S; ++k)
      Sum += in.x[i - l + k];
    return in.x[i + 1] + Sum;
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_div_storeRN{{.*}}: vectorized at width 8
// OK: vec_div_store: ok
void vec_div_store(sycl::queue &q, In in) { // lane-dependent branch: a masked store
  check(q, "vec_div_store", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      if (it.get_sub_group().get_local_linear_id() < 5)
        o[i] = 2.f * in.x[i];
    });
  }, [=](size_t i, size_t S) -> Opt { return lane(i, S) < 5 ? 2.f * in.x[i] : -1.f; });
}

// RV: [SubgroupVectorizer] {{.*}}vec_div_loopRN{{.*}}: vectorized at width 8
// OK: vec_div_loop: ok
void vec_div_loop(sycl::queue &q, In in) { // lane-dependent trip count: runs while any lane is active
  check(q, "vec_div_loop", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), l = it.get_sub_group().get_local_linear_id();
      float Sum = 0;
      for (size_t k = 0; k <= l; ++k)
        Sum += in.x[i - l + k];
      o[i] = Sum;
    });
  }, [=](size_t i, size_t S) -> Opt {
    float Sum = 0;
    for (size_t k = 0; k <= lane(i, S); ++k)
      Sum += in.x[i - lane(i, S) + k];
    return Sum;
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_div_reduceRN{{.*}}: vectorized at width 8
// OK: vec_div_reduce: ok
void vec_div_reduce(sycl::queue &q, In in) { // reduce by part of the sub-group: others add 0
  check(q, "vec_div_reduce", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      auto sg = it.get_sub_group();
      if (sg.get_local_linear_id() < 5)
        o[i] = sycl::reduce_over_group(sg, in.x[i], sycl::plus<float>{});
    });
  }, [=](size_t i, size_t S) -> Opt {
    const size_t l = lane(i, S);
    float Sum = 0;
    for (size_t k = 0; k < std::min<size_t>(5, S); ++k)
      Sum += in.x[i - l + k];
    return l < 5 ? Sum : -1.f;
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_local_barrierRN{{.*}}: vectorized at width 8
// OK: vec_local_barrier: ok
void vec_local_barrier(sycl::queue &q, In in) { // work-group barrier: once per sub-group
  check(q, "vec_local_barrier", [=](sycl::handler &h, float *o) {
    sycl::local_accessor<float, 1> Tmp{WG, h};
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), L = it.get_local_id(0);
      const float v = in.x[i];
      Tmp[L] = 2.f * v;
      sycl::group_barrier(it.get_group());
      o[i] = Tmp[WG - 1 - L] + v;
    });
  }, [=](size_t i, size_t) -> Opt {
    const size_t L = i % WG;
    return 2.f * in.x[i - L + WG - 1 - L] + in.x[i];
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_2dRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: vec_2d: ok
void vec_2d(sycl::queue &q, In in) { // GROMACS's shape: at S = 8 a sub-group spans 2 rows (x periodic)
  check(q, "vec_2d", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<2>{{8, 8}, {4, 4}}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_linear_id();
      auto sg = it.get_sub_group();
      o[i] = sycl::shift_group_left(sg, in.x[i], 1) + sycl::reduce_over_group(sg, in.x[i], sycl::plus<float>{});
    });
  }, [=](size_t i, size_t S) -> Opt {
    const size_t R = i / 8, C = i % 8, L = R % 4 * 4 + C % 4, l = L % S;
    auto At = [&](size_t K) { return in.x[(R / 4 * 4 + K / 4) * 8 + C / 4 * 4 + K % 4]; };
    if (l + 1 >= S)
      return std::nullopt;
    float Sum = 0;
    for (size_t k = L - l; k < L - l + S; ++k)
      Sum += At(k);
    return At(L + 1) + Sum;
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_float2_localRN{{.*}}: vectorized at width 8
// OK: vec_float2_local: ok
void vec_float2_local(sycl::queue &q, In in) { // float2 through global and local memory
  check(q, "vec_float2_local", [=](sycl::handler &h, float *o) {
    sycl::local_accessor<sycl::float2, 1> Tmp{WG, h};
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), L = it.get_local_id(0);
      const sycl::float2 a = in.xy[(i * 5 + 3) % N];
      Tmp[L] = in.xy[i];
      sycl::group_barrier(it.get_group());
      const sycl::float2 b = Tmp[(L * 3 + 1) % WG];
      o[i] = a[0] + 10.f * a[1] + 100.f * b[0] + 1000.f * b[1];
    });
  }, [=](size_t i, size_t) -> Opt {
    const size_t L = i % WG, A = (i * 5 + 3) % N, B = i - L + (L * 3 + 1) % WG;
    return in.x[A] + 10.f * in.y[A] + 100.f * in.x[B] + 1000.f * in.y[B];
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_float2_loopRN{{.*}}: vectorized at width 8
// OK: vec_float2_loop: ok
void vec_float2_loop(sycl::queue &q, In in) { // float2 carried across iterations, set per lane
  check(q, "vec_float2_loop", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), l = it.get_sub_group().get_local_linear_id();
      sycl::float2 c(0.f, 0.f);
      float Sum = 0;
      for (size_t k = 0; k < 4; ++k) {
        if ((l + k) % 3 != 0)
          c = in.xy[(i + k) % N];
        Sum += c[0] + 10.f * c[1];
      }
      o[i] = Sum;
    });
  }, [=](size_t i, size_t S) -> Opt {
    float C0 = 0, C1 = 0, Sum = 0;
    for (size_t k = 0; k < 4; ++k) {
      if ((lane(i, S) + k) % 3 != 0) {
        C0 = in.x[(i + k) % N];
        C1 = in.y[(i + k) % N];
      }
      Sum += C0 + 10.f * C1;
    }
    return Sum;
  });
}

// a private array with an aggregate initializer (a 24-byte memset `align 16`): RV keeps one copy
// per lane, 24 bytes apart, so the pass lowers the alignment (else aligned stores fault)
// RV: [SubgroupVectorizer] {{.*}}vec_private_arrayRN{{.*}}: vectorized at width 8
// OK: vec_private_array: ok
void vec_private_array(sycl::queue &q, In) {
  check(q, "vec_private_array", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      int r[6] = {-1, -1, -1, -1, -1, -1};
      r[i % 6] = int(i);
      o[i] = float(r[(i + 1) % 6] + 10 * r[i % 6]);
    });
  }, [=](size_t i, size_t) -> Opt { return float(-1 + 10 * int(i)); });
}

// GROMACS's i-force accumulators: a zeroed private array indexed by a uniform loop counter and
// updated under a divergent branch. Its memset is lowered, so RV lays it out element-major: one
// vector load and masked store per access instead of a gather and per-lane stores. The bound is
// read from memory (8), so the loop is not unrolled and the array not promoted before RV.
// RV: [SubgroupVectorizer] {{.*}}vec_private_accumRN{{.*}}: {{.*}} {{[1-9][0-9]*}} memset(s) lowered, {{[1-9][0-9]*}} private array(s) element-major
// RV: [SubgroupVectorizer] {{.*}}vec_private_accumRN{{.*}}: vectorized at width 8
// OK: vec_private_accum: ok
void vec_private_accum(sycl::queue &q, In in) {
  check(q, "vec_private_accum", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      const int n = sycl::min(int(in.y[6]) + 2, 8);
      float acc[8] = {};
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < n; ++k) {
          const float v = in.x[(i + 3 * j + k) % N];
          if (v > 6.f)
            acc[k] += v / (v - 5.f);
        }
      float r = 0.f;
      for (int k = 0; k < n; ++k)
        r += acc[k] * float(k + 1);
      o[i] = r;
    });
  }, [=](size_t i, size_t) -> Opt {
    const int n = std::min(int(in.y[6]) + 2, 8);
    float acc[8] = {};
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < n; ++k) {
        const float v = in.x[(i + 3 * j + k) % N];
        if (v > 6.f)
          acc[k] += v / (v - 5.f);
      }
    float r = 0.f;
    for (int k = 0; k < n; ++k)
      r += acc[k] * float(k + 1);
    return r;
  });
}

// a lane-decreasing index (stride -1) in work-groups of 24 (3 sub-groups at S = 8): RV used to
// build the lane offsets <0, -1, -2, ..> zero-extended (rv patch acpp_patch/contiguousVector.patch)
// RV: [SubgroupVectorizer] {{.*}}vec_reverseRN{{.*}}: vectorized at width 8, 3 sub-groups per work-group
// OK: vec_reverse: ok
void vec_reverse(sycl::queue &q, In in) {
  constexpr size_t W = 24;
  check(q, "vec_reverse", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{2 * W, W}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), L = it.get_local_id(0);
      o[i] = in.x[i - L + W - 1 - L] + in.y[i];
    });
  }, [=](size_t i, size_t) -> Opt {
    if (i >= 2 * W)
      return -1.f;
    const size_t L = i % W;
    return in.x[i - L + W - 1 - L] + in.y[i];
  });
}

// GROMACS's imask shape: 4 x 4 work-groups, S = 8, lid.x = lane % 4 is periodic for RV, so
// tidx / 8 looks lane dependent but is the same in a sub-group; pinned uniform, the mask loaded
// with it and the branches on its bits stay uniform (no masking)
// RV: [SubgroupVectorizer] {{.*}}vec_pinnedRN{{.*}}: {{[1-9][0-9]*}} lane-invariant value(s) pinned uniform
// RV: [SubgroupVectorizer] {{.*}}vec_pinnedRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: vec_pinned: ok
void vec_pinned(sycl::queue &q, In in) {
  check(q, "vec_pinned", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<2>{{8, 8}, {4, 4}}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_linear_id(), tidx = it.get_local_linear_id();
      const unsigned m = unsigned(in.y[it.get_group_linear_id() * 2 + tidx / 8]); // per sub-group
      float r = 0.f;
      for (unsigned b = 0; b < 3; ++b)
        if ((m >> b) & 1u)
          r += in.x[b * 16 + tidx];
      o[i] = r;
    });
  }, [=](size_t i, size_t) -> Opt {
    const size_t r0 = i / 8, c0 = i % 8, g = r0 / 4 * 2 + c0 / 4, tidx = r0 % 4 * 4 + c0 % 4;
    const unsigned m = unsigned(in.y[g * 2 + tidx / 8]);
    float r = 0.f;
    for (unsigned b = 0; b < 3; ++b)
      if ((m >> b) & 1u)
        r += in.x[b * 16 + tidx];
    return r;
  });
}

// GROMACS's prune kernel: the per-sub-group mask is also updated in the loop (a bit cleared when
// no lane of the sub-group needs it) under the uniform branch on that bit, whose join is also the
// one of a divergent branch inside; the joins are split, so the mask stays uniform: all branches
// but that one (without the split, the mask and the branches on it are varying)
// RV: [SubgroupVectorizer] {{.*}}vec_pinned_updateRN{{.*}}: {{[1-9][0-9]*}} lane-invariant value(s) pinned uniform, {{[1-9][0-9]*}} join(s) split, [[#U:]] of
// RV-SAME: [[#U+1]] branch(es) uniform
// RV: [SubgroupVectorizer] {{.*}}vec_pinned_updateRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: vec_pinned_update: ok
void vec_pinned_update(sycl::queue &q, In in) {
  check(q, "vec_pinned_update", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<2>{{8, 8}, {4, 4}}, [=](sycl::nd_item<2> it) {
      auto sg = it.get_sub_group();
      const size_t i = it.get_global_linear_id(), tidx = it.get_local_linear_id();
      unsigned m = unsigned(in.y[it.get_group_linear_id() * 2 + tidx / sg.get_max_local_range()[0]]) | 8u;
      float r = 0.f;
      for (unsigned b = 0; b < 4; ++b)
        if ((m >> b) & 1u) {
          const float v = in.x[b * 16 + tidx];
          if (!sycl::any_of_group(sg, v > 11.f))
            m &= ~(1u << b);
          if (v > 6.f) // divergent, too costly to become a select
            r += v / (v - 5.f) + sycl::sqrt(v);
        }
      o[i] = r + 100.f * float(m);
    });
  }, [=](size_t i, size_t S) -> Opt {
    const size_t r0 = i / 8, c0 = i % 8, g = r0 / 4 * 2 + c0 / 4, tidx = r0 % 4 * 4 + c0 % 4;
    unsigned m = unsigned(in.y[g * 2 + tidx / S]) | 8u;
    float r = 0.f;
    for (unsigned b = 0; b < 4; ++b)
      if ((m >> b) & 1u) {
        bool Any = false;
        for (size_t t = tidx / S * S; t < tidx / S * S + S; ++t)
          Any |= in.x[b * 16 + t] > 11.f;
        if (!Any)
          m &= ~(1u << b);
        const float v = in.x[b * 16 + tidx];
        if (v > 6.f)
          r += v / (v - 5.f) + std::sqrt(v);
      }
    return r + 100.f * float(m);
  });
}

// periodic and contiguous lanes through lid.x / lid.y (4 x 4 work-groups, S = 8: lid.x =
// <0,1,2,3,0,1,2,3>, lid.y two values): the float2 fields of 4 neighbouring elements share one
// window load, a masked read of 2 elements, a contiguous store to local memory and its reversed
// read become window loads / stores instead of gathers / scatters
// RV: [SubgroupVectorizer] {{.*}}vec_periodicRN{{.*}}: {{.*}} {{[1-9][0-9]*}} of {{[0-9]+}} gather(s) and {{[1-9][0-9]*}} of {{[0-9]+}} scatter(s) as windows
// RV: [SubgroupVectorizer] {{.*}}vec_periodicRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: vec_periodic: ok
void vec_periodic(sycl::queue &q, In in) {
  auto R = [=](size_t g, size_t t) { // item t (= ly·4 + lx) of work-group g
    const size_t lx = t % 4, ly = t / 4, i = (g / 2 * 4 + ly) * 8 + g % 2 * 4 + lx;
    float r = in.xy[g * 4 + lx].x() + 2.f * in.xy[g * 4 + lx].y();
    if (in.x[i] > 6.f)
      r += in.y[g * 4 + ly];
    return r;
  };
  check(q, "vec_periodic", [=](sycl::handler &h, float *o) {
    sycl::local_accessor<float, 1> Tmp{16, h};
    h.parallel_for(sycl::nd_range<2>{{8, 8}, {4, 4}}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_linear_id(), g = it.get_group_linear_id();
      const size_t lx = it.get_local_id(1), ly = it.get_local_id(0);
      const sycl::float2 a = in.xy[g * 4 + lx];
      float r = a.x() + 2.f * a.y();
      if (in.x[i] > 6.f)
        r += in.y[g * 4 + ly];
      Tmp[ly * 4 + lx] = r;
      sycl::group_barrier(it.get_group());
      o[i] = r + 0.5f * Tmp[15 - (ly * 4 + lx)];
    });
  }, [=](size_t i, size_t) -> Opt {
    const size_t r0 = i / 8, c0 = i % 8, g = r0 / 4 * 2 + c0 / 4, t = r0 % 4 * 4 + c0 % 4;
    return R(g, t) + 0.5f * R(g, 15 - t);
  });
}

// work-groups of 12: at S = 8 sub-groups of 8 and 4 lanes; the missing lanes are masked off
// RV: [SubgroupVectorizer] {{.*}}vec_partialRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group (the last partial)
// OK: vec_partial: ok
void vec_partial(sycl::queue &q, In in) { // reduce, votes and the range count existing lanes only
  constexpr size_t W = 12;
  check(q, "vec_partial", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{4 * W, W}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      auto sg = it.get_sub_group();
      const size_t l = sg.get_local_linear_id();
      o[i] = sycl::reduce_over_group(sg, in.x[i], sycl::plus<float>{}) + 1000.f * sg.get_local_linear_range() +
             100.f * sycl::any_of_group(sg, l >= 6) + 10.f * sycl::all_of_group(sg, l < 4);
    });
  }, [=](size_t i, size_t S) -> Opt {
    if (i >= 4 * W)
      return -1.f;
    const size_t L = i % W, l = L % S, n = std::min(S, W - L / S * S);
    float Sum = 0;
    for (size_t k = 0; k < n; ++k)
      Sum += in.x[i - l + k];
    return Sum + 1000.f * n + 100.f * (n > 6) + 10.f * (n <= 4);
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_partial_barrierRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group (the last partial)
// OK: vec_partial_barrier: ok
void vec_partial_barrier(sycl::queue &q, In in) { // a missing lane would write past Tmp
  constexpr size_t W = 12;
  check(q, "vec_partial_barrier", [=](sycl::handler &h, float *o) {
    sycl::local_accessor<float, 1> Tmp{W, h};
    h.parallel_for(sycl::nd_range<1>{4 * W, W}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0), L = it.get_local_id(0);
      Tmp[L] = 2.f * in.x[i];
      sycl::group_barrier(it.get_group());
      o[i] = Tmp[W - 1 - L] + in.x[i];
    });
  }, [=](size_t i, size_t) -> Opt {
    if (i >= 4 * W)
      return -1.f;
    const size_t L = i % W;
    return 2.f * in.x[i - L + W - 1 - L] + in.x[i];
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_partial_2dRN{{.*}}: vectorized at width 8, 4 sub-groups per work-group (the last partial)
// OK: vec_partial_2d: ok
void vec_partial_2d(sycl::queue &q, In in) { // 5 x 5 work-groups: sub-groups cross rows, the last has 1 lane
  constexpr size_t W = 25;
  check(q, "vec_partial_2d", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<2>{{10, 5}, {5, 5}}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_linear_id();
      auto sg = it.get_sub_group();
      o[i] = sycl::shift_group_left(sg, in.x[i], 1) + sycl::reduce_over_group(sg, in.x[i], sycl::plus<float>{}) +
             1000.f * sg.get_local_linear_range();
    });
  }, [=](size_t i, size_t S) -> Opt {
    if (i >= 2 * W)
      return -1.f;
    const size_t L = i % W, l = L % S, n = std::min(S, W - L / S * S);
    if (l + 1 >= n)
      return std::nullopt;
    float Sum = 0;
    for (size_t k = 0; k < n; ++k)
      Sum += in.x[i - l + k];
    return in.x[i + 1] + Sum + 1000.f * n;
  });
}

// RV: [SubgroupVectorizer] {{.*}}vec_wg_reduceRN{{.*}} stays scalar: __acpp_sscp_work_group_reduce_f32 is not supported
// OK: vec_wg_reduce: ok
void vec_wg_reduce(sycl::queue &q, In in) { // work-group collectives: not mapped
  check(q, "vec_wg_reduce", [=](sycl::handler &h, float *o) {
    h.parallel_for(sycl::nd_range<1>{N, WG}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      o[i] = sycl::reduce_over_group(it.get_group(), in.x[i], sycl::plus<float>{});
    });
  }, [=](size_t i, size_t) -> Opt {
    float Sum = 0;
    for (size_t k = 0; k < WG; ++k)
      Sum += in.x[i - i % WG + k];
    return Sum;
  });
}

int main() {
  sycl::queue q;
  float *x = sycl::malloc_shared<float>(N, q), *y = sycl::malloc_shared<float>(N, q);
  sycl::float2 *xy = sycl::malloc_shared<sycl::float2>(N, q);
  for (size_t i = 0; i < N; ++i) {
    x[i] = float(i % 13) + 0.5f;
    y[i] = float(i % 7);
    xy[i] = sycl::float2(x[i], y[i]);
  }
  const In in{x, y, xy};
  vec_arith(q, in);
  vec_shift_reduce(q, in);
  vec_div_store(q, in);
  vec_div_loop(q, in);
  vec_div_reduce(q, in);
  vec_local_barrier(q, in);
  vec_2d(q, in);
  vec_float2_local(q, in);
  vec_float2_loop(q, in);
  vec_private_array(q, in);
  vec_private_accum(q, in);
  vec_reverse(q, in);
  vec_pinned(q, in);
  vec_pinned_update(q, in);
  vec_periodic(q, in);
  vec_partial(q, in);
  vec_partial_barrier(q, in);
  vec_partial_2d(q, in);
  vec_wg_reduce(q, in);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(xy, q);
}
