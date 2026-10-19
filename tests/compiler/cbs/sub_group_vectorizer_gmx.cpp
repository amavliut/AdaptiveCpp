// REQUIRES: rv
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=1 ACPP_APPDB_DIR=%t.db ACPP_DEBUG_LEVEL=3 %t 2>&1 | FileCheck %s --check-prefixes=RV,OK
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=8 ACPP_OPENMP_VECTORIZE=0 ACPP_APPDB_DIR=%t.db %t | FileCheck %s --check-prefix=OK

// SubgroupVectorizerPass on two shapes of GROMACS's nbnxm force kernel (nbnxm_sycl_kernel_body.h,
// 4 x 4 work-groups, S = 8: a sub-group spans two rows). RV and OK as in
// sub_group_vectorizer.cpp.

#include <cmath>
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

constexpr int G = 64, JM = 4, NCI = 8;
using AtomicF = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>;

void report(const char *Name, int Bad) {
  std::printf("%s: %s\n", Name, Bad ? "FAIL" : "ok");
  std::fflush(stdout);
}

std::vector<float> inputs() {
  std::vector<float> X(G * 16);
  for (size_t i = 0; i < X.size(); ++i)
    X[i] = float(i % 17) * 0.25f + 1.f;
  return X;
}

// reduceForceJShuffle in the divergent j-loop (`if (!maskSet) continue;`, one mask per row):
// shuffles reached by one row of the sub-group only; lanes tidxi < 3 read their own row
// RV: [SubgroupVectorizer] {{.*}}gmx_shuffleRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: gmx_shuffle: ok
void gmx_shuffle(sycl::queue &q) {
  const auto X = inputs();
  std::vector<float> Out(G * JM * 3, 0.f);
  std::vector<unsigned> Mask(G * 4);
  for (int g = 0; g < G; ++g)
    for (int r = 0; r < 4; ++r)
      Mask[g * 4 + r] = (g * 7 + r * 5 + 3) & 0xF;
  {
    sycl::buffer<float> BX{X}, BO{Out};
    sycl::buffer<unsigned> BM{Mask};
    q.submit([&](sycl::handler &h) {
      sycl::accessor x{BX, h, sycl::read_only};
      sycl::accessor o{BO, h, sycl::read_write};
      sycl::accessor m{BM, h, sycl::read_only};
      h.parallel_for(sycl::nd_range<2>{{4 * G, 4}, {4, 4}}, [=](sycl::nd_item<2> it) {
        const int tidxi = it.get_local_id(1), tidxj = it.get_local_id(0), g = it.get_group(0);
        const unsigned MyMask = m[g * 4 + tidxj];
        auto sg = it.get_sub_group();
        float *a_f = o.template get_multi_ptr<sycl::access::decorated::no>().get();
        for (int jm = 0; jm < JM; ++jm) {
          if (!((MyMask >> jm) & 1u))
            continue;
          const float v = x[g * 16 + tidxj * 4 + tidxi] * float(jm + 1);
          sycl::float3 f(v, 2.f * v, 3.f * v);
          f[0] += sycl::shift_group_left(sg, f[0], 1);
          f[1] += sycl::shift_group_right(sg, f[1], 1);
          f[2] += sycl::shift_group_left(sg, f[2], 1);
          if (tidxi & 1)
            f[0] = f[1];
          f[0] += sycl::shift_group_left(sg, f[0], 2);
          f[2] += sycl::shift_group_right(sg, f[2], 2);
          if (tidxi & 2)
            f[0] = f[2];
          if (tidxi < 3)
            AtomicF(a_f[((g * JM + jm) % (G * JM)) * 3 + tidxi]).fetch_add(f[0]);
        }
      });
    });
  }
  int Bad = 0; // each active row r adds (c + 1)(jm + 1) * sum of its x to component c
  for (int g = 0; g < G; ++g)
    for (int jm = 0; jm < JM; ++jm)
      for (int c = 0; c < 3; ++c) {
        float Ref = 0;
        for (int r = 0; r < 4; ++r)
          if ((Mask[g * 4 + r] >> jm) & 1u)
            for (int i = 0; i < 4; ++i)
              Ref += float(c + 1) * float(jm + 1) * X[g * 16 + r * 4 + i];
        const float Got = Out[(g * JM + jm) * 3 + c];
        if (std::abs(Got - Ref) > 1e-3f * (1 + std::abs(Ref)) && Bad++ < 5)
          std::printf("gmx_shuffle: g=%d jm=%d c=%d: %g, expected %g\n", g, jm, c, Got, Ref);
      }
  report("gmx_shuffle", Bad);
}

// the i-force path: private float[8] accumulators zeroed before a work-group barrier, filled
// under per-row masks in the divergent j-loop, then reduceForceIAndFShiftShuffles<1>
// RV: [SubgroupVectorizer] {{.*}}gmx_iforceRN{{.*}}: vectorized at width 8, 2 sub-groups per work-group
// OK: gmx_iforce: ok
void gmx_iforce(sycl::queue &q) {
  const auto X = inputs();
  std::vector<float> Out(G * NCI * 4 * 3, 0.f);
  std::vector<unsigned> Mask(G * 4), IMask(G * 4);
  for (int g = 0; g < G; ++g)
    for (int r = 0; r < 4; ++r) {
      Mask[g * 4 + r] = (g * 7 + r * 5 + 3) & 0xF;
      IMask[g * 4 + r] = (g * 13 + r * 7 + 5) & 0xFF;
    }
  {
    sycl::buffer<float> BX{X}, BO{Out};
    sycl::buffer<unsigned> BM{Mask}, BI{IMask};
    q.submit([&](sycl::handler &h) {
      sycl::accessor x{BX, h, sycl::read_only};
      sycl::accessor o{BO, h, sycl::read_write};
      sycl::accessor m{BM, h, sycl::read_only};
      sycl::accessor im{BI, h, sycl::read_only};
      sycl::local_accessor<float, 1> Sm{16, h};
      h.parallel_for(sycl::nd_range<2>{{4 * G, 4}, {4, 4}}, [=](sycl::nd_item<2> it) {
        const int tidxi = it.get_local_id(1), tidxj = it.get_local_id(0), tidx = tidxj * 4 + tidxi;
        const int g = it.get_group(0);
        auto sg = it.get_sub_group();
        float *a_f = o.template get_multi_ptr<sycl::access::decorated::no>().get();
        float FX[NCI] = {0.f}, FY[NCI] = {0.f}, FZ[NCI] = {0.f};
        Sm[tidx] = x[g * 16 + tidx];
        sycl::group_barrier(it.get_group());
        const unsigned MyMask = m[g * 4 + tidxj], IM = im[g * 4 + tidxj];
        for (int jm = 0; jm < JM; ++jm) {
          if (!((MyMask >> jm) & 1u))
            continue;
          for (int i = 0; i < NCI; ++i)
            if ((IM >> i) & 1u) {
              const float v = Sm[(tidx + i + jm) % 16] * float(i + 1);
              FX[i] += v;
              FY[i] += 2.f * v;
              FZ[i] += 3.f * v;
            }
        }
        for (int ci = 0; ci < NCI; ++ci) {
          const int aidx = (g * NCI + ci) * 4 + tidxi;
          float fx = FX[ci], fy = FY[ci], fz = FZ[ci];
          fx += sycl::shift_group_left(sg, fx, 4);
          fy += sycl::shift_group_right(sg, fy, 4);
          fz += sycl::shift_group_left(sg, fz, 4);
          if (tidxj & 1)
            fx = fy;
          AtomicF(a_f[aidx * 3 + (tidxj & 1)]).fetch_add(fx);
          if ((tidxj & 1) == 0)
            AtomicF(a_f[aidx * 3 + 2]).fetch_add(fz);
          sycl::group_barrier(sg);
        }
      });
    });
  }
  int Bad = 0; // component c of (g, ci, tidxi) sums all 4 rows
  for (int g = 0; g < G; ++g)
    for (int ci = 0; ci < NCI; ++ci)
      for (int ti = 0; ti < 4; ++ti)
        for (int c = 0; c < 3; ++c) {
          float Ref = 0;
          for (int r = 0; r < 4; ++r)
            for (int jm = 0; jm < JM; ++jm)
              if (((Mask[g * 4 + r] >> jm) & 1u) && ((IMask[g * 4 + r] >> ci) & 1u))
                Ref += float(c + 1) * X[g * 16 + (r * 4 + ti + ci + jm) % 16] * float(ci + 1);
          const float Got = Out[((g * NCI + ci) * 4 + ti) * 3 + c];
          if (std::abs(Got - Ref) > 1e-3f * (1 + std::abs(Ref)) && Bad++ < 5)
            std::printf("gmx_iforce: g=%d ci=%d ti=%d c=%d: %g, expected %g\n", g, ci, ti, c, Got, Ref);
        }
  report("gmx_iforce", Bad);
}

int main() {
  sycl::queue q;
  gmx_shuffle(q);
  gmx_iforce(q);
}
