// Fused decoder MLP prologue (`mix_router`): keeps the exact arithmetic and reduction order of
// mix.cu and router.cu. Each owning CTA writes its normalized BF16 token to L2 (a.out_bf16),
// synchronizes across the cluster, triggers PDL for moe_up right before routing, and runs
// BF16 Tensor Core router MMA with DSMEM logit gather.
#include <array>
#include <atomic>
#include <cooperative_groups.h>
#include <mutex>

#include "kernels/common/l2.cuh"
#include "kernels/common/nvfp4.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dsmem.cuh"
#include "kernels/dense/reduce.cuh"

namespace ingine::kernels::dense {
namespace {

constexpr int MIX_T = 192, REF_W = H / 4 / 32, STAGED_QUADS = 2;
constexpr int MAX_M = 8, EXPERTS = 128, TOPK = 8;
constexpr l2::Evict WEIGHT_L2 = l2::Evict::none;

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
// Sum of squares of one channel quad for the output norms, with the FMA pattern of the reference build
// pinned: left to the compiler, the restructured kernel got another one here (same sum, other rounding of
// the first two products), which moved about one output element in 10^6 by a BF16 ulp.
__device__ __forceinline__ float sumsq4(float4 a) {
  return __fmaf_rn(a.w, a.w, __fmaf_rn(a.z, a.z, __fmaf_rn(a.x, a.x, __fmul_rn(a.y, a.y))));
}
__device__ __forceinline__ float4 scale4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
  __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

// Block sum of N per-quad values in two steps. sum_part: the warp stage for this thread's quad `u`, i.e.
// reference warp w + u * THREADS / 32 (butterflies, or the transposing reduce_scatter for N >= 6: the same
// additions), stored in `scratch` ([REF_W * block_sum_slots<N>()]). sum_total: one barrier, then lane i adds
// the 18 warp sums of value i in warp order and the totals are broadcast by shuffles (no second barrier;
// `scratch` belongs to one reduction).
template <int THREADS, int N>
__device__ __forceinline__ void sum_part(float (&v)[N], float* scratch, int u) {
  constexpr int S = block_sum_slots<N>();
  const int lane = threadIdx.x & 31, rw = (threadIdx.x >> 5) + u * (THREADS / 32);
  if constexpr (S == N) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = group_sum<32>(v[i]);
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < N; ++i) scratch[rw * N + i] = v[i];
    }
  } else {
    scratch[rw * 32 + lane] = reduce_scatter<32, N>(v);  // lane l: value l % P
  }
}
template <int N>
__device__ __forceinline__ float sum_lane(const float* scratch) {
  constexpr int S = block_sum_slots<N>();
  asm volatile("barrier.cta.sync 1, 192;" ::: "memory");  // Only the first six warps run mix (non-aligned partial CTA barrier).
  const int lane = threadIdx.x & 31;
  float s = 0.f;
  if (lane < N) {
#pragma unroll
    for (int w = 0; w < REF_W; ++w) s += scratch[w * S + lane];
  }
  return s;
}
template <int N>
__device__ __forceinline__ void sum_total(float (&out)[N], const float* scratch) {
  const float s = sum_lane<N>(scratch);
#pragma unroll
  for (int i = 0; i < N; ++i) out[i] = __shfl_sync(0xffffffffu, s, i);
}

struct MixWeights {
  static constexpr int QUADS = H / 4 / MIX_T;
  float4 q[QUADS], r4[QUADS], dt4[QUADS], g4[QUADS];
  float aw, ac, alo, atau;
};

struct MixScratch {
  const float* blocks[5];
  float ablation[REF_W * 2], states[REF_W * 32], norm[REF_W];
};

// Nine router warps leave fewer registers per thread. Stage two old-state quads and
// prefetch the output norm in shared memory for the eight-CTA mix_router specialization.
struct alignas(16) MixRouterStaging {
  float4 states[STAGED_QUADS * 5 * MIX_T];
  float4 norm[H / 4];
};

template <bool STAGE_MIX = false>
__device__ __forceinline__ void load_mix_weights(const MixParams& a, MixWeights& w, MixScratch& scratch,
                                                 MixRouterStaging* staged = nullptr) {
  const int t = threadIdx.x;
  auto ld4 = [](const float* p) { return *reinterpret_cast<const float4*>(p); };
  auto ch = [t](int u) { return 4 * (t + u * MIX_T); };
  auto& [q, r4, dt4, g4, aw, ac, alo, atau] = w;
  if (t < a.nblocks) scratch.blocks[t] = a.blocks[t];
  aw = 0.f, ac = 0.f, alo = 0.f, atau = 0.f;
#pragma unroll
  for (int u = 0; u < w.QUADS; ++u) {
    if (a.nblocks) q[u] = ld4(a.queries + ch(u));
    r4[u] = a.abl_r ? ld4(a.abl_r + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    dt4[u] = a.abl_r && a.abl_d ? ld4(a.abl_d + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (STAGE_MIX) cp_async16(&staged->norm[t + u * MIX_T], a.norm_w + ch(u));
    else g4[u] = ld4(a.norm_w + ch(u));
  }
  if constexpr (STAGE_MIX) asm volatile("cp.async.commit_group;" ::: "memory");
  if (a.abl_r) {
    const float4 ap = __ldg(reinterpret_cast<const float4*>(a.abl_w));
    aw = ap.x, ac = ap.y, alo = ap.z, atau = ap.w;
  }
}

// Keep the residual and history-copy registers out of the subsequent state reduction.
__device__ __forceinline__ void mix_residual(const MixParams& a, const MixWeights& w, int m, MixScratch& scratch,
                                            float4 (&nw)[MixWeights::QUADS]) {
  constexpr int QUADS = MixWeights::QUADS;
  const int t = threadIdx.x;
  const size_t row = size_t(m) * H;
  auto ld4 = [](const float* p) { return *reinterpret_cast<const float4*>(p); };
  auto ch = [t](int u) { return 4 * (t + u * MIX_T); };
  // Issue the side-copy loads ahead of the residual inputs, then retire their registers before the
  // state reduction. Longer history rows finish with a strided loop.
  constexpr int CPT = 2;
  uint4 cv[2][CPT];
  if (a.copies[0].dst || a.copies[1].dst) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const MixRowCopy& cp = a.copies[c];
      if (!cp.dst) continue;
      const float* src = cp.src + m * cp.src_stride;
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        if (t + j * MIX_T < cp.cols / 4) cv[c][j] = ld_act(src + 4 * (t + j * MIX_T));
    }
  }
  float4 y[QUADS];
#pragma unroll
  for (int u = 0; u < QUADS; ++u) {
    nw[u] = a.base ? ld4(a.base + row + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    if (a.write) y[u] = ld4(a.write + row + ch(u));
  }
  if (a.copies[0].dst || a.copies[1].dst) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const MixRowCopy& cp = a.copies[c];
      if (!cp.dst) continue;
      const float4* src = reinterpret_cast<const float4*>(cp.src + m * cp.src_stride);
      float4* dst = reinterpret_cast<float4*>(cp.dst + m * cp.dst_stride);
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        if (t + j * MIX_T < cp.cols / 4) *reinterpret_cast<uint4*>(dst + t + j * MIX_T) = cv[c][j];
      for (int k = t + CPT * MIX_T; k < cp.cols / 4; k += MIX_T) dst[k] = src[k];
    }
  }

  // new = base + (write - (w * max(write . r - offset, floor)) * r); offset 0 with floor -FLT_MAX is the plain
  // projection, bit for bit (x - 0 and max(x, -FLT_MAX) return x). With a detector the write is only changed
  // where write . detector exceeds its threshold (the same block reduction carries both dots).
  if (a.write) {
    if (a.abl_r) {
      float c;
      if (a.abl_d) {
#pragma unroll
        for (int u = 0; u < QUADS; ++u) {
          float d[2] = {dot4(y[u], w.r4[u]), dot4(y[u], w.dt4[u])};
          sum_part<MIX_T>(d, scratch.ablation, u);
        }
        float d[2];
        sum_total(d, scratch.ablation);
        c = d[1] > w.atau ? w.aw * fmaxf(d[0] - w.ac, w.alo) : 0.f;
      } else {
#pragma unroll
        for (int u = 0; u < QUADS; ++u) {
          float d[1] = {dot4(y[u], w.r4[u])};
          sum_part<MIX_T>(d, scratch.ablation, u);
        }
        float d[1];
        sum_total(d, scratch.ablation);
        c = w.aw * fmaxf(d[0] - w.ac, w.alo);
      }
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        const float4 r = w.r4[u];
        y[u] = make_float4(y[u].x - c * r.x, y[u].y - c * r.y, y[u].z - c * r.z, y[u].w - c * r.w);
      }
    }
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
      if (a.base) nw[u] = make_float4(nw[u].x + y[u].x, nw[u].y + y[u].y, nw[u].z + y[u].z, nw[u].w + y[u].w);
      else nw[u] = y[u];
    }
  }
}

// Same per-quad arithmetic and 18-warp summation order as mix_row in mix.cu.
template <int NS, bool STAGE_MIX>
__device__ __forceinline__ void mix_row(const MixParams& a, const MixWeights& w, int m, MixScratch& scratch,
                                       const float4 (&nw)[MixWeights::QUADS],
                                       MixRouterStaging* staged) {
  constexpr int NV = 2 * NS, QUADS = MixWeights::QUADS;
  const size_t row = size_t(m) * H;
  const int t = threadIdx.x;
  auto ch = [t](int u) { return 4 * (t + u * MIX_T); };
  float4 mixed[QUADS];
  if constexpr (NS == 1) {
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
      mixed[u] = nw[u];
      if (a.partial_out) *reinterpret_cast<float4*>(a.partial_out + row + ch(u)) = nw[u];
    }
  } else {
    float4 s[NS][QUADS];
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
#pragma unroll
      for (int i = 0; i < NS - 1; ++i) s[i][u] = *reinterpret_cast<const float4*>(scratch.blocks[i] + row + ch(u));
      s[NS - 1][u] = nw[u];
      if (a.partial_out) *reinterpret_cast<float4*>(a.partial_out + row + ch(u)) = nw[u];
      float p[NV];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        p[i] = dot4(s[i][u], s[i][u]);
        p[NS + i] = dot4(w.q[u], s[i][u]);
      }
      sum_part<MIX_T>(p, scratch.states, u);
      if constexpr (STAGE_MIX) {
        if (u < STAGED_QUADS) {
#pragma unroll
          for (int i = 0; i < NS - 1; ++i) staged->states[(u * 5 + i) * MIX_T + t] = s[i][u];
        }
      }
    }
    // Assign one state to each lane for softmax, using the same addition order as the
    // per-thread softmax in mix.cu. This frees the logits and reciprocals of the other states.
    constexpr unsigned warp = 0xffffffffu;
    const int lane = t & 31, state = lane < NS ? lane : 0;
    const float sums = sum_lane<NV>(scratch.states);
    const float rinv = rsqrtf(__shfl_sync(warp, sums, state) / H + EPS);
    const float logit = __shfl_sync(warp, sums, NS + state) * rinv * (1.f / 48.f);
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < NS; ++i) mx = fmaxf(mx, __shfl_sync(warp, logit, i));
    const float e = expf(logit - mx);
    float den = 0.f;
#pragma unroll
    for (int i = 0; i < NS; ++i) den += __shfl_sync(warp, e, i);
    const float prob = e / den;
#pragma unroll
    for (int u = 0; u < QUADS; ++u) mixed[u] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int i = 0; i < NS; ++i) {
      const float p = __shfl_sync(warp, prob, i);
      const float r = __shfl_sync(warp, rinv, i);
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        float4 source;
        if constexpr (STAGE_MIX) {
          source = u < STAGED_QUADS && i < NS - 1 ? staged->states[(u * 5 + i) * MIX_T + t] : s[i][u];
        } else {
          source = s[i][u];
        }
        const float4 o = mixed[u], v = scale4(source, r);
        mixed[u] = make_float4(o.x + p * v.x, o.y + p * v.y, o.z + p * v.z, o.w + p * v.w);
      }
    }
  }

#pragma unroll
  for (int u = 0; u < QUADS; ++u) {
    float ss[1] = {sumsq4(mixed[u])};
    sum_part<MIX_T>(ss, scratch.norm, u);
  }
  float ss[1];
  sum_total(ss, scratch.norm);
  const float rinv = rsqrtf(ss[0] / H + EPS);
  if constexpr (STAGE_MIX) asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
  for (int u = 0; u < QUADS; ++u) {
    float4 g;
    if constexpr (STAGE_MIX) g = staged->norm[t + u * MIX_T];
    else g = w.g4[u];
    const float4 n = scale4(mixed[u], rinv);
    const float4 o = make_float4(n.x * g.x, n.y * g.y, n.z * g.z, n.w * g.w);
    store_bf16x4(a.out_bf16 + row + ch(u), o);
  }
}

template <bool STAGE_MIX = false>
__device__ __forceinline__ void mix_one_token(const MixParams& a, const MixWeights& w, int m,
                                              MixScratch& scratch, MixRouterStaging* staged = nullptr) {
  float4 nw[MixWeights::QUADS];
  mix_residual(a, w, m, scratch, nw);
  // The first warp prefetched the pointer table before PDL. Publish it to the other mix warps.
  if (a.nblocks) asm volatile("barrier.cta.sync 1, 192;" ::: "memory");
  switch (a.nblocks) {
    case 0: mix_row<1, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
    case 1: mix_row<2, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
    case 2: mix_row<3, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
    case 3: mix_row<4, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
    case 4: mix_row<5, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
    case 5: mix_row<6, STAGE_MIX>(a, w, m, scratch, nw, staged); break;
  }
}

struct SelectionScratch {
  unsigned long long keys[EXPERTS];
  int cnt[2 * EXPERTS];
  float sel_s[TOPK];
  int sel_e[TOPK];
};

template <int NT, int KPARTS>
__device__ __forceinline__ void select_top8(int m, const float (&logits)[KPARTS][EXPERTS], float b, int* idx, float* wts,
                                            SelectionScratch& scratch) {
  // selection: unique 39-bit keys (order-preserving bits of v, then 127 - e: ties go to the lower expert),
  // rank = number of greater keys. Count the greater keys in NT / 128 partitions of the key range.
  const int t = threadIdx.x;
  constexpr int PARTS = NT / EXPERTS, W = EXPERTS / PARTS;
  auto& [keys, cnt, sel_s, sel_e] = scratch;
  float s = 0.f;
  if (t < EXPERTS) {
    float logit = logits[0][t];
    if constexpr (KPARTS == 2) logit += logits[1][t];
    s = 1.f / (1.f + expf(-logit));
    const uint32_t u = __float_as_uint(s + b + 0.f);  // + 0: -0 -> +0, which compares equal
    const uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    keys[t] = (static_cast<unsigned long long>(o) << 7) | static_cast<unsigned>(EXPERTS - 1 - t);
  }
  __syncthreads();
  if (t < PARTS * EXPERTS) {
    const int e = t % EXPERTS, part = t / EXPERTS;
    const unsigned long long mine = keys[e];
    int c[4] = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < W; j += 4)
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] += keys[part * W + j + k] > mine;
    cnt[t] = c[0] + c[1] + c[2] + c[3];
  }
  __syncthreads();
  if (t < EXPERTS) {
    int r = 0;
#pragma unroll
    for (int p = 0; p < PARTS; ++p) r += cnt[p * EXPERTS + t];
    if (r < TOPK) {
      sel_e[r] = t;
      sel_s[r] = s;
    }
  }
  __syncthreads();
  if (t < TOPK) {
    const float my_s = sel_s[t];
    float sum = my_s;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
    idx[m * TOPK + t] = sel_e[t];
    wts[m * TOPK + t] = my_s / (sum + 1e-20f) * 8.f;
  }
}

template <int KW>
struct MmaWeights {
  static constexpr int CHUNKS = KW / 32;
  static constexpr bool TAIL = KW % 32 != 0;
  static_assert(KW % 16 == 0);
  uint4 a[CHUNKS], b[CHUNKS];
  uint2 at, bt;
};

template <int KW>
__device__ __forceinline__ void load_mma_weights(MmaWeights<KW>& w, const __nv_bfloat16* src, int stride) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = lane >> 2, pair = lane & 3;
  const auto* wa = src + row * stride + warp * KW;
  const auto* wb = wa + 8 * stride;
  const auto policy = l2::policy<WEIGHT_L2>();
#pragma unroll
  for (int i = 0; i < w.CHUNKS; ++i) {
    const int k = i * 32 + pair * 8;
    w.a[i] = l2::ld_stream<WEIGHT_L2>(wa + k, policy);
    w.b[i] = l2::ld_stream<WEIGHT_L2>(wb + k, policy);
  }
  if constexpr (w.TAIL) {
    const int k = w.CHUNKS * 32 + pair * 4;
    w.at = __ldg(reinterpret_cast<const uint2*>(wa + k));
    w.bt = __ldg(reinterpret_cast<const uint2*>(wb + k));
  }
}

template <int KW>
__device__ __forceinline__ void project_mma(float (&acc)[4], const MmaWeights<KW>& w, const __nv_bfloat16* x) {
  const int pair = threadIdx.x & 3, warp = threadIdx.x >> 5;
  const auto* xb = x + warp * KW;
#pragma unroll
  for (int i = 0; i < w.CHUNKS; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(xb + i * 32 + pair * 8);
    nvfp4::mma_bf16(acc, w.a[i].x, w.b[i].x, w.a[i].y, w.b[i].y, v.x, v.y);
    nvfp4::mma_bf16(acc, w.a[i].z, w.b[i].z, w.a[i].w, w.b[i].w, v.z, v.w);
  }
  if constexpr (w.TAIL) {
    const uint2 v = *reinterpret_cast<const uint2*>(xb + w.CHUNKS * 32 + pair * 4);
    nvfp4::mma_bf16(acc, w.at.x, w.bt.x, w.at.y, w.bt.y, v.x, v.y);
  }
}

template <int NW, int M>
__device__ __forceinline__ float4 gather_mma(const float (&red)[NW][M][16], int m, int r) {
  float4 v = *reinterpret_cast<const float4*>(&red[0][m][r]);
#pragma unroll
  for (int i = 1; i < NW; ++i) {
    const float4 u = *reinterpret_cast<const float4*>(&red[i][m][r]);
    v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
  }
  return v;
}

struct RouterArgs {
  const __nv_bfloat16* w;
  const float* bias;
  int* idx;
  float* wts;
};

template <int CL>
struct RouterCfg {
  static constexpr int NW = CL == 8 ? 9 : 8, NT = 32 * NW, KPARTS = CL / 8;
  static constexpr int KCTA = H / KPARTS, KW = KCTA / NW;
  using Weights = MmaWeights<KW>;
  using X = __nv_bfloat16[MAX_M][KCTA];
  struct alignas(16) Scratch {
    float red[NW][MAX_M][16];
    float logits[KPARTS][EXPERTS];
    SelectionScratch selection;
  };
};

template <int CL>
__device__ __forceinline__ void load_router(const RouterArgs& a, int rank, typename RouterCfg<CL>::Weights& w) {
  using R = RouterCfg<CL>;
  const int tile = rank / R::KPARTS, part = rank % R::KPARTS;
  if (threadIdx.x < R::NT) load_mma_weights(w, a.w + tile * 16 * H + part * R::KCTA, H);
}

// The same K slices, MMA fragments and warp summation order as router_mma_kernel.
template <int CL>
__device__ __forceinline__ void route(const RouterArgs& a, int rank, int M,
                                      const typename RouterCfg<CL>::Weights& w, float bias_e,
                                      const typename RouterCfg<CL>::X& xs, typename RouterCfg<CL>::Scratch& s,
                                      uint32_t mb_l) {
  using R = RouterCfg<CL>;
  const int t = threadIdx.x, warp = t >> 5, row = (t & 31) >> 2, pair = t & 3;
  const int tile = rank / R::KPARTS, part = rank % R::KPARTS;
  if (warp < R::NW) {
    float acc[4] = {};
    project_mma(acc, w, xs[row < M ? row : 0]);
    s.red[warp][2 * pair][row] = acc[0];
    s.red[warp][2 * pair + 1][row] = acc[1];
    s.red[warp][2 * pair][row + 8] = acc[2];
    s.red[warp][2 * pair + 1][row + 8] = acc[3];
  }
  __syncthreads();
  if (t < M * 4) {
    const int m = t / 4, r = (t % 4) * 4;
    st_async4(remote(smem_u32(&s.logits[part][tile * 16 + r]), m),
              gather_mma(s.red, m, r), remote(mb_l, m));
  }
  if (rank < M) {
    mb_wait(mb_l);
    select_top8<R::NT, R::KPARTS>(rank, s.logits, bias_e, a.idx, a.wts, s.selection);
  }
}

template <int CL>
__global__ void __launch_bounds__(RouterCfg<CL>::NT) mix_router_kernel(MixParams a, RouterArgs router, uint32_t zero) {
  using R = RouterCfg<CL>;
  extern __shared__ __align__(16) uint8_t smem[];
  auto& xs = *reinterpret_cast<typename R::X*>(smem);
  __shared__ typename R::Scratch route_s;
  __shared__ MixScratch mix_s;
  __shared__ uint64_t mbar[1];

  const int rank = cooperative_groups::this_cluster().block_rank(), t = threadIdx.x;
  const int part = rank % R::KPARTS;
  const uint32_t mb_l = smem_u32(&mbar[0]);
  if (t == 0) {
    mb_init(mb_l);
    fence_mbarrier_init();
    if (rank < a.M) mb_expect(mb_l, sizeof(route_s.logits));
  }

  typename R::Weights rw;
  load_router<CL>(router, rank, rw);
  const float bias_e = t < EXPERTS ? __ldg(router.bias + t) : 0.f;
  MixWeights mw;
  const bool mix_thread = rank < a.M && t < MIX_T;
  auto* staged = reinterpret_cast<MixRouterStaging*>(smem + sizeof(xs));
  if (mix_thread) load_mix_weights<CL == 8>(a, mw, mix_s, staged);
  pdl_wait_fenced(zero);

  if (mix_thread) mix_one_token<CL == 8>(a, mw, rank, mix_s, staged);
  if (rank < a.M) {
    __syncthreads();
    __threadfence();
  }
  cluster_arrive();
  cluster_wait();
  pdl_trigger();
  for (int i = t; i < a.M * (R::KCTA / 8); i += R::NT) {
    const int m = i / (R::KCTA / 8), k = i % (R::KCTA / 8) * 8;
    cp_async16(&xs[m][k], a.out_bf16 + size_t(m) * H + part * R::KCTA + k);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  route<CL>(router, rank, a.M, rw, bias_e, xs, route_s, mb_l);
}

// Cache the per-device 16-CTA occupancy check once across host threads; launch_ex still sets
// per-kernel launch attributes (including non-portable cluster / dynamic SMEM opt-ins).
int prologue_cluster_size() {
  static constexpr int kMaxCachedDevices = 64;
  static std::array<std::atomic<int>, kMaxCachedDevices> cache{};
  static std::mutex mu;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device >= 0 && device < kMaxCachedDevices) {
    if (const int cached = cache[device].load(std::memory_order_acquire)) return cached;
  }
  std::lock_guard<std::mutex> lock(mu);
  if (device >= 0 && device < kMaxCachedDevices) {
    if (const int cached = cache[device].load(std::memory_order_relaxed)) return cached;
  }
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int active = 0;
  if (sms >= 96) {
    auto kernel = mix_router_kernel<16>;
    constexpr int SMEM = sizeof(RouterCfg<16>::X);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {16, 1, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(RouterCfg<16>::NT);
    cfg.dynamicSmemBytes = SMEM;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess ||
        cudaOccupancyMaxActiveClusters(&active, kernel, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      active = 0;
    }
  }
  const int cl = active > 0 ? 16 : 8;
  if (device >= 0 && device < kMaxCachedDevices) cache[device].store(cl, std::memory_order_release);
  return cl;
}

void validate_mix(const MixParams& mp) {
  require(mp.M >= 2 && mp.M <= MAX_M && mp.Q == 1, "mix_router: needs 2..8 tokens and one query");
  require(mp.nblocks >= 0 && mp.nblocks <= 5, "mix_router: unsupported block count");
  require((mp.base || mp.write) && (!mp.nblocks || (mp.blocks && mp.queries)), "mix_router: missing mix inputs");
  require(!mp.abl_r || mp.abl_w, "mix_router: ablation weights required");
  require(!mp.out_f32 && !mp.out_copy && !mp.new_bf16, "mix_router: unsupported mix output");
  for (const auto& copy : mp.copies)
    require(!copy.dst || (copy.src && copy.cols % 4 == 0 && copy.src_stride % 4 == 0 && copy.dst_stride % 4 == 0),
            "mix_router: row copies need 16-byte aligned rows");
}

}  // namespace

bool mlp_prologue_default_enabled() { return prologue_cluster_size() == 16; }

void mix_router(const MixParams& mp, const __nv_bfloat16* router_w, const float* router_bias, int* idx, float* wts,
                bool pdl, cudaStream_t stream) {
  validate_mix(mp);
  require(mp.out_bf16 && mp.norm_w && router_w && router_bias && idx && wts, "mix_router: null argument");
  const RouterArgs router{router_w, router_bias, idx, wts};
  if (prologue_cluster_size() == 16)
    launch_ex(mix_router_kernel<16>, dim3(16), RouterCfg<16>::NT, sizeof(RouterCfg<16>::X), 16, pdl, stream, mp,
              router, 0u);
  else
    launch_ex(mix_router_kernel<8>, dim3(8), RouterCfg<8>::NT, sizeof(RouterCfg<8>::X) + sizeof(MixRouterStaging),
              8, pdl, stream, mp, router, 0u);
}

}  // namespace ingine::kernels::dense
