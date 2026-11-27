// mix: ablated residual update + attention residual over block states + weighted RMS norm.
//
// Decode uses one CTA per token whose threads own a few channel
// quads each (one float4 of every state, query and output per quad: coalesced 16-byte accesses, states live
// in registers for the whole kernel). At most three block reductions, each a warp stage + one shared round
// with a single barrier: (1) r . write for the ablation, (2) for every state i its sum of squares and the
// raw dots q_j . s_i (q_j . n_i = rinv_i * (q_j . s_i), so normalisation, logits and softmax need a single
// reduction; its up to 30 values use the transposing reduce_scatter, 31 shuffles instead of 150), (3) the
// sum of squares of each mixed output for the final RMS norm. The state count (blocks + 1) and query
// count are template parameters, so no work is spent on absent states. Block pointers are read from
// device memory (constant tables, loaded before the PDL wait), so the graph can be replayed with other
// blocks; counts (blocks, queries, which optional tensors exist) are fixed per call site. Queries,
// ablation direction/weight, norm weights and the block pointers are loaded before griddepcontrol.wait
// (pdl=True; pdl_wait_fenced, as ptxas would sink them below a plain wait), every state after it. Side
// outputs (a BF16 copy of `new`, a strided copy of out_bf16, plain FP32 row copies) let the decode step drop
// separate conversion/memcpy nodes, which would break the PDL chain. `embed` (also here) reproduces torch's
// reduction order so that it is bit-exact with model.Model.embed.
// Ported from engine/csrc/dense/mix.cu (torch launchers -> raw pointers). Results are bit-identical to that
// kernel (checked byte for byte against it over all state/query counts, M = 1, 2, 3, 8): same expressions
// and the same order of every FP32 sum (see REF_W below); changed are the thread count, the barriers, what
// is loaded before the PDL wait, and the side outputs.
// Bulk prefill reuses the constants across four rows per CTA and distributes the state softmax across lanes.
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

namespace ingine::kernels::dense {
namespace {

// The arithmetic is that of the reference kernel (576 threads, thread = one channel quad): every block sum
// adds the per-quad terms of each of its 18 warps with a 32-lane butterfly, then the 18 warp sums in warp
// order. A thread owns quads t + u * THREADS (the same lane of reference warp w + u * THREADS / 32).
constexpr int REF_W = H / 4 / 32;
constexpr int MIX_T = 192;
constexpr int BULK_T = 288;
constexpr int MAXB = 5, MAXQ = 4;
constexpr int BULK_MIN_ROWS = 256;
constexpr int BULK_ROWS = 4;

struct MixArgs {
  const float* const* blocks;  // NS - 1 device pointers, each [M, H]
  const float* base;   // [M, H] or null
  const float* write;  // [M, H] or null
  const float* abl_r;  // [H] or null (no ablation)
  const float* abl_d;  // [H] detector or null (no detector: the ablation always acts)
  const float* abl_w;  // [4]: weight, offset, floor, detector threshold
  float* partial;      // [M, H] or null
  const float* queries;  // [Q, H]
  const float* norm_w;   // [Q, H] or null (direct BF16 conversion)
  __nv_bfloat16* out_bf16;  // [Q, M, H] or null
  float* out_f32;           // [Q, M, H] or null
  __nv_bfloat16* new_bf16;  // row m at + m * new_bf16_stride, or null
  long long new_bf16_stride;
  __nv_bfloat16* out_copy;  // Q == 1: row m at + m * out_copy_stride, or null
  long long out_copy_stride;
  MixRowCopy copies[2];     // dst == null: unused
  int M;
  uint32_t zero;  // 0 (opaque to the compiler: pdl_wait_fenced)
};

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
  __syncthreads();
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

template <int NS, int Q, int THREADS>
struct MixWeights {
  static_assert(THREADS % 32 == 0 && H / 4 % THREADS == 0);
  static constexpr int QUADS = H / 4 / THREADS;
  float4 q[Q][QUADS], r4[QUADS], dt4[QUADS], g4[Q][QUADS];
  const float* bp[NS > 1 ? NS - 1 : 1];
  float aw, ac, alo, atau;
};

// Each thread keeps its channel quads of the weights in registers across the CTA's rows.
template <bool RAW, int NS, int Q, int THREADS>
__device__ __forceinline__ void load_mix_weights(const MixArgs& a, MixWeights<NS, Q, THREADS>& w) {
  const int t = threadIdx.x;
  auto ld4 = [](const float* p) { return *reinterpret_cast<const float4*>(p); };
  auto ch = [t](int u) { return 4 * (t + u * THREADS); };
  auto& [q, r4, dt4, g4, bp, aw, ac, alo, atau] = w;
#pragma unroll
  for (int i = 0; i < NS - 1; ++i) bp[i] = a.blocks[i];
  aw = 0.f, ac = 0.f, alo = 0.f, atau = 0.f;
#pragma unroll
  for (int u = 0; u < w.QUADS; ++u) {
    if constexpr (NS != 1 || Q != 1) {
#pragma unroll
      for (int j = 0; j < Q; ++j) q[j][u] = ld4(a.queries + size_t(j) * H + ch(u));
    }
    r4[u] = a.abl_r ? ld4(a.abl_r + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    dt4[u] = a.abl_d ? ld4(a.abl_d + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    if (!RAW && a.out_bf16 && a.norm_w) {
#pragma unroll
      for (int j = 0; j < Q; ++j) g4[j][u] = ld4(a.norm_w + size_t(j) * H + ch(u));
    }
  }
  if (a.abl_r) {
    const float4 ap = __ldg(reinterpret_cast<const float4*>(a.abl_w));
    aw = ap.x, ac = ap.y, alo = ap.z, atau = ap.w;
  }
}

// Each lane handles one state's softmax term. Visit the terms in the original state order for the max
// and denominator. The returned probability is used only from the first NS lanes.
template <int NS>
__device__ __forceinline__ float mix_probability(float sum, float rinv, int state, int j) {
  constexpr unsigned warp = 0xffffffffu;
  const float logit = __shfl_sync(warp, sum, NS + j * NS + state) * rinv * (1.f / 48.f);
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < NS; ++i) mx = fmaxf(mx, __shfl_sync(warp, logit, i));
  const float e = expf(logit - mx);
  float den = 0.f;
#pragma unroll
  for (int i = 0; i < NS; ++i) den += __shfl_sync(warp, e, i);
  return e / den;
}

// NS = states in the mix (blocks + new), Q = queries; NS == 1 && Q == 1 skips mixing.
// red1/red2/red3: [REF_W * block_sum_slots<N>()] for N = 2, NS + Q NS, Q.
template <bool COPIES, bool BULK, bool RAW, int NS, int Q, int THREADS>
__device__ __forceinline__ void mix_row(const MixArgs& a, const MixWeights<NS, Q, THREADS>& w, int m, float* red1,
                                       float* red2, float* red3) {
  constexpr bool SKIP = NS == 1 && Q == 1;
  constexpr int NV = NS + Q * NS;  // sums of squares + query dots
  constexpr int QUADS = MixWeights<NS, Q, THREADS>::QUADS;
  const auto& [q, r4, dt4, g4, bp, aw, ac, alo, atau] = w;
  const int t = threadIdx.x;
  const size_t row = size_t(m) * H;
  auto ld4 = [](const float* p) { return *reinterpret_cast<const float4*>(p); };
  auto ch = [t](int u) { return 4 * (t + u * THREADS); };
  float4 s[NS][QUADS];
  // side copies: their loads are issued first and fly during the mix; the stores come after the states'
  // loads below. (A store waits for its value: stores ahead of the state loads put a round trip in front of
  // them, measured +0.7 us per mix with two copies.) CPT 16-byte pieces per thread and copy are held in
  // registers (rows up to CPT * THREADS * 4 floats); longer rows finish with a plain loop.
  constexpr int CPT = 2;
  uint4 cv[2][CPT];
  if constexpr (COPIES) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const MixRowCopy& cp = a.copies[c];
      if (!cp.dst) continue;
      const float* src = cp.src + m * cp.src_stride;
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        if (t + j * THREADS < cp.cols / 4) cv[c][j] = ld_act(src + 4 * (t + j * THREADS));
    }
  }
  float4 nw[QUADS], y[QUADS];
#pragma unroll
  for (int u = 0; u < QUADS; ++u) {
#pragma unroll
    for (int i = 0; i < NS - 1; ++i) s[i][u] = ld4(bp[i] + row + ch(u));
    nw[u] = a.base ? ld4(a.base + row + ch(u)) : make_float4(0.f, 0.f, 0.f, 0.f);
    if (a.write) y[u] = ld4(a.write + row + ch(u));
  }
  if constexpr (COPIES) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const MixRowCopy& cp = a.copies[c];
      if (!cp.dst) continue;
      const float4* src = reinterpret_cast<const float4*>(cp.src + m * cp.src_stride);
      float4* dst = reinterpret_cast<float4*>(cp.dst + m * cp.dst_stride);
#pragma unroll
      for (int j = 0; j < CPT; ++j)
        if (t + j * THREADS < cp.cols / 4) *reinterpret_cast<uint4*>(dst + t + j * THREADS) = cv[c][j];
      for (int k = t + CPT * THREADS; k < cp.cols / 4; k += THREADS) dst[k] = src[k];
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
          float d[2] = {dot4(y[u], r4[u]), dot4(y[u], dt4[u])};
          sum_part<THREADS>(d, red1, u);
        }
        float d[2];
        sum_total(d, red1);
        c = d[1] > atau ? aw * fmaxf(d[0] - ac, alo) : 0.f;
      } else {
#pragma unroll
        for (int u = 0; u < QUADS; ++u) {
          float d[1] = {dot4(y[u], r4[u])};
          sum_part<THREADS>(d, red1, u);
        }
        float d[1];
        sum_total(d, red1);
        c = aw * fmaxf(d[0] - ac, alo);
      }
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        const float4 r = r4[u];
        y[u] = make_float4(y[u].x - c * r.x, y[u].y - c * r.y, y[u].z - c * r.z, y[u].w - c * r.w);
      }
    }
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
      if (a.base) nw[u] = make_float4(nw[u].x + y[u].x, nw[u].y + y[u].y, nw[u].z + y[u].z, nw[u].w + y[u].w);
      else nw[u] = y[u];
    }
  }
#pragma unroll
  for (int u = 0; u < QUADS; ++u) {
    if (a.partial) *reinterpret_cast<float4*>(a.partial + row + ch(u)) = nw[u];
    if (a.new_bf16) store_bf16x4(a.new_bf16 + m * a.new_bf16_stride + ch(u), nw[u]);
  }

  if constexpr (SKIP) {
    float rinv = 0.f;
    if (!BULK || (!RAW && a.norm_w)) {
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        float ss[1] = {sumsq4(nw[u])};
        sum_part<THREADS>(ss, red3, u);
      }
      float ss[1];
      sum_total(ss, red3);
      rinv = rsqrtf(ss[0] / H + EPS);
    }
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
      if (a.out_f32) *reinterpret_cast<float4*>(a.out_f32 + row + ch(u)) = nw[u];
      if (a.out_bf16) {
        if (!RAW && a.norm_w) {
          const float4 g = g4[0][u];
          const float4 n = scale4(nw[u], rinv);
          const float4 o = make_float4(n.x * g.x, n.y * g.y, n.z * g.z, n.w * g.w);
          store_bf16x4(a.out_bf16 + row + ch(u), o);
          if (a.out_copy) store_bf16x4(a.out_copy + m * a.out_copy_stride + ch(u), o);
        } else {
          store_bf16x4(a.out_bf16 + row + ch(u), nw[u]);
        }
      }
    }
    return;
  } else {
#pragma unroll
    for (int u = 0; u < QUADS; ++u) {
      s[NS - 1][u] = nw[u];
      float p[NV];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        p[i] = dot4(s[i][u], s[i][u]);
#pragma unroll
        for (int j = 0; j < Q; ++j) p[NS + j * NS + i] = dot4(q[j][u], s[i][u]);
      }
      sum_part<THREADS>(p, red2, u);
    }
    constexpr unsigned warp = 0xffffffffu;
    const int lane = t & 31;
    const int state = lane < NS ? lane : 0;
    float v[NV], sum, rinv_lane;
    if constexpr (BULK) {
      sum = sum_lane<NV>(red2);
      rinv_lane = rsqrtf(__shfl_sync(warp, sum, state) / H + EPS);
    } else {
      sum_total(v, red2);
    }
    float rinv[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
      const float r = BULK ? __shfl_sync(warp, rinv_lane, i) : rsqrtf(v[i] / H + EPS);
      if constexpr (!BULK) rinv[i] = r;
#pragma unroll
      for (int u = 0; u < QUADS; ++u) s[i][u] = scale4(s[i][u], r);  // n_i
    }
    const float inv_sqrt_h = 1.f / 48.f;  // H ** -0.5
    float4 mix[Q][QUADS];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
      float lg[NS], den = 0.f, p_lane;
      if constexpr (BULK) {
        p_lane = mix_probability<NS>(sum, rinv_lane, state, j);
      } else {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
          lg[i] = v[NS + j * NS + i] * rinv[i] * inv_sqrt_h;
          mx = fmaxf(mx, lg[i]);
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
          lg[i] = expf(lg[i] - mx);
          den += lg[i];
        }
      }
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int i = 0; i < NS; ++i) {
          const float p = BULK ? __shfl_sync(warp, p_lane, i) : lg[i] / den;
          o = make_float4(o.x + p * s[i][u].x, o.y + p * s[i][u].y, o.z + p * s[i][u].z, o.w + p * s[i][u].w);
        }
        mix[j][u] = o;
      }
    }
    if (a.out_f32) {
#pragma unroll
      for (int j = 0; j < Q; ++j)
#pragma unroll
        for (int u = 0; u < QUADS; ++u)
          *reinterpret_cast<float4*>(a.out_f32 + size_t(j) * a.M * H + row + ch(u)) = mix[j][u];
    }
    if (!a.out_bf16) return;
    if (RAW || !a.norm_w) {
#pragma unroll
      for (int j = 0; j < Q; ++j)
#pragma unroll
        for (int u = 0; u < QUADS; ++u)
          store_bf16x4(a.out_bf16 + size_t(j) * a.M * H + row + ch(u), mix[j][u]);
    } else {
#pragma unroll
      for (int u = 0; u < QUADS; ++u) {
        float p[Q];
#pragma unroll
        for (int j = 0; j < Q; ++j) p[j] = sumsq4(mix[j][u]);
        sum_part<THREADS>(p, red3, u);
      }
      float ms[Q];
      sum_total(ms, red3);
#pragma unroll
      for (int j = 0; j < Q; ++j) {
        const float r = rsqrtf(ms[j] / H + EPS);
#pragma unroll
        for (int u = 0; u < QUADS; ++u) {
          const float4 g = g4[j][u];
          const float4 n = scale4(mix[j][u], r);
          const float4 o = make_float4(n.x * g.x, n.y * g.y, n.z * g.z, n.w * g.w);
          store_bf16x4(a.out_bf16 + size_t(j) * a.M * H + row + ch(u), o);
          if (j == 0 && a.out_copy) store_bf16x4(a.out_copy + m * a.out_copy_stride + ch(u), o);
        }
      }
    }
  }
}

template <int NS, int Q, bool COPIES>
__device__ __forceinline__ void mix_token(const MixArgs& a, int m, float* red1, float* red2, float* red3) {
  MixWeights<NS, Q, MIX_T> w;
  load_mix_weights<false>(a, w);
  pdl_wait_fenced(a.zero);  // keeps the constant loads ahead of the wait
  pdl_trigger();
  mix_row<COPIES, false, false>(a, w, m, red1, red2, red3);
}

template <int RED2, int RED3>
struct MixScratch {
  float red1[REF_W * block_sum_slots<2>()];
  float red2[RED2];
  float red3[RED3];
};

template <int NS, int Q, bool RAW, class Scratch>
__device__ __forceinline__ void mix_tile(const MixArgs& a, int tile, Scratch (&scratch)[2]) {
  MixWeights<NS, Q, BULK_T> w;
  load_mix_weights<RAW>(a, w);
  const int first = tile * BULK_ROWS;
  const int count = min(BULK_ROWS, a.M - first);
  // Alternate scratch storage. Any row using scratch has a block barrier, so every warp finishes row
  // i's reads before reaching the first barrier in row i+1. The pointer flags are the same for all rows.
#pragma unroll 1
  for (int i = 0; i < count; ++i) {
    Scratch& s = scratch[i & 1];
    mix_row<false, true, RAW>(a, w, first + i, s.red1, s.red2, s.red3);
  }
}

template <int NS, int Q, bool COPIES>
__global__ void __launch_bounds__(MIX_T) mix_kernel(MixArgs a) {
  __shared__ float red1[REF_W * block_sum_slots<2>()], red2[REF_W * block_sum_slots<NS + Q * NS>()],
      red3[REF_W * block_sum_slots<Q>()];
  mix_token<NS, Q, COPIES>(a, blockIdx.x, red1, red2, red3);
}

template <int NS, int Q, bool RAW>
__global__ void __launch_bounds__(BULK_T) mix_bulk_kernel(MixArgs a) {
  __shared__ MixScratch<REF_W * block_sum_slots<NS + Q * NS>(), RAW ? 1 : REF_W * block_sum_slots<Q>()> scratch[2];
  mix_tile<NS, Q, RAW>(a, blockIdx.x, scratch);
}

// Two independent mixes in one launch: the first half of the grid runs a, the second half runs the
// message mix b (one block, four queries). The launcher rejects row copies for both mixes.
template <int NS, int Q>
__global__ void __launch_bounds__(MIX_T) mix_pair_kernel(MixArgs a, MixArgs b) {
  __shared__ float red1[REF_W * block_sum_slots<2>()], red2[REF_W * 32], red3[REF_W * MAXQ];
  if (static_cast<int>(blockIdx.x) < a.M) mix_token<NS, Q, false>(a, blockIdx.x, red1, red2, red3);
  else mix_token<2, 4, false>(b, blockIdx.x - a.M, red1, red2, red3);
}

template <int NS, int Q>
__global__ void __launch_bounds__(BULK_T) mix_pair_bulk_kernel(MixArgs a, MixArgs b) {
  __shared__ MixScratch<REF_W * 32, 1> scratch[2];
  const int tiles = (a.M - 1) / BULK_ROWS + 1;
  if (static_cast<int>(blockIdx.x) < tiles) mix_tile<NS, Q, true>(a, blockIdx.x, scratch);
  else mix_tile<2, 4, true>(b, blockIdx.x - tiles, scratch);
}

template <int NS, int Q>
void launch_mix(int M, const MixArgs& a, bool pdl, cudaStream_t stream) {
  if (a.copies[0].dst || a.copies[1].dst) {
    launch_ex(mix_kernel<NS, Q, true>, dim3(M), MIX_T, 0, 1, pdl, stream, a);
    return;
  }
  if (!pdl && M >= BULK_MIN_ROWS) {
    const dim3 grid((M - 1) / BULK_ROWS + 1);
    if (!a.norm_w) {
      launch_ex(mix_bulk_kernel<NS, Q, true>, grid, BULK_T, 0, 1, false, stream, a);
      return;
    }
    // Prefill's four-query message mix is unnormalized. Larger normalized four-query mixes spill
    // with the tiled kernel, so keep them on the one-row launch.
    if constexpr (NS < 4 || Q < 4) {
      launch_ex(mix_bulk_kernel<NS, Q, false>, grid, BULK_T, 0, 1, false, stream, a);
      return;
    }
  }
  launch_ex(mix_kernel<NS, Q, false>, dim3(M), MIX_T, 0, 1, pdl, stream, a);
}

template <int NS>
void launch_mix_q(int Q, int M, const MixArgs& a, bool pdl, cudaStream_t stream) {
  switch (Q) {
    case 1: launch_mix<NS, 1>(M, a, pdl, stream); break;
    case 2: launch_mix<NS, 2>(M, a, pdl, stream); break;
    case 3: launch_mix<NS, 3>(M, a, pdl, stream); break;
    case 4: launch_mix<NS, 4>(M, a, pdl, stream); break;
  }
}

// embed: out[m] = float(bf16(rms_norm(float(table[token[m]])) * norm_w)), one CTA per token.
// Bit-exact with model.Model.embed: the sum of squares follows torch's CUDA mean reduction for a call
// with M rows (ATen Reduce.cuh: input vectorised by 4, block width BW = 512 / last_pow2(M) threads,
// thread t accumulates vectors t, t + BW, ... in 4 per-lane partials combined in order, then a shared
// tree for offsets BW/2 .. 32 and a shuffle-down tree 16 .. 1), mean = sum * fl(1/2304), rsqrtf as
// torch.rsqrt, and every product rounded separately (no FMA contraction) as torch's separate kernels.
template <int BW>
__global__ void __launch_bounds__(BW) embed_kernel(const int64_t* __restrict__ token,
                                                  const __nv_bfloat16* __restrict__ table,
                                                  const float* __restrict__ norm_w, float* __restrict__ out) {
  constexpr int NV = H / 4;                   // 576 four-element vectors
  constexpr int PER = (NV + BW - 1) / BW;     // vectors per thread (some threads one fewer)
  __shared__ float part[BW];
  __shared__ float rinv_s;
  const int m = blockIdx.x, t = threadIdx.x;
  float4 g[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k)
    if (t + k * BW < NV) g[k] = *reinterpret_cast<const float4*>(norm_w + 4 * (t + k * BW));
  pdl_wait();  // the token id comes from an earlier kernel
  pdl_trigger();
  const __nv_bfloat16* row = table + size_t(token[m]) * H;
  float4 v[PER];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (t + k * BW < NV) {
      const uint2 u = *reinterpret_cast<const uint2*>(row + 4 * (t + k * BW));
      v[k] = make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
      acc[0] = __fadd_rn(acc[0], __fmul_rn(v[k].x, v[k].x));
      acc[1] = __fadd_rn(acc[1], __fmul_rn(v[k].y, v[k].y));
      acc[2] = __fadd_rn(acc[2], __fmul_rn(v[k].z, v[k].z));
      acc[3] = __fadd_rn(acc[3], __fmul_rn(v[k].w, v[k].w));
    }
  }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
  if constexpr (BW > 32) {
    part[t] = s;
#pragma unroll
    for (int o = BW / 2; o >= 32; o >>= 1) {
      __syncthreads();
      if (t < o) {
        s = __fadd_rn(s, part[t + o]);
        part[t] = s;
      }
    }
    __syncthreads();
  }
  if (t < 32) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, o));
    if (t == 0) rinv_s = rsqrtf(__fadd_rn(__fmul_rn(s, 1.f / H), EPS));
  }
  __syncthreads();
  const float rinv = rinv_s;
  float* o = out + size_t(m) * H;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (t + k * BW < NV) {
      const float4 x = v[k], w = g[k];
      *reinterpret_cast<float4*>(o + 4 * (t + k * BW)) =
          make_float4(round_bf16(__fmul_rn(__fmul_rn(x.x, rinv), w.x)), round_bf16(__fmul_rn(__fmul_rn(x.y, rinv), w.y)),
                      round_bf16(__fmul_rn(__fmul_rn(x.z, rinv), w.z)), round_bf16(__fmul_rn(__fmul_rn(x.w, rinv), w.w)));
    }
  }
}

}  // namespace

namespace {

// Validates `p` and builds the kernel arguments.
MixArgs mix_args(const MixParams& p) {
  require(p.base || p.write, "mix needs base and/or write");
  require(p.M >= 1, "mix: M must be >= 1");
  const int nb = p.nblocks;
  require(nb >= 0 && nb <= MAXB, "mix: at most 5 blocks");
  require(nb == 0 || p.blocks, "mix: blocks table required");
  const int Q = p.Q;
  require(Q >= 1 && Q <= MAXQ, "mix: 1..4 queries");
  MixArgs a;
  a.blocks = nb > 0 ? p.blocks : nullptr;
  a.base = p.base;
  a.write = p.write;
  a.abl_r = p.abl_r;
  a.abl_d = a.abl_r ? p.abl_d : nullptr;
  a.abl_w = p.abl_w;
  require(!a.abl_r || a.abl_w, "mix: abl_w is required with abl_r");
  a.partial = p.partial_out;
  const bool skip = nb == 0 && Q == 1;
  a.queries = p.queries;
  require(skip || a.queries, "mix: queries are required unless the mix is skipped");
  a.out_f32 = p.out_f32;
  a.out_bf16 = p.out_bf16;
  a.norm_w = p.out_bf16 ? p.norm_w : nullptr;
  a.new_bf16 = p.new_bf16;
  a.new_bf16_stride = p.new_bf16_stride;
  require(a.new_bf16_stride % 4 == 0 && p.out_copy_stride % 4 == 0, "mix: bf16 copy strides must be multiples of 4");
  a.out_copy = p.out_copy;
  a.out_copy_stride = p.out_copy_stride;
  require(!a.out_copy || (a.out_bf16 && a.norm_w && Q == 1), "mix: out_copy needs out_bf16, norm_w and Q == 1");
  for (int c = 0; c < 2; ++c) {
    a.copies[c] = p.copies[c];
    require(!a.copies[c].dst || (a.copies[c].src && a.copies[c].cols % 4 == 0 && a.copies[c].src_stride % 4 == 0 &&
                                 a.copies[c].dst_stride % 4 == 0),
            "mix: row copies need 16-byte aligned rows");
  }
  require(a.out_f32 || a.out_bf16 || a.partial || a.new_bf16, "mix has no output");
  a.M = p.M;
  a.zero = 0;
  return a;
}

template <int NS>
void launch_pair_q(int Q, int M, const MixArgs& a, const MixArgs& b, bool pdl, cudaStream_t stream) {
  if (!pdl && M >= BULK_MIN_ROWS && !a.norm_w && !b.norm_w) {
    const dim3 grid(2 * ((M - 1) / BULK_ROWS + 1));
    switch (Q) {
      case 1: launch_ex(mix_pair_bulk_kernel<NS, 1>, grid, BULK_T, 0, 1, false, stream, a, b); return;
      case 3: launch_ex(mix_pair_bulk_kernel<NS, 3>, grid, BULK_T, 0, 1, false, stream, a, b); return;
    }
  }
  switch (Q) {
    case 1: launch_ex(mix_pair_kernel<NS, 1>, dim3(2 * M), MIX_T, 0, 1, pdl, stream, a, b); break;
    case 3: launch_ex(mix_pair_kernel<NS, 3>, dim3(2 * M), MIX_T, 0, 1, pdl, stream, a, b); break;
    default: require(false, "mix_pair: the first mix takes 1 or 3 queries");
  }
}

}  // namespace

void mix(const MixParams& p, bool pdl, cudaStream_t stream) {
  const MixArgs a = mix_args(p);
  switch (p.nblocks + 1) {
    case 1: launch_mix_q<1>(p.Q, p.M, a, pdl, stream); break;
    case 2: launch_mix_q<2>(p.Q, p.M, a, pdl, stream); break;
    case 3: launch_mix_q<3>(p.Q, p.M, a, pdl, stream); break;
    case 4: launch_mix_q<4>(p.Q, p.M, a, pdl, stream); break;
    case 5: launch_mix_q<5>(p.Q, p.M, a, pdl, stream); break;
    case 6: launch_mix_q<6>(p.Q, p.M, a, pdl, stream); break;
  }
}

void mix_pair(const MixParams& p, const MixParams& msg, bool pdl, cudaStream_t stream) {
  const MixArgs a = mix_args(p), b = mix_args(msg);
  require(msg.nblocks == 1 && msg.Q == 4, "mix_pair: the message mix takes one block and four queries");
  require(p.M == msg.M, "mix_pair: both mixes take the same tokens");
  require(!a.copies[0].dst && !a.copies[1].dst && !b.copies[0].dst && !b.copies[1].dst,
          "mix_pair: row copies go with mix");
  switch (p.nblocks + 1) {
    case 2: launch_pair_q<2>(p.Q, p.M, a, b, pdl, stream); break;
    case 3: launch_pair_q<3>(p.Q, p.M, a, b, pdl, stream); break;
    case 4: launch_pair_q<4>(p.Q, p.M, a, b, pdl, stream); break;
    case 5: launch_pair_q<5>(p.Q, p.M, a, b, pdl, stream); break;
    default: require(false, "mix_pair: the first mix takes 1..4 blocks");
  }
}

void embed(const std::int64_t* token, int M, const __nv_bfloat16* table, const float* norm_w, float* out,
           bool pdl, cudaStream_t stream) {
  require(M >= 1, "embed: M must be >= 1");  // one CTA per row (prefill calls it with M = chunk length)
  require(token && table && norm_w && out, "embed: null argument");
  // torch's reduction block width for an [M, 2304] fp32 mean: 512 / last_pow2(M)
  if (M == 1) launch_ex(embed_kernel<512>, dim3(M), 512, 0, 1, pdl, stream, token, table, norm_w, out);
  else if (M < 4) launch_ex(embed_kernel<256>, dim3(M), 256, 0, 1, pdl, stream, token, table, norm_w, out);
  else if (M < 8) launch_ex(embed_kernel<128>, dim3(M), 128, 0, 1, pdl, stream, token, table, norm_w, out);
  else launch_ex(embed_kernel<64>, dim3(M), 64, 0, 1, pdl, stream, token, table, norm_w, out);
}

}  // namespace ingine::kernels::dense
