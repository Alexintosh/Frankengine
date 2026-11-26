// Prefill attention (flex.attend + model._visit's output gate) as a flash-attention kernel for 24 query /
// 8 KV heads, head dim 128.
//
// Design. A CTA owns one KV head and 128 query rows of the chunk, rows ordered (token, head of the GQA group):
// row R is token R / 3, head 3 g + R % 3, so all of them read the same keys and values and every key/value
// tile in shared memory serves 128 rows. 8 warps of 16 rows. Q (BF16) stays in registers as m16n8k16 A
// fragments. Per 64-key tile: S = Q K^T with BF16 MMA and FP32 accumulation (BF16 values are exact, as in
// flex's TF32 dots), scale 1/sqrt(128), relative bias rel_logits[t, h, d] for distances d < 1024, mask
// (causal, window, empty slots), online softmax in FP32 (base 2), P = FP16(p * 2^15) (an FP16 / TF32 10-bit
// mantissa without FP16's underflow below 2^-14), O += P V with FP16 MMA (values converted BF16/E4M3 -> FP16
// when staged: exact for the model's RMS-normed values), FP32 accumulation. The row sum uses the FP32 p.
// Epilogue: o = O 2^-15 / l, lse = ln(sum e^s), o * sigmoid(lse - sink) rounded to BF16, times
// sigmoid(gate), rounded to BF16.
// Keys/values are staged through registers into double-buffered swizzled shared memory: tile i + 1 is loaded
// from global memory at the start of tile i and stored to the other buffer right after tile i's Q.K^T (one
// barrier per tile); fragment loads of a k16 group are issued before its MMAs. Keys are either implicit (key j at
// position j: the full-attention cache prefix) or explicit (kpos, the sliding ring + chunk rows) with per-tile
// position bounds deciding which tiles are skipped, need no mask, or need the bias. CTAs run heaviest row tiles
// (latest tokens) first.
// The fused sliding path uses a single shared K/V tile. Relative bias uses the retired K buffer after
// Q.K^T, then the next K tile is copied asynchronously while the current P.V runs.
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace ingine::kernels::prefill {
namespace {

constexpr int D = kHeadDim;   // 128
constexpr int DCH = D / 8;    // 16-byte chunks per 16-bit row
constexpr int BK = kAttnKeyTile;
constexpr int NT = 256, BR = NT / 2;
constexpr int GQA = kHeads / kKvHeads;
constexpr float kScale = 0.08838834764831845f;                    // 128 ** -0.5
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kPScale = 32768.f, kPUnscale = 1.f / 32768.f;

// A contiguous 64-key tile and 128 GQA rows span at most 64 + ceil(127 / 3) distances.
// Pad weights to the MMA width and keep the bias rows aligned for BF16 pair stores.
constexpr int kBiasCols = (BK + (BR + GQA - 2) / GQA + 3) / 4 * 4;
constexpr int kBiasMmaCols = (kBiasCols + 7) / 8 * 8;

struct RelTiles {
  alignas(16) uint16_t bias[BR][kBiasCols];
  alignas(16) uint16_t weights[kBiasMmaCols][kRel];
};
static_assert(offsetof(RelTiles, weights) >= BK * D * sizeof(uint16_t));
struct TileBounds { int4 data[kAttnMaxTiles]; };
struct NoTileBounds {};

template <bool Local>
struct Smem {
  static constexpr int Buffers = Local ? 1 : 2;
  static constexpr int ValueOffset = Local ? sizeof(RelTiles) / sizeof(uint16_t) : Buffers * BK * D;
  static_assert(ValueOffset >= Buffers * BK * D);
  // Initial Q staging uses both regions. During attention, the first region alternates K and relative bias
  // for the fused kernel, or holds two K buffers for the legacy kernel.
  alignas(16) uint16_t data[ValueOffset + Buffers * BK * D];
  int kpos[Buffers][BK];
  [[no_unique_address]] std::conditional_t<Local, NoTileBounds, TileBounds> bounds;
  int list[kAttnMaxTiles];
  int nlist;

  __device__ uint16_t* keys(int buf) { return data + buf * BK * D; }
  __device__ uint16_t* values(int buf) { return data + ValueOffset + buf * BK * D; }
  __device__ RelTiles& relative() { return *reinterpret_cast<RelTiles*>(data); }
};
static_assert(sizeof(Smem<true>) <= 48 * 1024);

template <bool FP8>
struct KvRegs {
  static constexpr int N = FP8 ? 2 : 4;  // 16-byte loads per thread per matrix
  uint4 k[N], v[N];
  int kp;
};

template <bool FP8, bool EXPLICIT, int KvHeads>
__device__ __forceinline__ void load_tile(const AttnParams& p, int g, int j0, KvRegs<FP8>& r) {
  const int tid = threadIdx.x;
  if constexpr (FP8) {
    const uint8_t* K = static_cast<const uint8_t*>(p.k);
    const uint8_t* V = static_cast<const uint8_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = size_t(j) * (KvHeads * D) + g * D + c * 16;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  } else {
    const uint16_t* K = static_cast<const uint16_t*>(p.k);
    const uint16_t* V = static_cast<const uint16_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = size_t(j) * (KvHeads * D) + g * D + c * 8;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) r.kp = __ldg(p.kpos + j0 + tid);
  }
}

__device__ __forceinline__ uint32_t bf16x2_to_half2(uint32_t u) { return pack_half2(bf_lo(u), bf_hi(u)); }

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t u) {
  const float2 f = e4m3x2_to_float2(u);
  return pack_bf16x2(f.x, f.y);
}

template <bool FP8, bool EXPLICIT>
__device__ __forceinline__ void store_tile(Smem<false>& s, int buf, const KvRegs<FP8>& r) {
  const int tid = threadIdx.x;
  if constexpr (FP8) {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const uint32_t kw[4] = {r.k[i].x, r.k[i].y, r.k[i].z, r.k[i].w};
      const uint32_t vw[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
      uint4 k0, k1, v0, v1;
      k0.x = e4m3x2_to_bf16x2(kw[0] & 0xffff); k0.y = e4m3x2_to_bf16x2(kw[0] >> 16);
      k0.z = e4m3x2_to_bf16x2(kw[1] & 0xffff); k0.w = e4m3x2_to_bf16x2(kw[1] >> 16);
      k1.x = e4m3x2_to_bf16x2(kw[2] & 0xffff); k1.y = e4m3x2_to_bf16x2(kw[2] >> 16);
      k1.z = e4m3x2_to_bf16x2(kw[3] & 0xffff); k1.w = e4m3x2_to_bf16x2(kw[3] >> 16);
      v0.x = e4m3x2_to_half2(vw[0] & 0xffff); v0.y = e4m3x2_to_half2(vw[0] >> 16);
      v0.z = e4m3x2_to_half2(vw[1] & 0xffff); v0.w = e4m3x2_to_half2(vw[1] >> 16);
      v1.x = e4m3x2_to_half2(vw[2] & 0xffff); v1.y = e4m3x2_to_half2(vw[2] >> 16);
      v1.z = e4m3x2_to_half2(vw[3] & 0xffff); v1.w = e4m3x2_to_half2(vw[3] >> 16);
      *reinterpret_cast<uint4*>(s.keys(buf) + swz<DCH>(row, 2 * c)) = k0;
      *reinterpret_cast<uint4*>(s.keys(buf) + swz<DCH>(row, 2 * c + 1)) = k1;
      *reinterpret_cast<uint4*>(s.values(buf) + swz<DCH>(row, 2 * c)) = v0;
      *reinterpret_cast<uint4*>(s.values(buf) + swz<DCH>(row, 2 * c + 1)) = v1;
    }
  } else {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      *reinterpret_cast<uint4*>(s.keys(buf) + swz<DCH>(row, c)) = r.k[i];
      uint4 v;
      v.x = bf16x2_to_half2(r.v[i].x);
      v.y = bf16x2_to_half2(r.v[i].y);
      v.z = bf16x2_to_half2(r.v[i].z);
      v.w = bf16x2_to_half2(r.v[i].w);
      *reinterpret_cast<uint4*>(s.values(buf) + swz<DCH>(row, c)) = v;
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) s.kpos[buf][tid] = r.kp;
  }
}

template <bool Keys, int KvHeads>
__device__ __forceinline__ void load_local_tile(Smem<true>& s, const AttnParams& p, int g, int j0) {
  const int tid = threadIdx.x;
  const auto* src = static_cast<const uint16_t*>(Keys ? p.k : p.v);
#pragma unroll
  for (int i = 0; i < BK * DCH / NT; ++i) {
    const int id = tid + i * NT, row = id / DCH, c = id % DCH;
    const int j = j0 + row;
    const size_t off = size_t(j) * KvHeads * D + g * D + c * 8;
    if constexpr (Keys) {
      cp_async16(s.keys(0) + swz<DCH>(row, c), j < p.L ? src + off : src, j < p.L ? 16 : 0);
    } else {
      uint4 v = j < p.L ? __ldg(reinterpret_cast<const uint4*>(src + off)) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(s.values(0) + swz<DCH>(row, c)) =
          make_uint4(bf16x2_to_half2(v.x), bf16x2_to_half2(v.y), bf16x2_to_half2(v.z), bf16x2_to_half2(v.w));
    }
  }
  if constexpr (Keys) cp_commit();
  else if (tid < BK) s.kpos[0][tid] = j0 + tid < p.L ? p.kpos[j0 + tid] : -1;
}

__device__ __forceinline__ void load_relative_weights(RelTiles& rel, const __nv_bfloat16* weights, int lo, int hi) {
  const int tid = threadIdx.x;
  const int cols = hi - lo + 1;
  const int groups = (cols + 7) / 8;
  if (tid < groups * 16) {
    const int col = tid / 2, half = (tid % 2) * 8;
    const bool valid = col < cols;
    cp_async16(&rel.weights[col][half], valid ? weights + (lo + col) * kRel + half : weights, valid ? 16 : 0);
  }
  cp_commit();
}

__device__ __forceinline__ void stage_relative_bias(RelTiles& rel, int cols, const unsigned (&q)[4], int qr,
                                                    int lane) {
  // Same BF16 m16n8k16 dot and rounding as rel_logits_mma_kernel, with distances as the N axis.
  const int groups = (cols + 7) / 8;
  for (int group = 0; group < groups; ++group) {
    const int br = group * 8 + (lane & 7), bc = ((lane >> 3) & 1) * 8;
    unsigned b0, b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b0), "=r"(b1) : "r"(smem_addr(&rel.weights[br][bc])));
    float bias[4] = {};
    mma_bf16(bias, q, b0, b1);
    const int c = group * 8 + (lane & 3) * 2;
    if (c < kBiasCols) {
      *reinterpret_cast<uint32_t*>(&rel.bias[qr][c]) = pack_bf16x2(bias[0], bias[1]);
      *reinterpret_cast<uint32_t*>(&rel.bias[qr + 8][c]) = pack_bf16x2(bias[2], bias[3]);
    }
  }
  __syncwarp();
}

__device__ __forceinline__ void local_scores(Smem<true>& s, const AttnParams& p, int first, int last, bool need_mask,
                                             const int (&qpos)[2], const unsigned (&relq)[4],
                                             float (&sc)[BK / 8][4], float (&rmax)[2], int warp, int lane) {
  const int qr = warp * 16 + (lane >> 2);
  const bool single_slab = last - first < kBiasCols;
  RelTiles& rel = s.relative();
  for (int lo = first; lo <= last; lo += kBiasCols) {
    const int hi = min(last, lo + kBiasCols - 1);
    if (lo != first) {
      __syncthreads();  // all warps have consumed the previous weights
      load_relative_weights(rel, p.rel_proj, lo, hi);
      cp_wait<0>();
      __syncthreads();
    }
    stage_relative_bias(rel, hi - lo + 1, relq, qr, lane);
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e >> 1, jl = nt * 8 + (lane & 3) * 2 + (e & 1);
        const int kp = s.kpos[0][jl], d = qpos[r] - kp;
        if (need_mask && (kp < 0 || d < 0 || d >= kRelExtent)) {
          sc[nt][e] = -INFINITY;
        } else if (single_slab || (lo <= d && d <= hi)) {
          // Keep the scale rounded before adding BF16 bias, as in the unfused score path.
          const float scaled = __fmul_rn(sc[nt][e], kScale);
          const float x = (scaled + bf2f(rel.bias[qr + r * 8][d - lo])) * kLog2e;
          sc[nt][e] = x;
          rmax[r] = fmaxf(rmax[r], x);
        }
      }
    }
  }
}

template <bool FP8, bool EXPLICIT, int KvHeads, bool Local>
__global__ void __launch_bounds__(NT, 1) attn_kernel(const AttnParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& s = *reinterpret_cast<Smem<Local>*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.y;
  constexpr int Heads = KvHeads * GQA;
  constexpr int GateCol = Heads * D, SinkCol = Heads * (2 * D + kRel);
  const int rows = GQA * p.T;
  const int R0 = (gridDim.x - 1 - blockIdx.x) * BR;  // heaviest (latest) rows first

  // ---- Q tile -> registers (staged through the key buffers)
  {
    uint16_t* stage = s.data;
#pragma unroll
    for (int i = 0; i < (BR * DCH) / NT; ++i) {
      const int id = tid + i * NT, r = id >> 4, c = id & 15;
      const int R = min(R0 + r, rows - 1);
      const int t = R / GQA, h = g * GQA + R % GQA;
      cp_async16(stage + swz<DCH>(r, c), p.q + (size_t(t) * Heads + h) * D + c * 8);
    }
    cp_commit();
  }
  // per-thread rows: lane / 4 and lane / 4 + 8 of the warp's 16
  int qpos[2];
  const uint16_t* relp[2];
  unsigned relq[4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int R = min(R0 + warp * 16 + (lane >> 2) + r * 8, rows - 1);
    const int t = R / GQA, h = g * GQA + R % GQA;
    qpos[r] = p.pos0 + t;
    if constexpr (Local) {
      // Direct loads of the four A registers that ldmatrix.x4 would produce for this pair of query rows.
      const auto* q = reinterpret_cast<const uint32_t*>(p.rel_query + (size_t(t) * Heads + h) * kRel);
      relq[r] = q[lane & 3];
      relq[r + 2] = q[4 + (lane & 3)];
    } else {
      relp[r] = reinterpret_cast<const uint16_t*>(p.rel) + (size_t(t) * Heads + h) * kRelExtent;
    }
  }
  const int qmin = p.pos0 + R0 / GQA;
  const int qmax = p.pos0 + min(R0 + BR - 1, rows - 1) / GQA;

  // ---- key tiles to visit
  int first = 0, ntiles = 0;
  if constexpr (EXPLICIT) {
    const int total = (p.L + BK - 1) / BK;
    if constexpr (Local) {
      if (warp == 0) {
        int n = 0;
        for (int base = 0; base < total; base += 32) {
          const int i = base + lane;
          const int4 b = i < total ? __ldg(p.bounds + i) : make_int4(0, 0, 0, 0);
          const bool reach = b.w && qmax - b.x >= 0 && qmin - b.y < kRelExtent;
          const unsigned mask = __ballot_sync(0xffffffffu, reach);
          if (reach) s.list[n + __popc(mask & ((1u << lane) - 1))] = i;
          n += __popc(mask);
        }
        if (lane == 0) s.nlist = n;
      }
    } else {
      if (tid < total) s.bounds.data[tid] = p.bounds[tid];
      __syncthreads();
      if (tid == 0) {
        int n = 0;
        for (int i = 0; i < total; ++i) {
          const int4 b = s.bounds.data[i];
          const bool reach = b.w && qmax - b.x >= 0 && (p.window == 0 || qmin - b.y < p.window);
          if (reach) s.list[n++] = i;
        }
        s.nlist = n;
      }
    }
  } else {
    const int lo = p.window ? max(0, qmin - p.window + 1) : 0;
    const int hi = min(qmax, p.L - 1);
    first = lo / BK;
    ntiles = hi >= lo ? hi / BK - first + 1 : 0;
  }
  cp_wait<0>();
  __syncthreads();
  if constexpr (EXPLICIT) ntiles = s.nlist;

  unsigned qf[DCH / 2][4];
#pragma unroll
  for (int kk = 0; kk < DCH / 2; ++kk)
    ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                smem_addr(s.data + swz<DCH>(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))));
  __syncthreads();  // Q staging is overwritten by the first key tile

  auto tile_at = [&](int i) { return EXPLICIT ? s.list[i] : first + i; };

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  KvRegs<FP8> regs;
  if (ntiles > 0) {
    if constexpr (Local) {
      load_local_tile<true, KvHeads>(s, p, g, tile_at(0) * BK);
      load_local_tile<false, KvHeads>(s, p, g, tile_at(0) * BK);
      cp_wait<0>();
    } else {
      load_tile<FP8, EXPLICIT, KvHeads>(p, g, tile_at(0) * BK, regs);
      store_tile<FP8, EXPLICIT>(s, 0, regs);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int kt = tile_at(i), buf = Local ? 0 : i & 1;
    const bool more = i + 1 < ntiles;
    if constexpr (!Local) {
      if (more) load_tile<FP8, EXPLICIT, KvHeads>(p, g, tile_at(i + 1) * BK, regs);
    }

    // tile class (uniform over the CTA)
    int kmin, kmax;
    bool allv;
    if constexpr (EXPLICIT) {
      int4 b;
      if constexpr (Local) b = __ldg(p.bounds + kt);
      else b = s.bounds.data[kt];
      kmin = b.x, kmax = b.y, allv = b.z != 0;
    } else {
      kmin = kt * BK;
      kmax = min(kt * BK + BK - 1, p.L - 1);
      allv = kt * BK + BK <= p.L;
    }
    const int dmin = qmin - kmax, dmax = qmax - kmin;
    const bool need_mask = !(allv && dmin >= 0 && (p.window == 0 || dmax < p.window));
    const bool need_bias = dmin < kRelExtent;
    const int rel_lo = max(0, dmin), rel_hi = min(kRelExtent - 1, dmax);
    if constexpr (Local)
      load_relative_weights(s.relative(), p.rel_proj, rel_lo, min(rel_hi, rel_lo + kBiasCols - 1));

    // ---- S = Q K^T
    float sc[BK / 8][4];
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
    const uint16_t* Ks = s.keys(buf);
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      // all of the k16 group's key fragments first, then its 8 MMAs (each MMA no longer waits on the ldmatrix
      // issued right before it)
      unsigned b[BK / 16][4];
#pragma unroll
      for (int np = 0; np < BK / 16; ++np)
        ldmatrix_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                    smem_addr(Ks + swz<DCH>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
      for (int np = 0; np < BK / 16; ++np) {
        mma_bf16(sc[2 * np], qf[kk], b[np][0], b[np][1]);
        mma_bf16(sc[2 * np + 1], qf[kk], b[np][2], b[np][3]);
      }
    }
    // stage the next tile now (its buffer was last read before the previous barrier): the conversion and the
    // shared stores overlap the Q.K^T MMAs in flight instead of trailing the P.V MMAs
    if constexpr (!Local) {
      if (more) store_tile<FP8, EXPLICIT>(s, buf ^ 1, regs);
    }

    // ---- scores (log2 units), bias, mask
    float rmax[2] = {m[0], m[1]};
    if constexpr (Local) {
      cp_wait<0>();
      __syncthreads();  // all warps have finished Q.K^T before relative bias reuses K
      local_scores(s, p, rel_lo, rel_hi, need_mask, qpos, relq, sc, rmax, warp, lane);
      if (more) {
        __syncthreads();  // all warps have consumed relative bias
        load_local_tile<true, KvHeads>(s, p, g, tile_at(i + 1) * BK);
      }
    } else {
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = e >> 1, jl = nt * 8 + (lane & 3) * 2 + (e & 1);
          float x = sc[nt][e] * kScale;
          if (need_mask || need_bias) {
            const int kp = EXPLICIT ? s.kpos[buf][jl] : kt * BK + jl;
            const int d = qpos[r] - kp;
            const bool ok = kp >= 0 && d >= 0 && (p.window == 0 || d < p.window) && (EXPLICIT || kp < p.L);
            if (need_bias && ok && d < kRelExtent) x += bf2f(__ldg(relp[r] + d));
            x = ok ? x * kLog2e : -INFINITY;
          } else {
            x *= kLog2e;
          }
          sc[nt][e] = x;
          rmax[r] = fmaxf(rmax[r], x);
        }
      }
    }
    float mref[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
      mref[r] = rmax[r] == -INFINITY ? 0.f : rmax[r];
      const float alpha = exp2f(m[r] - mref[r]);
      m[r] = rmax[r];
      l[r] *= alpha;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        o[dt][2 * r] *= alpha;
        o[dt][2 * r + 1] *= alpha;
      }
    }
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pv = exp2f(sc[nt][e] - mref[e >> 1]);
        l[e >> 1] += pv;
        sc[nt][e] = pv * kPScale;
      }
    }

    // ---- O += P V
    const uint16_t* Vs = s.values(buf);
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const unsigned a[4] = {pack_half2(sc[2 * kk][0], sc[2 * kk][1]), pack_half2(sc[2 * kk][2], sc[2 * kk][3]),
                             pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                             pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
#pragma unroll
      for (int dq = 0; dq < D / 32; ++dq) {  // two value fragment loads, then their four MMAs
        unsigned b[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
          ldmatrix_x4_t(b[h][0], b[h][1], b[h][2], b[h][3],
                        smem_addr(Vs + swz<DCH>(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                                (2 * dq + h) * 2 + (lane >> 4))));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mma_f16(o[2 * (2 * dq + h)], a, b[h][0], b[h][1]);
          mma_f16(o[2 * (2 * dq + h) + 1], a, b[h][2], b[h][3]);
        }
      }
    }

    if constexpr (Local) {
      if (more) {
        __syncthreads();  // retire the current V tile before reusing its buffer
        load_local_tile<false, KvHeads>(s, p, g, tile_at(i + 1) * BK);
        cp_wait<0>();
        __syncthreads();
      }
    } else {
      __syncthreads();
    }
  }

  // ---- epilogue
  const uint16_t* proj = reinterpret_cast<const uint16_t*>(p.proj);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float lr = l[r];
    lr += __shfl_xor_sync(0xffffffffu, lr, 1);
    lr += __shfl_xor_sync(0xffffffffu, lr, 2);
    const int R = R0 + warp * 16 + (lane >> 2) + r * 8;
    if (R >= rows) continue;
    const int t = R / GQA, h = g * GQA + R % GQA;
    const uint16_t* prow = proj + size_t(t) * p.ld;
    const float lse = (m[r] + log2f(lr)) * kLn2;
    const float factor = 1.f / (1.f + expf(-(lse - bf2f(prow[SinkCol + h]))));
    uint16_t* out = reinterpret_cast<uint16_t*>(p.out) + (size_t(t) * Heads + h) * D;
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
      const int col = dt * 8 + (lane & 3) * 2;
      const uint32_t gw = *reinterpret_cast<const uint32_t*>(prow + GateCol + h * D + col);
      const float g0 = bf_lo(gw), g1 = bf_hi(gw);
      const float a0 = round_bf16(o[dt][2 * r] * kPUnscale / lr * factor);
      const float a1 = round_bf16(o[dt][2 * r + 1] * kPUnscale / lr * factor);
      const float s0 = 1.f / (1.f + expf(-g0)), s1 = 1.f / (1.f + expf(-g1));
      *reinterpret_cast<uint32_t*>(out + col) = pack_bf16x2(a0 * s0, a1 * s1);
    }
  }
}

template <bool FP8, bool EXPLICIT, int KvHeads, bool Local = false>
void launch(const AttnParams& p, cudaStream_t stream) {
  constexpr int shared_bytes = sizeof(Smem<Local>);
  static const cudaError_t attr =
      cudaFuncSetAttribute(attn_kernel<FP8, EXPLICIT, KvHeads, Local>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (attr != cudaSuccess) throw std::runtime_error("prefill attention: shared memory attribute failed");
  const dim3 grid((GQA * p.T + BR - 1) / BR, KvHeads);
  attn_kernel<FP8, EXPLICIT, KvHeads, Local><<<grid, NT, shared_bytes, stream>>>(p);
}

// ---- sliding-visit key rows

__global__ void ring_gather_kernel(const uint4* __restrict__ k_ring, const uint4* __restrict__ v_ring,
                                   const int64_t* __restrict__ ring_pos, int ring_rows, int pos0, int T,
                                   uint4* __restrict__ k_rows, uint4* __restrict__ v_rows, int* __restrict__ kpos,
                                   int kpos_rows) {
  const int kRowVecs = blockDim.x;
  const int i = blockIdx.x;
  const int slot = ring_rows == kRelExtent ? ((pos0 + i) & (kRelExtent - 1)) : i;
  const bool has = slot < ring_rows;
  k_rows[size_t(i) * kRowVecs + threadIdx.x] = has ? k_ring[size_t(slot) * kRowVecs + threadIdx.x] : make_uint4(0, 0, 0, 0);
  v_rows[size_t(i) * kRowVecs + threadIdx.x] = has ? v_ring[size_t(slot) * kRowVecs + threadIdx.x] : make_uint4(0, 0, 0, 0);
  if (threadIdx.x == 0) kpos[i] = has ? int(ring_pos[slot]) : -1;
  for (int r = kRelExtent + i * kRowVecs + threadIdx.x; r < kpos_rows; r += gridDim.x * kRowVecs)
    kpos[r] = r - kRelExtent < T ? pos0 + r - kRelExtent : -1;
}

__global__ void ring_update_kernel(const uint4* __restrict__ k_rows, const uint4* __restrict__ v_rows, int pos0, int T,
                                   uint4* __restrict__ k_ring, uint4* __restrict__ v_ring, int64_t* __restrict__ ring_pos) {
  const int kRowVecs = blockDim.x;
  const int n = min(T, kRelExtent);
  const int c = T - n + blockIdx.x;  // chunk row
  const int pos = pos0 + c, slot = pos & (kRelExtent - 1);
  k_ring[size_t(slot) * kRowVecs + threadIdx.x] = k_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  v_ring[size_t(slot) * kRowVecs + threadIdx.x] = v_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  if (threadIdx.x == 0) ring_pos[slot] = pos;
}

__global__ void bounds_kernel(const int* __restrict__ kpos, int L, int4* __restrict__ bounds) {
  const int tile = blockIdx.x, lane = threadIdx.x;
  int lo = INT_MAX, hi = INT_MIN, valid = 0;
#pragma unroll
  for (int i = 0; i < BK / 32; ++i) {
    const int j = tile * BK + i * 32 + lane;
    const int kp = j < L ? kpos[j] : -1;
    if (kp >= 0) {
      lo = min(lo, kp);
      hi = max(hi, kp);
      ++valid;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
    hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    valid += __shfl_xor_sync(0xffffffffu, valid, o);
  }
  if (lane == 0) bounds[tile] = make_int4(lo, hi, valid == BK ? 1 : 0, valid > 0 ? 1 : 0);
}

}  // namespace

void attention(const AttnParams& p, cudaStream_t stream) {
  const bool local = p.rel_query || p.rel_proj;
  if (p.T <= 0 || p.L <= 0 || !p.q || !p.k || !p.v || (!local && !p.rel) || !p.proj || !p.out)
    throw std::invalid_argument("prefill attention: missing argument");
  const bool expl = p.kpos != nullptr;
  if (expl && (!p.bounds || (p.L + BK - 1) / BK > kAttnMaxTiles))
    throw std::invalid_argument("prefill attention: explicit keys need bounds and at most 64 tiles");
  if (local && (!p.rel_query || !p.rel_proj || !expl || p.kv_fp8 || p.window != kRelExtent))
    throw std::invalid_argument("prefill attention: tiled bias requires explicit BF16 sliding keys");
  auto dispatch = [&]<int KvHeads>() {
    if (local) {
      launch<false, true, KvHeads, true>(p, stream);
    } else if (p.kv_fp8) {
      if (expl) launch<true, true, KvHeads>(p, stream);
      else launch<true, false, KvHeads>(p, stream);
    } else {
      if (expl) launch<false, true, KvHeads>(p, stream);
      else launch<false, false, KvHeads>(p, stream);
    }
  };
  if (p.hkv == kKvHeads) dispatch.template operator()<kKvHeads>();
  else if (p.hkv == kKvHeads / 2) dispatch.template operator()<kKvHeads / 2>();
  else throw std::invalid_argument("prefill attention: unsupported KV head count");
}

void ring_gather(const __nv_bfloat16* k_ring, const __nv_bfloat16* v_ring, const int64_t* ring_pos, int pos0, int T,
                 __nv_bfloat16* k_rows, __nv_bfloat16* v_rows, int* kpos, int kpos_rows, cudaStream_t stream, int hkv, int ring_rows) {
  ring_gather_kernel<<<kRelExtent, hkv * D / 8, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_ring), reinterpret_cast<const uint4*>(v_ring), ring_pos, ring_rows, pos0, T,
      reinterpret_cast<uint4*>(k_rows), reinterpret_cast<uint4*>(v_rows), kpos, kpos_rows);
}

void ring_update(const __nv_bfloat16* k_rows, const __nv_bfloat16* v_rows, int pos0, int T, __nv_bfloat16* k_ring,
                 __nv_bfloat16* v_ring, int64_t* ring_pos, cudaStream_t stream, int hkv) {
  ring_update_kernel<<<std::min(T, kRelExtent), hkv * D / 8, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_rows), reinterpret_cast<const uint4*>(v_rows), pos0, T,
      reinterpret_cast<uint4*>(k_ring), reinterpret_cast<uint4*>(v_ring), ring_pos);
}

void key_tile_bounds(const int* kpos, int L, int4* bounds, cudaStream_t stream) {
  bounds_kernel<<<(L + BK - 1) / BK, 32, 0, stream>>>(kpos, L, bounds);
}

}  // namespace ingine::kernels::prefill
