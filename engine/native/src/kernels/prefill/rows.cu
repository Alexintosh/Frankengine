// Prefill per-row kernels: attention input post-processing (model._qkv), relative logits, the short
// convolutions over time, message-gate pieces, taps, and the last row's logits. Arithmetic follows the PyTorch
// reference operation by operation (separately rounded products/sums where torch runs separate kernels;
// divisions by Python scalars as multiplications by the FP32 reciprocal, as torch does).
#include "kernels/prefill/common.cuh"
#include "kernels/common/kv12.cuh"
#include "kernels/common/pico12.cuh"
#include "kernels/common/q4_g64.cuh"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace ingine::kernels::prefill {
namespace {

constexpr int H = kHidden;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// model.l2_clip factor: (threshold * reciprocal(norm + 1e-6)).clamp(max=1) (torch scalar / tensor).
__device__ __forceinline__ float clip_factor(float sumsq, float threshold) {
  return fminf((1.f / (sqrtf(sumsq) + kEps)) * threshold, 1.f);
}

__device__ __forceinline__ void load_bf16x4(const uint16_t* p, float (&f)[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  f[0] = bf_lo(u.x), f[1] = bf_hi(u.x), f[2] = bf_lo(u.y), f[3] = bf_hi(u.y);
}

__device__ __forceinline__ void store_bf16x4(uint16_t* p, const float (&f)[4]) {
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]));
}

// ---- attn_prep: one CTA (8 warps) per token

template <int KvHeads, bool WriteRing, bool Q4 = false, bool KV12 = false>
__global__ void __launch_bounds__(256) attn_prep_kernel(const AttnPrepParams p) {
  constexpr int Heads = KvHeads * (kHeads / kKvHeads), KvDim = KvHeads * kHeadDim;
  constexpr int OffRel = 2 * Heads * kHeadDim, OffK = OffRel + Heads * (kRel + 1);
  constexpr int OffV = OffK + KvDim, OffKdyn = OffV + KvDim, OffVdyn = OffKdyn + 4 * KvHeads;
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* row = reinterpret_cast<const uint16_t*>(p.proj) + size_t(t) * p.ld;
  const int pos = p.pos0 + t;
  const bool publish = WriteRing && t >= p.T - kRelExtent;
  const int slot = pos & (kRelExtent - 1);
  // model.log_scale: 1 + 0.1 * log(clamp((p + 1) / 8192, min=1))
  const float tau = 1.f + 0.1f * logf(fmaxf(float(pos + 1) * (1.f / 8192.f), 1.f));
  const float qk_clip = 22.627416997969522f, rel_clip = 8.f;

  // q: heads warp, warp + 8, warp + 16; 4 dims per lane
#pragma unroll
  for (int h = warp; h < Heads; h += 8) {
    const int d = lane * 4;
    float x[4];
    load_bf16x4(row + h * kHeadDim + d, x);
    const float4 sc = *reinterpret_cast<const float4*>(p.q_scale + h * kHeadDim + d);
    x[0] *= sc.x, x[1] *= sc.y, x[2] *= sc.z, x[3] *= sc.w;
    const float f = clip_factor(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), qk_clip);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[j] = round_bf16(x[j] * f);
      if (p.full) x[j] = round_bf16(x[j] * tau);
    }
    store_bf16x4(reinterpret_cast<uint16_t*>(p.q) + (size_t(t) * Heads + h) * kHeadDim + d, x);
  }
  // rel: heads warp, warp + 8, warp + 16; lanes 0-15 one dim each (16-31 duplicate)
#pragma unroll
  for (int h = warp; h < Heads; h += 8) {
    const int j = lane & 15;
    float x = bf2f(row[OffRel + h * kRel + j]) * p.r_scale[h * kRel + j];
    float ss = x * x;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    x = round_bf16(x * clip_factor(ss, rel_clip));
    if (p.full) x = round_bf16(x * tau);
    if (lane < 16) reinterpret_cast<uint16_t*>(p.rel)[(size_t(t) * Heads + h) * kRel + j] = f2bf_bits(x);
  }
  if (publish && threadIdx.x == 0) p.ring_pos[slot] = pos;
  // keys / values of KV head `warp`, 4 channels per lane
  if (warp >= KvHeads) return;
  const int c = warp * kHeadDim + lane * 4;
#pragma unroll
  for (int kv = 0; kv < 2; ++kv) {
    const int off = kv == 0 ? OffK : OffV;
    const float* state = kv == 0 ? p.conv_k : p.conv_v;
    const float* wt = kv == 0 ? p.k_w : p.v_w;
    // ext[j] = conv input at t - 3 + j (BF16-rounded projections; the state before the chunk)
    float ext[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int src = t - 3 + j;
      if (src >= 0) {
        load_bf16x4(reinterpret_cast<const uint16_t*>(p.proj) + size_t(src) * p.ld + off + c, ext[j]);
      } else {
        const float4 s4 = *reinterpret_cast<const float4*>(state + (3 + src) * KvDim + c);
        ext[j][0] = s4.x, ext[j][1] = s4.y, ext[j][2] = s4.z, ext[j][3] = s4.w;
      }
    }
    float y[4];
    if (p.full) {  // model.dynamic_conv: w = static_w[lag] + dyn[t, head, lag]; (sum + x) / 3
      const int dyn_off = kv == 0 ? OffKdyn : OffVdyn;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int lag = 0; lag < 4; ++lag) {
        const float dyn = bf2f(row[dyn_off + warp * 4 + lag]);
        const float4 sw = *reinterpret_cast<const float4*>(wt + lag * KvDim + c);
        const float swv[4] = {sw.x, sw.y, sw.z, sw.w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
          acc[q] = __fadd_rn(acc[q], __fmul_rn(__fadd_rn(swv[q], dyn), ext[3 - lag][q]));
      }
      const float inv = 1.f / 3.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(__fmul_rn(__fadd_rn(acc[q], ext[3][q]), inv));
    } else {  // model.static_conv: weight [C, 4] conv1d order, (sum + x) / sqrt(5)
      const float inv = 1.f / 2.2360679774997896f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 w4 = *reinterpret_cast<const float4*>(wt + size_t(c + q) * 4);
        float acc = __fmul_rn(w4.x, ext[0][q]);
        acc = __fadd_rn(acc, __fmul_rn(w4.y, ext[1][q]));
        acc = __fadd_rn(acc, __fmul_rn(w4.z, ext[2][q]));
        acc = __fadd_rn(acc, __fmul_rn(w4.w, ext[3][q]));
        y[q] = round_bf16(__fmul_rn(__fadd_rn(acc, ext[3][q]), inv));
      }
    }
    const float ss = warp_sum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
    if (kv == 0) {  // keys: l2_clip(QK_CLIP), BF16
      const float f = clip_factor(ss, qk_clip);
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(y[q] * f);
    } else {  // values: rms_norm without weight, BF16
      const float r = rsqrtf(ss * (1.f / kHeadDim) + kEps);
#pragma unroll
      for (int q = 0; q < 4; ++q) y[q] = round_bf16(y[q] * r);
    }
    void* dst = kv == 0 ? p.k_out : p.v_out;
    if constexpr (KV12) {
      const size_t head = size_t(t) * KvHeads + warp;
      const uint2 bits = make_uint2(pack_bf16x2(y[0], y[1]), pack_bf16x2(y[2], y[3]));
      void* side = kv == 0 ? p.k_scale_out : p.v_scale_out;
      kv12::pack4(bits, static_cast<kv12::Head*>(dst)[head], static_cast<kv12::Spill*>(side)[head]);
    } else if constexpr (Q4) {
      const size_t head = size_t(t) * KvHeads + warp;
      __half scale;
      const uint16_t bits = q4_g64::pack4(y, scale);
      static_cast<uint16_t*>(dst)[head * (q4_g64::head_bytes / 2) + lane] = bits;
      if ((lane & 15) == 0) {
        auto* scales = static_cast<__half*>(kv == 0 ? p.k_scale_out : p.v_scale_out);
        scales[head * q4_g64::groups + (lane >> 4)] = scale;
      }
    } else if (p.out_fp8) {  // State.full_store: saturate to +-448, E4M3 round to nearest even
      const uint32_t b = uint32_t(f2e4m3(y[0])) | (uint32_t(f2e4m3(y[1])) << 8) | (uint32_t(f2e4m3(y[2])) << 16) |
                         (uint32_t(f2e4m3(y[3])) << 24);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(dst) + size_t(t) * KvDim + c) = b;
    } else {
      const uint2 packed = make_uint2(pack_bf16x2(y[0], y[1]), pack_bf16x2(y[2], y[3]));
      const uint2 transient = kv == 1 && p.v_fp16
                                  ? make_uint2(pack_half2(y[0], y[1]), pack_half2(y[2], y[3]))
                                  : packed;
      *reinterpret_cast<uint2*>(static_cast<uint16_t*>(dst) + size_t(t) * KvDim + c) = transient;
      if (publish) {
        auto* ring = reinterpret_cast<uint16_t*>(kv == 0 ? p.k_ring : p.v_ring);
        *reinterpret_cast<uint2*>(ring + size_t(slot) * KvDim + c) = packed;
      }
    }
  }
}

// ---- rel_logits: BF16 Tensor Core MMA (m16n8k16) over [M = T*Heads, 16] x [16, 1024] -> [M, 1024]
// Keeps all 64 columns of bf16(rel_proj) in 16 registers per lane and writes coalesced uint4 stores.

__global__ void __launch_bounds__(128, 4)
rel_logits_mma_kernel(const uint16_t* __restrict__ rel, const float* __restrict__ rel_proj,
                      int M, uint16_t* __restrict__ out) {
  constexpr int BM = 64, BN = 64, R_ITERS = 4;
  __shared__ __align__(128) uint16_t sB[BN][kRel];
  __shared__ __align__(128) uint16_t sA[BM][kRel];
  __shared__ __align__(128) uint16_t sC[BM][BN];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int col0 = blockIdx.x * BN;
  const int row_group = blockIdx.y * (BM * R_ITERS);

  for (int i = tid; i < 256; i += 128) {
    const int j = i >> 4;
    const int c4 = (i & 15) * 4;
    const float4 v = *reinterpret_cast<const float4*>(rel_proj + j * kRelExtent + col0 + c4);
    sB[c4 + 0][j] = f2bf_bits(v.x);
    sB[c4 + 1][j] = f2bf_bits(v.y);
    sB[c4 + 2][j] = f2bf_bits(v.z);
    sB[c4 + 3][j] = f2bf_bits(v.w);
  }
  __syncthreads();

  uint32_t b_reg[8][2];
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    uint32_t r0, r1, r2, r3;
    const int br = ni * 16 + (lane & 15);
    const int bc = (lane >> 4) * 8;
    ldmatrix_x4(r0, r1, r2, r3, smem_addr(&sB[br][bc]));
    b_reg[2 * ni][0]     = r0; b_reg[2 * ni][1]     = r2;
    b_reg[2 * ni + 1][0] = r1; b_reg[2 * ni + 1][1] = r3;
  }

  const int grp = lane >> 2, t4 = lane & 3;
  for (int it = 0; it < R_ITERS; ++it) {
    const int m0 = row_group + it * BM;
    if (m0 >= M) break;
    const int r = tid >> 1, c8 = (tid & 1) * 8;
    if (m0 + r < M)
      *reinterpret_cast<uint4*>(&sA[r][c8]) = *reinterpret_cast<const uint4*>(rel + size_t(m0 + r) * kRel + c8);
    else
      *reinterpret_cast<uint4*>(&sA[r][c8]) = make_uint4(0, 0, 0, 0);
    __syncthreads();

    uint32_t a[4];
    ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(&sA[warp * 16 + (lane & 15)][(lane >> 4) * 8]));

    const int wr0 = warp * 16 + grp;
    const int wr1 = wr0 + 8;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      mma_bf16(d, a, b_reg[ni][0], b_reg[ni][1]);
      const int c = ni * 8 + t4 * 2;
      *reinterpret_cast<uint32_t*>(&sC[wr0][c]) = pack_bf16x2(d[0], d[1]);
      *reinterpret_cast<uint32_t*>(&sC[wr1][c]) = pack_bf16x2(d[2], d[3]);
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = tid + u * 128;
      const int sr = idx >> 3;
      const int sc = (idx & 7) * 8;
      if (m0 + sr < M) {
        *reinterpret_cast<uint4*>(out + size_t(m0 + sr) * kRelExtent + col0 + sc) =
            *reinterpret_cast<const uint4*>(&sC[sr][sc]);
      }
    }
    __syncthreads();
  }
}

// ---- static conv over time: CONV_ROWS consecutive rows, one or four channels per thread

constexpr int CONV_ROWS = 16;

template <int Channels>
__device__ __forceinline__ void load_conv_row(const uint16_t* x, int ld, const float* state, int C, int c,
                                              int src, float (&f)[Channels]) {
  if constexpr (Channels == 4) {
    if (src >= 0) {
      load_bf16x4(x + size_t(src) * ld + c, f);
    } else {
      const float4 v = *reinterpret_cast<const float4*>(state + (3 + src) * C + c);
      f[0] = v.x, f[1] = v.y, f[2] = v.z, f[3] = v.w;
    }
  } else {
    f[0] = src >= 0 ? bf2f(x[size_t(src) * ld + c]) : state[(3 + src) * C + c];
  }
}

template <int Channels>
__device__ __forceinline__ void store_conv_row(float* dst, const float (&f)[Channels]) {
  if constexpr (Channels == 4)
    *reinterpret_cast<float4*>(dst) = make_float4(f[0], f[1], f[2], f[3]);
  else
    *dst = f[0];
}

template <int Channels>
__device__ __forceinline__ void update_conv_rows(float* state, const uint16_t* x, int ld, int T, int C, int c) {
  for (int i = 0; i < 3; ++i) {  // old state is read from a later row than the one being written
    float f[Channels];
    load_conv_row(x, ld, state, C, c, T - 3 + i, f);
    store_conv_row(state + i * C + c, f);
  }
}

template <int Channels, bool Update>
__global__ void static_conv_kernel(const uint16_t* __restrict__ x, int ld, int T, int C, const float* __restrict__ w,
                                   std::conditional_t<Update, float, const float>* __restrict__ state,
                                   float* __restrict__ out) {
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * Channels;
  if (c >= C) return;
  const int t0 = blockIdx.y * CONV_ROWS, t1 = min(T, t0 + CONV_ROWS);
  float4 weights[Channels];
#pragma unroll
  for (int i = 0; i < Channels; ++i)
    weights[i] = *reinterpret_cast<const float4*>(w + size_t(c + i) * 4);
  float e0[Channels], e1[Channels], e2[Channels];
  load_conv_row(x, ld, state, C, c, t0 - 3, e0);
  load_conv_row(x, ld, state, C, c, t0 - 2, e1);
  load_conv_row(x, ld, state, C, c, t0 - 1, e2);
  const float inv = 1.f / 2.2360679774997896f;
  for (int t = t0; t < t1; ++t) {
    float e3[Channels], result[Channels];
    load_conv_row(x, ld, state, C, c, t, e3);
#pragma unroll
    for (int i = 0; i < Channels; ++i) {
      const float4 w4 = weights[i];
      float acc = __fmul_rn(w4.x, e0[i]);
      acc = __fadd_rn(acc, __fmul_rn(w4.y, e1[i]));
      acc = __fadd_rn(acc, __fmul_rn(w4.z, e2[i]));
      acc = __fadd_rn(acc, __fmul_rn(w4.w, e3[i]));
      result[i] = __fmul_rn(__fadd_rn(acc, e3[i]), inv);
      e0[i] = e1[i], e1[i] = e2[i], e2[i] = e3[i];
    }
    store_conv_row(out + size_t(t) * C + c, result);
  }
  if constexpr (Update) {
    static_assert(CONV_ROWS >= 3);
    // Only the first time tile needs the old state. Each thread owns its channels throughout.
    if (t0 == 0) update_conv_rows<Channels>(state, x, ld, T, C, c);
  }
}

template <int Channels>
__global__ void conv_state_kernel(float* state_k, const uint16_t* x_k, int ld, int T, int C,
                                   float* state_v, const uint16_t* x_v) {
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) * Channels;
  if (c >= C) return;
  if (blockIdx.y == 0) update_conv_rows<Channels>(state_k, x_k, ld, T, C, c);
  else update_conv_rows<Channels>(state_v, x_v, ld, T, C, c);
}

bool conv_vector_aligned(const __nv_bfloat16* x, int ld, int C, const float* state) {
  return C % 4 == 0 && ld % 4 == 0 && reinterpret_cast<uintptr_t>(x) % 8 == 0 &&
         reinterpret_cast<uintptr_t>(state) % 16 == 0;
}

__global__ void pack_rel_proj_kernel(const float* __restrict__ w, uint16_t* __restrict__ out) {
  constexpr int Distances = 32;
  __shared__ uint16_t weights[kRel][Distances + 1];
  const int d0 = blockIdx.x * Distances;
  const int tid = threadIdx.x;
  const int d = tid % Distances;
#pragma unroll
  for (int j = tid / Distances; j < kRel; j += 128 / Distances)
    weights[j][d] = f2bf_bits(w[j * kRelExtent + d0 + d]);
  __syncthreads();
  if (tid < Distances * 2) {
    const int col = tid / 2, half = tid % 2;
    uint4 v;
    v.x = uint32_t(weights[8 * half][col]) | (uint32_t(weights[8 * half + 1][col]) << 16);
    v.y = uint32_t(weights[8 * half + 2][col]) | (uint32_t(weights[8 * half + 3][col]) << 16);
    v.z = uint32_t(weights[8 * half + 4][col]) | (uint32_t(weights[8 * half + 5][col]) << 16);
    v.w = uint32_t(weights[8 * half + 6][col]) | (uint32_t(weights[8 * half + 7][col]) << 16);
    *reinterpret_cast<uint4*>(out + (d0 + col) * kRel + 8 * half) = v;
  }
}

// ---- message gates (H / 4 threads per row, float4 each)

constexpr int ROW_T = H / 4;  // 576
constexpr int ROW_W = ROW_T / 32;

template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N]) {
  __shared__ float red[ROW_W][N];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < N; ++i) red[warp][i] = v[i];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) {
    float s = 0.f;
    for (int w = 0; w < ROW_W; ++w) s += red[w][i];
    v[i] = s;
  }
}

template <bool InBf16, class RowsT>
__global__ void __launch_bounds__(ROW_T) gate_inputs_kernel(const RowsT rows, int T, uint16_t* __restrict__ gin) {
  const int t = blockIdx.x, g = blockIdx.y, c = threadIdx.x * 4;
  float hv[4], mv[4];
  if constexpr (InBf16) {
    load_bf16x4(reinterpret_cast<const uint16_t*>(rows.h[g]) + size_t(t) * H + c, hv);
    load_bf16x4(reinterpret_cast<const uint16_t*>(rows.msg[g]) + size_t(t) * H + c, mv);
  } else {
    const float4 h = *reinterpret_cast<const float4*>(rows.h[g] + size_t(t) * H + c);
    const float4 m = *reinterpret_cast<const float4*>(rows.msg[g] + size_t(t) * H + c);
    hv[0] = h.x, hv[1] = h.y, hv[2] = h.z, hv[3] = h.w;
    mv[0] = m.x, mv[1] = m.y, mv[2] = m.z, mv[3] = m.w;
  }
  float ss[2] = {hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2] + hv[3] * hv[3],
                 mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2] + mv[3] * mv[3]};
  block_sum(ss);
  const float rh = rsqrtf(ss[0] * (1.f / H) + kEps), rm = rsqrtf(ss[1] * (1.f / H) + kEps);
  uint16_t* o = gin + (size_t(g) * T + t) * 2 * H;
  store_bf16x4(o + c, {hv[0] * rh, hv[1] * rh, hv[2] * rh, hv[3] * rh});
  store_bf16x4(o + H + c, {mv[0] * rm, mv[1] * rm, mv[2] * rm, mv[3] * rm});
}

// Fused BF16 Tensor Core forget/gain projection + tanh gate + RMSNorm.
// Eliminates both [T, 64] x [H, 64]^T cuBLAS GEMMs and the 2x[T, H] fpre/gpre DRAM round-trip.
struct GateFgSmem {
  alignas(128) uint16_t sA[16][64];     // 2 KB: g16[t0..t0+15, 0..63]
  alignas(128) uint16_t sB[8][16][64];  // 16 KB: per-warp weight tile [16, 64]
  alignas(128) uint16_t sY[16][H];      // 72 KB: gated output y before RMSNorm
  alignas(16) float sRed[8][16];        // 512 B: per-warp row sum-of-squares
  alignas(16) float sRinv[16];          // 64 B: row inverse RMS
};
static_assert(sizeof(GateFgSmem) <= 99 * 1024);

__global__ void __launch_bounds__(256, 1)
gate_fg_apply_bf16_kernel(const uint16_t* __restrict__ g16, const uint16_t* __restrict__ w_forget,
                          const uint16_t* __restrict__ w_gain, const uint16_t* __restrict__ h,
                          const uint16_t* __restrict__ msg, const float* __restrict__ norm_w, int T,
                          uint16_t* __restrict__ out) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<GateFgSmem*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = lane >> 2, t4 = lane & 3;
  const int t0 = blockIdx.x * 16;

  // Load 16x64 BF16 g16 tile (128 uint4 loads -> first 128 threads).
  if (tid < 128) {
    const int r = tid >> 3, c8 = (tid & 7) * 8;
    if (t0 + r < T)
      *reinterpret_cast<uint4*>(&sm.sA[r][c8]) = *reinterpret_cast<const uint4*>(g16 + size_t(t0 + r) * 64 + c8);
    else
      *reinterpret_cast<uint4*>(&sm.sA[r][c8]) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  // Keep all 4 K-slices of g16 [16, 64] in registers across the entire H=2304 sweep.
  uint32_t a[4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    ldmatrix_x4(a[kk][0], a[kk][1], a[kk][2], a[kk][3],
                smem_addr(&sm.sA[lane & 15][kk * 16 + (lane >> 4) * 8]));
  }

  float ss0 = 0.f, ss1 = 0.f;
  const int r0 = grp, r1 = grp + 8;
  const bool v0 = (t0 + r0) < T, v1 = (t0 + r1) < T;
  const size_t h_off0 = size_t(t0 + r0) * H, h_off1 = size_t(t0 + r1) * H;

  // 8 warps x 16 cols = 128 cols/iter -> 18 iters cover H = 2304.
#pragma unroll 1
  for (int c_base = 0; c_base < H; c_base += 128) {
    const int c0 = c_base + warp * 16;
    float f0[4] = {0.f, 0.f, 0.f, 0.f}, f1[4] = {0.f, 0.f, 0.f, 0.f};
    float g0[4] = {0.f, 0.f, 0.f, 0.f}, g1[4] = {0.f, 0.f, 0.f, 0.f};

    // Load w_forget[c0..c0+15, 0..63] into warp-private sB[warp]
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int id = lane + u * 32;
      const int br = id >> 3, bc8 = (id & 7) * 8;
      *reinterpret_cast<uint4*>(&sm.sB[warp][br][bc8]) =
          __ldg(reinterpret_cast<const uint4*>(w_forget + size_t(c0 + br) * 64 + bc8));
    }
    __syncwarp();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, smem_addr(&sm.sB[warp][lane & 15][kk * 16 + (lane >> 4) * 8]));
      mma_bf16(f0, a[kk], b0, b2);
      mma_bf16(f1, a[kk], b1, b3);
    }

    // Load w_gain[c0..c0+15, 0..63] into warp-private sB[warp]
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int id = lane + u * 32;
      const int br = id >> 3, bc8 = (id & 7) * 8;
      *reinterpret_cast<uint4*>(&sm.sB[warp][br][bc8]) =
          __ldg(reinterpret_cast<const uint4*>(w_gain + size_t(c0 + br) * 64 + bc8));
    }
    __syncwarp();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, smem_addr(&sm.sB[warp][lane & 15][kk * 16 + (lane >> 4) * 8]));
      mma_bf16(g0, a[kk], b0, b2);
      mma_bf16(g1, a[kk], b1, b3);
    }

    // Apply gate: y = (1 - tanh(f)) * h + tanh(g) * msg, accumulate ss, stage y in sY
    auto apply_pair = [&](int col, const float (&fv)[4], const float (&gv)[4]) {
      if (v0) {
        const uint32_t hu = *reinterpret_cast<const uint32_t*>(h + h_off0 + col);
        const uint32_t mu = *reinterpret_cast<const uint32_t*>(msg + h_off0 + col);
        const float y0 = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[0])), bf_lo(hu)), __fmul_rn(tanhf(gv[0]), bf_lo(mu)));
        const float y1 = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[1])), bf_hi(hu)), __fmul_rn(tanhf(gv[1]), bf_hi(mu)));
        ss0 += y0 * y0 + y1 * y1;
        *reinterpret_cast<uint32_t*>(&sm.sY[r0][col]) = pack_bf16x2(y0, y1);
      }
      if (v1) {
        const uint32_t hu = *reinterpret_cast<const uint32_t*>(h + h_off1 + col);
        const uint32_t mu = *reinterpret_cast<const uint32_t*>(msg + h_off1 + col);
        const float y2 = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[2])), bf_lo(hu)), __fmul_rn(tanhf(gv[2]), bf_lo(mu)));
        const float y3 = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[3])), bf_hi(hu)), __fmul_rn(tanhf(gv[3]), bf_hi(mu)));
        ss1 += y2 * y2 + y3 * y3;
        *reinterpret_cast<uint32_t*>(&sm.sY[r1][col]) = pack_bf16x2(y2, y3);
      }
    };
    apply_pair(c0 + t4 * 2, f0, g0);
    apply_pair(c0 + 8 + t4 * 2, f1, g1);
  }

  // Reduce row sum-of-squares across the 4 lanes per row, then across the 8 warps.
  ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1);
  ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
  ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1);
  ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
  if (t4 == 0) {
    sm.sRed[warp][r0] = ss0;
    sm.sRed[warp][r1] = ss1;
  }
  __syncthreads();
  if (tid < 16) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += sm.sRed[w][tid];
    sm.sRinv[tid] = rsqrtf(s * (1.f / H) + kEps);
  }
  __syncthreads();

  // Stream normalized output to DRAM with coalesced uint2 (4xBF16) stores.
#pragma unroll
  for (int r = warp; r < 16; r += 8) {
    if (t0 + r >= T) continue;
    const float rinv = sm.sRinv[r];
    uint16_t* dst = out + size_t(t0 + r) * H;
    for (int q = lane; q < H / 4; q += 32) {
      const int c = q * 4;
      float y[4];
      load_bf16x4(&sm.sY[r][c], y);
      const float4 w4 = __ldg(reinterpret_cast<const float4*>(norm_w + c));
      store_bf16x4(dst + c, {y[0] * rinv * w4.x, y[1] * rinv * w4.y, y[2] * rinv * w4.z, y[3] * rinv * w4.w});
    }
  }
}

template <bool FgBf16>
__global__ void __launch_bounds__(ROW_T) gate_apply_kernel(const float* __restrict__ h, const float* __restrict__ msg,
                                                           const void* __restrict__ f, const void* __restrict__ gn,
                                                           const float* __restrict__ norm_w, uint16_t* __restrict__ out) {
  const size_t off = size_t(blockIdx.x) * H + threadIdx.x * 4;
  const float4 h4 = *reinterpret_cast<const float4*>(h + off), m4 = *reinterpret_cast<const float4*>(msg + off);
  const float hv[4] = {h4.x, h4.y, h4.z, h4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
  float fv[4], gv[4];
  if constexpr (FgBf16) {
    load_bf16x4(static_cast<const uint16_t*>(f) + off, fv);
    load_bf16x4(static_cast<const uint16_t*>(gn) + off, gv);
  } else {
    const float4 f4 = *reinterpret_cast<const float4*>(static_cast<const float*>(f) + off);
    const float4 g4 = *reinterpret_cast<const float4*>(static_cast<const float*>(gn) + off);
    fv[0] = f4.x, fv[1] = f4.y, fv[2] = f4.z, fv[3] = f4.w;
    gv[0] = g4.x, gv[1] = g4.y, gv[2] = g4.z, gv[3] = g4.w;
  }
  float y[4];
  float ss[1] = {0.f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // (1 - forget) * hidden + gain * message
    y[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[i])), hv[i]), __fmul_rn(tanhf(gv[i]), mv[i]));
    ss[0] += y[i] * y[i];
  }
  block_sum(ss);
  const float r = rsqrtf(ss[0] * (1.f / H) + kEps);
  const float4 w4 = *reinterpret_cast<const float4*>(norm_w + threadIdx.x * 4);
  store_bf16x4(out + off, {y[0] * r * w4.x, y[1] * r * w4.y, y[2] * r * w4.z, y[3] * r * w4.w});
}


// ---- misc

__global__ void widen_kernel(const int32_t* __restrict__ in, int T, int64_t* __restrict__ out, int32_t negative_as) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T) out[i] = in[i] < 0 ? negative_as : in[i];
}

template <bool Aligned>
__global__ void __launch_bounds__(ROW_T) put_rows_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst) {
  const size_t off = size_t(blockIdx.x) * H + threadIdx.x * 4;
  const uint16_t* s = src + off;
  float v[4];
  if constexpr (Aligned) load_bf16x4(s, v);
  else v[0] = bf2f(s[0]), v[1] = bf2f(s[1]), v[2] = bf2f(s[2]), v[3] = bf2f(s[3]);
  *reinterpret_cast<float4*>(dst + off) = make_float4(v[0], v[1], v[2], v[3]);
}

__global__ void __launch_bounds__(ROW_T) tap_kernel(const float* __restrict__ x, uint16_t* __restrict__ out,
                                                    int64_t stride) {
  const int t = blockIdx.x, c = threadIdx.x * 4;
  const float4 v = *reinterpret_cast<const float4*>(x + size_t(t) * H + c);
  store_bf16x4(out + size_t(t) * stride + c, {v.x, v.y, v.z, v.w});
}

// Warp per vocabulary row: dot in FP32 of bf16(w) (or bf16(fp8 * scale)) with x, BF16-rounded, softcapped.
template <bool FP8, bool Packed = false, bool Escapes = false>
__global__ void __launch_bounds__(256) head_kernel(const uint16_t* __restrict__ x, const void* __restrict__ w,
                                                   const float* __restrict__ scale, int V, float* __restrict__ out,
                                                   const __grid_constant__ pico12::DeviceView pico) {
  __shared__ float xs[H];
  for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = bf2f(x[i]);
  __syncthreads();
  const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (v >= V) return;
  float acc = 0.f;
  if constexpr (FP8) {
    const uint8_t* row = static_cast<const uint8_t*>(w) + size_t(v) * H;
    const float s = scale[v];
    for (int k = lane * 8; k < H; k += 256) {
      const uint2 u = __ldg(reinterpret_cast<const uint2*>(row + k));
      const uint32_t ws[2] = {u.x, u.y};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float2 a = e4m3x2_to_float2(ws[q] & 0xffff), b = e4m3x2_to_float2(ws[q] >> 16);
        acc = fmaf(round_bf16(a.x * s), xs[k + 4 * q], acc);
        acc = fmaf(round_bf16(a.y * s), xs[k + 4 * q + 1], acc);
        acc = fmaf(round_bf16(b.x * s), xs[k + 4 * q + 2], acc);
        acc = fmaf(round_bf16(b.y * s), xs[k + 4 * q + 3], acc);
      }
    }
  } else {
    constexpr size_t stride = Packed ? pico12::row_bytes(H) : H * sizeof(uint16_t);
    const auto* row = static_cast<const uint8_t*>(w) + size_t(v) * stride;
    for (int k = lane * 8; k < H; k += 256) {
      uint4 u;
      if constexpr (Packed) {
        const auto chunk = pico12::load8<l2::Evict::none>(row, H, k, 0);
        if constexpr (Escapes) u = pico12::restore8<false>(chunk, pico, v, k);
        else u = pico12::decode8(chunk, pico);
      } else {
        u = __ldg(reinterpret_cast<const uint4*>(row + k * sizeof(uint16_t)));
      }
      const uint32_t ws[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc = fmaf(bf_lo(ws[q]), xs[k + 2 * q], acc);
        acc = fmaf(bf_hi(ws[q]), xs[k + 2 * q + 1], acc);
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) out[v] = tanhf(round_bf16(acc) * (1.f / 30.f)) * 30.f;
}

}  // namespace

void attn_prep(const AttnPrepParams& p, cudaStream_t stream) {
  if (p.T <= 0 || !p.proj || !p.q || !p.rel || !p.k_out || !p.v_out || !p.conv_k || !p.conv_v)
    throw std::invalid_argument("attn_prep: missing argument");
  const bool ring = p.k_ring || p.v_ring || p.ring_pos;
  if (p.v_fp16 && (p.full || p.out_fp8 || p.out_q4 || p.out_kv12))
    throw std::invalid_argument("attn_prep: FP16 values need transient sliding rows");
  if (p.out_kv12 && (!p.full || p.out_fp8 || p.out_q4 || !p.k_scale_out || !p.v_scale_out))
    throw std::invalid_argument("attn_prep: KV12 needs full-attention K/V overflow destinations");
  if (p.out_q4 && (!p.full || p.out_fp8 || !p.k_scale_out || !p.v_scale_out))
    throw std::invalid_argument("attn_prep: Q4 needs full-attention K/V scale destinations");
  if (ring && (p.full || p.out_fp8 || !p.k_ring || !p.v_ring || !p.ring_pos))
    throw std::invalid_argument("attn_prep: ring needs all BF16 sliding cache destinations");
  auto dispatch = [&]<int KvHeads>() {
    if (p.out_kv12) attn_prep_kernel<KvHeads, false, false, true><<<p.T, 256, 0, stream>>>(p);
    else if (p.out_q4) attn_prep_kernel<KvHeads, false, true><<<p.T, 256, 0, stream>>>(p);
    else if (ring) attn_prep_kernel<KvHeads, true><<<p.T, 256, 0, stream>>>(p);
    else attn_prep_kernel<KvHeads, false><<<p.T, 256, 0, stream>>>(p);
  };
  if (p.hkv == kKvHeads) dispatch.template operator()<kKvHeads>();
  else if (p.hkv == kKvHeads / 2) dispatch.template operator()<kKvHeads / 2>();
  else throw std::invalid_argument("attn_prep: unsupported KV head count");
}

void rel_logits(const __nv_bfloat16* rel, const float* rel_proj, int T, __nv_bfloat16* out, cudaStream_t stream,
                int heads) {
  if (heads != kHeads && heads != kHeads / 2)
    throw std::invalid_argument("rel_logits: unsupported head count");
  const int M = T * heads;
  const dim3 grid(kRelExtent / 64, (M + 255) / 256);
  rel_logits_mma_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(rel), rel_proj, M, reinterpret_cast<uint16_t*>(out));
}

void pack_rel_proj(const float* rel_proj, __nv_bfloat16* out, cudaStream_t stream) {
  pack_rel_proj_kernel<<<kRelExtent / 32, 128, 0, stream>>>(
      rel_proj, reinterpret_cast<uint16_t*>(out));
}

void static_conv(const __nv_bfloat16* x, int ld, int T, int C, const float* w, const float* state, float* out,
                 cudaStream_t stream) {
  const dim3 grid((C + 127) / 128, (T + CONV_ROWS - 1) / CONV_ROWS);
  static_conv_kernel<1, false><<<grid, 128, 0, stream>>>(reinterpret_cast<const uint16_t*>(x), ld, T, C, w, state, out);
}

void static_conv_update(const __nv_bfloat16* x, int ld, int T, int C, const float* w, float* state, float* out,
                        cudaStream_t stream) {
  const auto* xb = reinterpret_cast<const uint16_t*>(x);
  const int tiles = (T + CONV_ROWS - 1) / CONV_ROWS;
  if (T >= CONV_ROWS && conv_vector_aligned(x, ld, C, state) && reinterpret_cast<uintptr_t>(out) % 16 == 0)
    static_conv_kernel<4, true><<<dim3((C + 511) / 512, tiles), 128, 0, stream>>>(xb, ld, T, C, w, state, out);
  else
    static_conv_kernel<1, true><<<dim3((C + 127) / 128, tiles), 128, 0, stream>>>(xb, ld, T, C, w, state, out);
}

void conv_state_update(float* state, const __nv_bfloat16* x, int ld, int T, int C, cudaStream_t stream) {
  conv_state_kernel<1><<<(C + 127) / 128, 128, 0, stream>>>(
      state, reinterpret_cast<const uint16_t*>(x), ld, T, C, nullptr, nullptr);
}

void kv_conv_state_update(float* state_k, float* state_v, const __nv_bfloat16* x_k, const __nv_bfloat16* x_v,
                          int ld, int T, int C, cudaStream_t stream) {
  const auto* k = reinterpret_cast<const uint16_t*>(x_k);
  const auto* v = reinterpret_cast<const uint16_t*>(x_v);
  if (conv_vector_aligned(x_k, ld, C, state_k) && conv_vector_aligned(x_v, ld, C, state_v))
    conv_state_kernel<4><<<dim3((C + 511) / 512, 2), 128, 0, stream>>>(state_k, k, ld, T, C, state_v, v);
  else
    conv_state_kernel<1><<<dim3((C + 127) / 128, 2), 128, 0, stream>>>(state_k, k, ld, T, C, state_v, v);
}

void gate_inputs(const GateRows& rows, int G, int T, __nv_bfloat16* gin, cudaStream_t stream) {
  gate_inputs_kernel<false><<<dim3(T, G), ROW_T, 0, stream>>>(rows, T, reinterpret_cast<uint16_t*>(gin));
}

void gate_inputs_bf16(const GateRowsBf16& rows, int G, int T, __nv_bfloat16* gin, cudaStream_t stream) {
  gate_inputs_kernel<true><<<dim3(T, G), ROW_T, 0, stream>>>(rows, T, reinterpret_cast<uint16_t*>(gin));
}

void gate_apply(const float* h, const float* msg, const float* f, const float* g, const float* norm_w, int T,
                __nv_bfloat16* out, cudaStream_t stream) {
  gate_apply_kernel<false><<<T, ROW_T, 0, stream>>>(h, msg, f, g, norm_w, reinterpret_cast<uint16_t*>(out));
}

void gate_apply_bf16(const float* h, const float* msg, const __nv_bfloat16* f, const __nv_bfloat16* g,
                     const float* norm_w, int T, __nv_bfloat16* out, cudaStream_t stream) {
  gate_apply_kernel<true><<<T, ROW_T, 0, stream>>>(h, msg, f, g, norm_w, reinterpret_cast<uint16_t*>(out));
}

void gate_fg_apply_bf16(const __nv_bfloat16* g16, const __nv_bfloat16* w_forget, const __nv_bfloat16* w_gain,
                        const __nv_bfloat16* h, const __nv_bfloat16* msg, const float* norm_w, int T,
                        __nv_bfloat16* out, cudaStream_t stream) {
  constexpr int shared_bytes = sizeof(GateFgSmem);
  static const cudaError_t attr =
      cudaFuncSetAttribute(gate_fg_apply_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (attr != cudaSuccess) throw std::runtime_error("gate_fg_apply_bf16: shared memory attribute failed");
  gate_fg_apply_bf16_kernel<<<(T + 15) / 16, 256, shared_bytes, stream>>>(
      reinterpret_cast<const uint16_t*>(g16), reinterpret_cast<const uint16_t*>(w_forget),
      reinterpret_cast<const uint16_t*>(w_gain), reinterpret_cast<const uint16_t*>(h),
      reinterpret_cast<const uint16_t*>(msg), norm_w, T, reinterpret_cast<uint16_t*>(out));
}


void widen_ids(const int32_t* in, int T, int64_t* out, int32_t negative_as, cudaStream_t stream) {
  widen_kernel<<<(T + 255) / 256, 256, 0, stream>>>(in, T, out, negative_as);
}

void put_rows(const __nv_bfloat16* src, int count, float* dst, cudaStream_t stream) {
  if (count <= 0) return;
  const auto* x = reinterpret_cast<const uint16_t*>(src);
  if (reinterpret_cast<uintptr_t>(x) % 8 == 0) put_rows_kernel<true><<<count, ROW_T, 0, stream>>>(x, dst);
  else put_rows_kernel<false><<<count, ROW_T, 0, stream>>>(x, dst);
}

void store_tap(const float* x, int T, __nv_bfloat16* out, int64_t out_stride, cudaStream_t stream) {
  tap_kernel<<<T, ROW_T, 0, stream>>>(x, reinterpret_cast<uint16_t*>(out), out_stride);
}

void head_logits(const __nv_bfloat16* x, const void* w, const float* scale, int V, float* out, cudaStream_t stream,
                  const pico12::Layout* packed) {
  const auto* xs = reinterpret_cast<const uint16_t*>(x);
  const int blocks = (V + 7) / 8;
  if (packed) {
    if (scale || packed->rows != V || packed->cols != H)
      throw std::invalid_argument("head_logits: invalid PICO12 matrix");
    const auto view = pico12::device_view(*packed, static_cast<const uint8_t*>(w));
    if (packed->escapes) head_kernel<false, true, true><<<blocks, 256, 0, stream>>>(xs, w, nullptr, V, out, view);
    else head_kernel<false, true><<<blocks, 256, 0, stream>>>(xs, w, nullptr, V, out, view);
  } else if (scale) {
    head_kernel<true><<<blocks, 256, 0, stream>>>(xs, w, scale, V, out, pico12::DeviceView{});
  } else {
    head_kernel<false><<<blocks, 256, 0, stream>>>(xs, w, nullptr, V, out, pico12::DeviceView{});
  }
}

}  // namespace ingine::kernels::prefill
