#include "model/batch_dflash2.h"
#include "model/tp_runtime.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/safetensors.h"
#include "kernels/batch/dense.h"
#include "kernels/batch/draft.h"
#include "kernels/batch/mx_gemm.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/gemm.h"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::model {
namespace {

using Bf = __nv_bfloat16;
namespace head = kernels::head;
namespace kbd = kernels::bdraft;

constexpr int H = 2304, I = 6912, Q = 4096, KV = 1024, W = BatchDFlash2::WINDOW;
constexpr int D = 128, QHeads = Q / D, KVHeads = KV / D, QPerKV = QHeads / KVHeads;
constexpr int Group = 16, Groups = H / Group, Coefficients = 4 * Groups;
constexpr int Threads = 128;
constexpr int Block = BatchDFlash2::BLOCK, Proposals = BatchDFlash2::GAMMA;
constexpr int Rank = 256, Vocabulary = 201024;
constexpr int NumLayers = 5;

constexpr int GqaTile = 32;
constexpr int GqaRows = Block * QPerKV;  // 8 * 4 = 32
constexpr int MaxChunks = 32;
constexpr int TargetChunks = 128;
static_assert(GqaRows == 32 && GqaTile == 32 && QPerKV == 4 && D == 128);

__device__ __forceinline__ std::uint8_t bf_to_e4m3(Bf x) {
  return __nv_fp8_e4m3(__bfloat162float(x)).__x;
}

__device__ __forceinline__ Bf e4m3_to_bf(std::uint8_t x) {
  __nv_fp8_e4m3 v;
  v.__x = x;
  return __float2bfloat16_rn(float(v));
}

template <typename Vector>
__device__ __forceinline__ bool is_aligned(const void* ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignof(Vector) == 0;
}

__device__ __forceinline__ unsigned int pack_bf16_pair(Bf low, Bf high) {
  return unsigned(__bfloat16_as_ushort(low)) | (unsigned(__bfloat16_as_ushort(high)) << 16);
}

__device__ __forceinline__ unsigned int normalize_pair(unsigned int x, unsigned int w, float scale) {
  const float low = __uint_as_float(x << 16) * scale * __uint_as_float(w << 16);
  const float high = __uint_as_float(x & 0xffff0000u) * scale * __uint_as_float(w & 0xffff0000u);
  return pack_bf16_pair(__float2bfloat16(low), __float2bfloat16(high));
}

__device__ __forceinline__ float rms_sum_squares(const Bf* values, int width, int lane) {
  float sum = 0.f;
  for (int i = lane; i < width; i += Threads) {
    const float value = __bfloat162float(values[i]);
    sum += value * value;
  }
  return sum;
}

__device__ __forceinline__ void rmsnorm_row(const Bf* x, const Bf* weight, int width, Bf* out,
                                           float* sums, uint4* row_cache) {
  const int lane = threadIdx.x;
  const bool vectorized = width % 8 == 0 && is_aligned<uint4>(x) && is_aligned<uint4>(weight) &&
                          is_aligned<uint4>(out);
  // Keep the scalar kernel's per-thread summation order and reduction tree.
  if (vectorized) {
    const auto* input = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < width / 8; i += Threads) row_cache[i] = input[i];
    __syncthreads();
    sums[lane] = rms_sum_squares(reinterpret_cast<const Bf*>(row_cache), width, lane);
  } else {
    sums[lane] = rms_sum_squares(x, width, lane);
  }
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sums[0] / width + 1e-6f);
  if (vectorized) {
    const auto* weights = reinterpret_cast<const uint4*>(weight);
    auto* output = reinterpret_cast<uint4*>(out);
    for (int i = lane; i < width / 8; i += Threads) {
      const uint4 v = row_cache[i];
      const uint4 w = weights[i];
      output[i] = make_uint4(normalize_pair(v.x, w.x, scale), normalize_pair(v.y, w.y, scale),
                             normalize_pair(v.z, w.z, scale), normalize_pair(v.w, w.w, scale));
    }
  } else {
    for (int i = lane; i < width; i += Threads)
      out[i] = __float2bfloat16(__bfloat162float(x[i]) * scale * __bfloat162float(weight[i]));
  }
}

// RMSNorm [rows, width], also used in place when committing context.
__global__ void rmsnorm_rows(const Bf* x, const Bf* weight, int width, Bf* out) {
  __shared__ float sums[Threads];
  extern __shared__ uint4 row_cache[];
  const std::size_t off = std::size_t(blockIdx.x) * width;
  rmsnorm_row(x + off, weight, width, out + off, sums, row_cache);
}

// Normalize the budgeted proposal rows of each block into contiguous [total_rows, H].
__global__ void final_norm_compact(const Bf* __restrict__ x, const Bf* __restrict__ weight,
                                   const std::int32_t* __restrict__ row_map,
                                   Bf* __restrict__ out) {
  __shared__ float sums[Threads];
  extern __shared__ uint4 row_cache[];
  const int row = blockIdx.x;
  const int src_row = row_map[2 * row] * Block + row_map[2 * row + 1] + 1;
  rmsnorm_row(x + std::size_t(src_row) * H, weight, H, out + std::size_t(row) * H, sums, row_cache);
}

// Commit scatter: kv [rows, 2*KV] (k | v) -> ring_k, ring_v at [slot, layer, pos % W, KV].
// Applies k_norm + RoPE to K, copies V; stores as FP8 E4M3 or BF16.
template <bool KV_FP8>
__global__ void commit_store_kernel(const Bf* __restrict__ kv, const Bf* __restrict__ knorm,
                                    const std::int32_t* __restrict__ row_slot,
                                    const std::int32_t* __restrict__ row_pos, int layer,
                                    void* __restrict__ k_rings, void* __restrict__ v_rings) {
  __shared__ float sums[Threads];
  const int row = blockIdx.x / KVHeads;
  const int head_idx = blockIdx.x % KVHeads;
  const int lane = threadIdx.x;  // 0..127 == D
  const int slot = row_slot[row];
  const int pos = row_pos[row];
  const int ring_row = pos % W;

  const Bf* k_in = kv + std::size_t(row) * (2 * KV) + head_idx * D;
  const Bf* v_in = kv + std::size_t(row) * (2 * KV) + KV + head_idx * D;

  const float kv_val = __bfloat162float(k_in[lane]);
  sums[lane] = kv_val * kv_val;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sums[0] / D + 1e-6f);
  const int other = (lane + D / 2) % D;
  const float value = __bfloat162float(__float2bfloat16(kv_val * scale * __bfloat162float(knorm[lane])));
  const float pair =
      __bfloat162float(__float2bfloat16(__bfloat162float(k_in[other]) * scale * __bfloat162float(knorm[other])));
  float sine, cosine;
  sincosf(pos * powf(10000.f, -2.f * (lane % (D / 2)) / D), &sine, &cosine);
  const Bf k_rot = __float2bfloat16(value * cosine + (lane < D / 2 ? -pair : pair) * sine);
  const Bf v_val = v_in[lane];

  const std::size_t dst_off = ((std::size_t(slot) * NumLayers + layer) * W + ring_row) * KV + head_idx * D + lane;
  if constexpr (KV_FP8) {
    static_cast<std::uint8_t*>(k_rings)[dst_off] = bf_to_e4m3(k_rot);
    static_cast<std::uint8_t*>(v_rings)[dst_off] = bf_to_e4m3(v_val);
  } else {
    static_cast<Bf*>(k_rings)[dst_off] = k_rot;
    static_cast<Bf*>(v_rings)[dst_off] = v_val;
  }
}

// Build [n * 8] int64 token ids for kernels::dense::embed: [anchor_b, mask*7] per sequence b.
__global__ void build_block_ids(const std::int32_t* __restrict__ anchors, std::int64_t* __restrict__ ids) {
  const int seq = blockIdx.x;
  const int r = threadIdx.x;  // 0..7
  ids[seq * Block + r] = r ? BatchDFlash2::MASK_TOKEN : anchors[seq];
}

__global__ void f32_to_bf16(const float* __restrict__ x, Bf* __restrict__ out, int size) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < size && is_aligned<float4>(x) && is_aligned<uint2>(out)) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    *reinterpret_cast<uint2*>(out + i) = make_uint2(
        pack_bf16_pair(__float2bfloat16(v.x), __float2bfloat16(v.y)),
        pack_bf16_pair(__float2bfloat16(v.z), __float2bfloat16(v.w)));
  } else {
    for (int j = 0; j < 4 && i + j < size; ++j) out[i + j] = __float2bfloat16(x[i + j]);
  }
}

// Batched 2-tap dynamic conv: resets shifted to 0 at every block boundary (row % 8 == 0).
__global__ void batch_convolution(const Bf* __restrict__ x, const Bf* __restrict__ coeff,
                                  const Bf* __restrict__ base, int part, Bf* __restrict__ out, bool residual) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= H) return;
  const int row_in_block = row % Block;
  coeff += std::size_t(row) * Coefficients + part * 2 * Groups + col / Group;
  base += part * 2 * H + col;
  const std::size_t index = std::size_t(row) * H + col;
  const Bf a = __hadd(base[0], coeff[0]);
  const Bf b = __hadd(base[H], coeff[Groups]);
  const Bf shifted = row_in_block ? x[index - H] : __float2bfloat16(0.f);
  const Bf value = __hadd_rn(__hmul_rn(a, x[index]), __hmul_rn(b, shifted));
  out[index] = residual ? __hadd(out[index], value) : value;
}

// QKV split + Q/K RMSNorm + RoPE for all n * 8 block rows.
// qkv: [M, Q + 2*KV]; grid: M * (QHeads + KVHeads), block: 128.
template <bool KV_FP8>
__global__ void batch_qkv_rope(const Bf* __restrict__ qkv, const std::int32_t* __restrict__ seq_pos,
                               const Bf* __restrict__ qnorm, const Bf* __restrict__ knorm,
                               Bf* __restrict__ q_out, void* __restrict__ bk_out, void* __restrict__ bv_out) {
  __shared__ float sums[Threads];
  constexpr int TotalHeads = QHeads + KVHeads;
  constexpr int QKVWidth = Q + 2 * KV;
  const int row = blockIdx.x / TotalHeads;
  const int h = blockIdx.x % TotalHeads;
  const int seq = row / Block;
  const int r = row % Block;
  const int pos = seq_pos[seq] + r;
  const int lane = threadIdx.x;  // 0..127

  const bool is_q = h < QHeads;
  const int kh = is_q ? h : (h - QHeads);
  const Bf* in = qkv + std::size_t(row) * QKVWidth + (is_q ? h * D : Q + kh * D);
  const Bf* norm_w = is_q ? qnorm : knorm;

  const float val = __bfloat162float(in[lane]);
  sums[lane] = val * val;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sums[0] / D + 1e-6f);
  const int other = (lane + D / 2) % D;
  const float normed = __bfloat162float(__float2bfloat16(val * scale * __bfloat162float(norm_w[lane])));
  const float pair =
      __bfloat162float(__float2bfloat16(__bfloat162float(in[other]) * scale * __bfloat162float(norm_w[other])));
  float sine, cosine;
  sincosf(pos * powf(10000.f, -2.f * (lane % (D / 2)) / D), &sine, &cosine);
  const Bf rot = __float2bfloat16(normed * cosine + (lane < D / 2 ? -pair : pair) * sine);

  if (is_q) {
    q_out[(std::size_t(row) * QHeads + h) * D + lane] = rot;
  } else {
    const Bf v_val = qkv[std::size_t(row) * QKVWidth + Q + KV + kh * D + lane];
    const std::size_t off = (std::size_t(row) * KVHeads + kh) * D + lane;
    if constexpr (KV_FP8) {
      static_cast<std::uint8_t*>(bk_out)[off] = bf_to_e4m3(rot);
      static_cast<std::uint8_t*>(bv_out)[off] = bf_to_e4m3(v_val);
    } else {
      static_cast<Bf*>(bk_out)[off] = rot;
      static_cast<Bf*>(bv_out)[off] = v_val;
    }
  }
}

__device__ __forceinline__ Bf swiglu_value(Bf gate, Bf up) {
  const float g = __bfloat162float(gate);
  return __hmul(__float2bfloat16(g / (1.f + expf(-g))), up);
}

__device__ __forceinline__ unsigned int swiglu_pair(unsigned int gate, unsigned int up) {
  const Bf low = swiglu_value(__ushort_as_bfloat16(gate & 0xffffu), __ushort_as_bfloat16(up & 0xffffu));
  const Bf high = swiglu_value(__ushort_as_bfloat16(gate >> 16), __ushort_as_bfloat16(up >> 16));
  return pack_bf16_pair(low, high);
}

// Batched SwiGLU on concatenated gate_up [M, 2*I] -> out [M, I].
__global__ void batch_swiglu(const Bf* __restrict__ gate_up, Bf* __restrict__ out, int total) {
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx >= total) return;
  if (I % 8 == 0 && idx + 7 < total && is_aligned<uint4>(gate_up) && is_aligned<uint4>(out)) {
    const int row = idx / I;
    const int col = idx % I;
    const Bf* gu = gate_up + std::size_t(row) * (2 * I) + col;
    const uint4 gate = *reinterpret_cast<const uint4*>(gu);
    const uint4 up = *reinterpret_cast<const uint4*>(gu + I);
    *reinterpret_cast<uint4*>(out + idx) = make_uint4(swiglu_pair(gate.x, up.x), swiglu_pair(gate.y, up.y),
                                                     swiglu_pair(gate.z, up.z), swiglu_pair(gate.w, up.w));
  } else {
    for (int j = 0; j < 8 && idx + j < total; ++j) {
      const int i = idx + j;
      const int row = i / I;
      const int col = i % I;
      const Bf* gu = gate_up + std::size_t(row) * (2 * I) + col;
      out[i] = swiglu_value(gu[0], gu[I]);
    }
  }
}

// Tensor-core GQA batched sliding-window + bidirectional block attention.
// Grid: (chunks, KVHeads, n), Block: 128 threads (4 warps).
// Static smem: qs(8KB) + kv(8KB) + scores(4KB) + probs(2KB) + scratch(4KB) + out_acc(16KB) + stats(0.4KB) = 42.4KB.
template <bool KV_FP8>
__global__ void __launch_bounds__(Threads)
    batch_gqa_attention(const Bf* __restrict__ query, const void* __restrict__ k_rings,
                        const void* __restrict__ v_rings, int layer, const void* __restrict__ block_k,
                        const void* __restrict__ block_v, const std::int32_t* __restrict__ d_slots,
                        const std::int32_t* __restrict__ d_pos, const std::int32_t* __restrict__ d_vbegin,
                        const std::int32_t* __restrict__ d_vend, int chunks, int chunk_keys,
                        float* __restrict__ partial_o, float* __restrict__ partial_ml) {
  using namespace nvcuda::wmma;
  __shared__ __align__(32) Bf qs[GqaRows][D];
  __shared__ __align__(32) Bf kv[GqaTile][D];
  __shared__ __align__(32) float scores[GqaRows][GqaTile];
  __shared__ __align__(32) Bf probs[GqaRows][GqaTile];
  __shared__ __align__(32) float scratch[4][16][16];
  __shared__ __align__(32) float out_acc[GqaRows][D];
  __shared__ float m_run[GqaRows], l_run[GqaRows], alpha_run[GqaRows];

  const int chunk = blockIdx.x;
  const int head = blockIdx.y;
  const int seq = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  const int slot = d_slots[seq];
  const int P = d_pos[seq];
  const int vbegin = d_vbegin[seq];
  const int vend = min(d_vend[seq], P);
  const int seq_begin = max(vbegin, P - W + 1);
  const int ring_count = max(0, vend - seq_begin);
  const int nkeys = ring_count + Block;
  const int k0 = chunk * chunk_keys;
  const int kend = min(k0 + chunk_keys, nkeys);
  if (k0 >= nkeys) return;

  // Load qs[32][128] once for this (seq, kv_head) and zero out_acc.
  for (int i = tid; i < GqaRows * (D / 8); i += Threads) {
    const int row = i / (D / 8), col = i % (D / 8);
    const int qrow = row / QPerKV;
    const int qh = head * QPerKV + (row % QPerKV);
    const std::size_t qidx = (std::size_t(seq) * Block + qrow) * QHeads + qh;
    reinterpret_cast<uint4*>(qs[row])[col] = reinterpret_cast<const uint4*>(query + qidx * D)[col];
    reinterpret_cast<float4*>(&out_acc[row][col * 8])[0] = make_float4(0.f, 0.f, 0.f, 0.f);
    reinterpret_cast<float4*>(&out_acc[row][col * 8])[1] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (tid < GqaRows) {
    m_run[tid] = -CUDART_INF_F;
    l_run[tid] = 0.f;
    alpha_run[tid] = 0.f;
  }
  __syncthreads();

  const std::size_t ring_base = (std::size_t(slot) * NumLayers + layer) * W * KV;
  const std::size_t blk_base = std::size_t(seq) * Block * KV;

  auto load_kv_tile = [&](const void* ring_ptr, const void* blk_ptr, int t0) {
    for (int i = tid; i < GqaTile * (D / 8); i += Threads) {
      const int row = i / (D / 8), col = i % (D / 8);
      const int k = t0 + row;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (k < kend) {
        const bool in_ring = k < ring_count;
        const std::size_t elem_off = in_ring
                                         ? (ring_base + std::size_t((seq_begin + k) % W) * KV + head * D)
                                         : (blk_base + std::size_t(k - ring_count) * KV + head * D);
        const void* src_base = in_ring ? ring_ptr : blk_ptr;
        if constexpr (KV_FP8) {
          const std::uint8_t* p8 = static_cast<const std::uint8_t*>(src_base) + elem_off + col * 8;
          const uint2 packed = *reinterpret_cast<const uint2*>(p8);
          std::uint8_t bytes[8];
          *reinterpret_cast<uint2*>(bytes) = packed;
          Bf out8[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) out8[e] = e4m3_to_bf(bytes[e]);
          v = *reinterpret_cast<const uint4*>(out8);
        } else {
          const Bf* p16 = static_cast<const Bf*>(src_base) + elem_off + col * 8;
          v = *reinterpret_cast<const uint4*>(p16);
        }
      }
      reinterpret_cast<uint4*>(kv[row])[col] = v;
    }
  };

  fragment<matrix_a, 16, 16, 16, Bf, row_major> a;
  fragment<matrix_b, 16, 16, 16, Bf, col_major> bk_frag;
  fragment<matrix_b, 16, 16, 16, Bf, row_major> bv_frag;
  fragment<accumulator, 16, 16, 16, float> c;

  for (int t0 = k0; t0 < kend; t0 += GqaTile) {
    load_kv_tile(k_rings, block_k, t0);
    __syncthreads();

    fill_fragment(c, 0.f);
    for (int d = 0; d < D; d += 16) {
      load_matrix_sync(a, &qs[16 * (warp / 2)][d], D);
      load_matrix_sync(bk_frag, &kv[16 * (warp % 2)][d], D);
      mma_sync(c, a, bk_frag, c);
    }
    store_matrix_sync(&scores[16 * (warp / 2)][16 * (warp % 2)], c, GqaTile, mem_row_major);
    __syncthreads();

    for (int qr = warp; qr < GqaRows; qr += 4) {
      const int qrow = qr / QPerKV;
      const int k = t0 + lane;
      const int q_min_pos = max(vbegin, P + qrow - W + 1);
      const bool valid = (k < kend) && (k >= ring_count || (seq_begin + k) >= q_min_pos);
      const float score = valid ? scores[qr][lane] * rsqrtf(float(D)) : -CUDART_INF_F;
      float tile_peak = score;
      for (int off = 16; off; off /= 2) tile_peak = fmaxf(tile_peak, __shfl_xor_sync(0xffffffffu, tile_peak, off));
      const float old_m = m_run[qr];
      const float new_m = fmaxf(old_m, tile_peak);
      const float carry = (old_m == -CUDART_INF_F || new_m == -CUDART_INF_F) ? 0.f : __expf(old_m - new_m);
      const float p = (valid && new_m > -CUDART_INF_F) ? __expf(score - new_m) : 0.f;
      probs[qr][lane] = __float2bfloat16(p);
      float tile_sum = p;
      for (int off = 16; off; off /= 2) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, off);
      if (lane == 0) {
        alpha_run[qr] = carry;
        m_run[qr] = new_m;
        l_run[qr] = l_run[qr] * carry + tile_sum;
      }
    }
    __syncthreads();

    load_kv_tile(v_rings, block_v, t0);
    __syncthreads();

    for (int d = (warp % 2) * 64; d < (warp % 2 + 1) * 64; d += 16) {
      fill_fragment(c, 0.f);
      for (int t = 0; t < GqaTile; t += 16) {
        load_matrix_sync(a, &probs[(warp / 2) * 16][t], GqaTile);
        load_matrix_sync(bv_frag, &kv[t][d], D);
        mma_sync(c, a, bv_frag, c);
      }
      store_matrix_sync(&scratch[warp][0][0], c, 16, mem_row_major);
      __syncwarp();
      for (int i = lane; i < 256; i += 32) {
        const int qr = (warp / 2) * 16 + i / 16;
        const int col = d + (i % 16);
        out_acc[qr][col] = out_acc[qr][col] * alpha_run[qr] + scratch[warp][i / 16][i % 16];
      }
      __syncwarp();
    }
    __syncthreads();
  }

  const std::size_t base = ((std::size_t(seq) * chunks + chunk) * KVHeads + head) * GqaRows;
  for (int i = tid; i < GqaRows * (D / 4); i += Threads) {
    const int qr = i / (D / 4), col4 = i % (D / 4);
    reinterpret_cast<float4*>(partial_o + (base + qr) * D)[col4] =
        reinterpret_cast<const float4*>(out_acc[qr])[col4];
  }
  if (tid < GqaRows) {
    partial_ml[(base + tid) * 2 + 0] = m_run[tid];
    partial_ml[(base + tid) * 2 + 1] = l_run[tid];
  }
}

// Merge attention chunks: Grid (Block * QHeads, n), Block 128.
__global__ void __launch_bounds__(Threads)
    batch_merge_attention(const float* __restrict__ partial_o, const float* __restrict__ partial_ml,
                          const std::int32_t* __restrict__ d_pos, const std::int32_t* __restrict__ d_vbegin,
                          const std::int32_t* __restrict__ d_vend, int chunks, int chunk_keys, Bf* __restrict__ out) {
  __shared__ float factors[MaxChunks];
  __shared__ float denominator;
  const int qidx = blockIdx.x;  // 0..Block*QHeads-1
  const int seq = blockIdx.y;
  const int d = threadIdx.x;    // 0..127

  const int qrow = qidx / QHeads;
  const int qh = qidx % QHeads;
  const int head = qh / QPerKV;
  const int qr = qrow * QPerKV + (qh % QPerKV);

  const int P = d_pos[seq];
  const int vbegin = d_vbegin[seq];
  const int vend = min(d_vend[seq], P);
  const int seq_begin = max(vbegin, P - W + 1);
  const int nkeys = max(0, vend - seq_begin) + Block;
  const int mine = (nkeys + chunk_keys - 1) / chunk_keys;

  if (d == 0) {
    float peak = -CUDART_INF_F;
    for (int c = 0; c < mine; ++c) {
      const std::size_t base = ((std::size_t(seq) * chunks + c) * KVHeads + head) * GqaRows + qr;
      peak = fmaxf(peak, partial_ml[base * 2 + 0]);
    }
    float sum = 0.f;
    for (int c = 0; c < mine; ++c) {
      const std::size_t base = ((std::size_t(seq) * chunks + c) * KVHeads + head) * GqaRows + qr;
      const float mc = partial_ml[base * 2 + 0];
      const float lc = partial_ml[base * 2 + 1];
      const float f = (mc == -CUDART_INF_F) ? 0.f : __expf(mc - peak);
      factors[c] = f;
      sum += f * lc;
    }
    denominator = sum;
  }
  __syncthreads();

  float num = 0.f;
  for (int c = 0; c < mine; ++c) {
    const std::size_t base = ((std::size_t(seq) * chunks + c) * KVHeads + head) * GqaRows + qr;
    num += factors[c] * partial_o[base * D + d];
  }
  const std::size_t out_row = std::size_t(seq) * Block + qrow;
  out[out_row * Q + qh * D + d] = __float2bfloat16(denominator > 0.f ? num / denominator : 0.f);
}

// Batched candidate selector edge scoring: exact match to kernels/draft/dflash2_selector.cu.
__global__ void batch_score_edges(const std::int32_t* __restrict__ ids, const float* __restrict__ unary,
                                  const Bf* __restrict__ projected, const Bf* __restrict__ predecessors,
                                  const Bf* __restrict__ successors, const std::int32_t* __restrict__ anchors,
                                  const std::int32_t* __restrict__ row_map,
                                  float* __restrict__ lattice) {
  const int row = blockIdx.x;
  const int seq = row_map[2 * row];
  const int edge = row_map[2 * row + 1];
  const int previous = blockIdx.y;
  const int next = blockIdx.z;
  const int lane = threadIdx.x;

  const int predecessor = (edge == 0) ? anchors[seq] : ids[(row - 1) * 16 + previous];
  const int candidate = ids[row * 16 + next];
  float dot = 0.f;
  const Bf* pred_row = predecessors + std::size_t(predecessor) * Rank;
  const Bf* succ_row = successors + std::size_t(candidate) * Rank;
  const Bf* proj_row = projected + std::size_t(row) * Rank;
  for (int r = lane; r < Rank; r += 32) {
    const auto weighted = __float2bfloat16_rn(__bfloat162float(pred_row[r]) * __bfloat162float(proj_row[r]));
    dot = fmaf(__bfloat162float(weighted), __bfloat162float(succ_row[r]), dot);
  }
  for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_down_sync(0xffffffffu, dot, offset);
  if (lane == 0) {
    const float score = unary[row * 16 + next] + __bfloat162float(__float2bfloat16_rn(dot));
    lattice[(row * 16 + previous) * 16 + next] = 30.0f * tanhf(score / 30.0f);
  }
}

__global__ void batch_walk(const std::int32_t* __restrict__ ids, const float* __restrict__ lattice,
                           const std::int32_t* __restrict__ prop_off, const std::int32_t* __restrict__ prop_cnt,
                           std::int32_t* __restrict__ tokens, float* __restrict__ confidence, bool cumulative) {
  const int seq = blockIdx.x;
  const int start = prop_off[seq];
  const int count = prop_cnt[seq];
  const int output = seq * Proposals;
  int previous = 0;
  float prod = 1.f;
  for (int edge = 0; edge < count; ++edge) {
    const int row = start + edge;
    const float* scores = lattice + (row * 16 + previous) * 16;
    int best = 0;
    for (int next = 1; next < 16; ++next) {
      if (scores[next] > scores[best]) best = next;
    }
    tokens[output + edge] = ids[row * 16 + best];
    float mass = 0.f;
    for (int next = 0; next < 16; ++next) mass += expf(scores[next] - scores[best]);
    const float c = 1.f / mass;
    prod *= c;
    confidence[output + edge] = cumulative ? prod : c;
    previous = best;
  }
  for (int edge = count; edge < Proposals; ++edge) {
    tokens[output + edge] = 0;
    confidence[output + edge] = 0.f;
  }
}

__global__ void pack_batch_candidates(const float* __restrict__ values, const std::int32_t* __restrict__ ids,
                                      int count, head::Candidate* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = {values[i], ids[i]};
}

__global__ void merge_batch_head_candidates(const head::Candidate* __restrict__ gathered, int rows,
                                            std::int32_t* __restrict__ ids, float* __restrict__ values) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int rank_stride = rows * 16;
  const auto* first = gathered + row * 16;
  const auto* second = first + rank_stride;
  int a = 0, b = 0;
  for (int i = 0; i < 16; ++i) {
    const auto x = first[a], y = second[b];
    const bool pick_first = x.val > y.val || (x.val == y.val && x.global_id < y.global_id);
    const auto cand = pick_first ? first[a++] : second[b++];
    ids[row * 16 + i] = cand.global_id;
    values[row * 16 + i] = cand.val;
  }
}

int chunk_target(int n) { return std::clamp((TargetChunks + n - 1) / n, 1, MaxChunks); }

struct LinearW {
  const Bf* bf16 = nullptr;
  kbd::Mx mx{};
  int rows = 0, cols = 0;
};

struct LayerW {
  const Bf *pre = nullptr, *post = nullptr, *qnorm = nullptr, *knorm = nullptr;
  const Bf *attn_conv_base = nullptr, *attn_conv_proj = nullptr;
  const Bf *mlp_conv_base = nullptr, *mlp_conv_proj = nullptr;
  const Bf* kv_commit = nullptr;  // [2 * KV, H] BF16 for commit
  LinearW qkv, o, gate_up, down;
};

}  // namespace

// =================================================================================================
// BatchDFlash2Head (TP=2 sharded NVFP4 head)
// =================================================================================================

struct BatchDFlash2Head::Impl {
  TpRuntime& tp;
  head::Nvfp4Matrix matrix;
  int max_seqs;
  cudaStream_t stream;
  std::vector<DeviceBuffer> storage;
  std::size_t bytes = 0;
  Bf* hidden = nullptr;
  void* head_ws = nullptr;
  float* top_vals = nullptr;
  std::int32_t* top_ids = nullptr;
  head::Candidate* local_cand = nullptr;
  head::Candidate* gathered_cand = nullptr;

  template <typename T = std::byte>
  T* alloc(std::size_t n) {
    storage.emplace_back(n * sizeof(T));
    bytes += storage.back().bytes;
    return static_cast<T*>(storage.back().p);
  }

  Impl(TpRuntime& tp_in, head::Nvfp4Matrix m, int max_s, cudaStream_t s)
      : tp(tp_in), matrix(m), max_seqs(max_s), stream(s) {
    if (!matrix.codes || !matrix.scales || matrix.rows < 16)
      throw std::invalid_argument("BatchDFlash2Head requires --head-nvfp4");
    const int mh_max = max_seqs * GAMMA;
    hidden = alloc<Bf>(std::size_t(mh_max) * H);
    head_ws = alloc<std::byte>(kernels::batch::head_topk16_nvfp4_workspace_bytes(mh_max, matrix.rows));
    top_vals = alloc<float>(std::size_t(mh_max) * 16);
    top_ids = alloc<std::int32_t>(std::size_t(mh_max) * 16);
    local_cand = alloc<head::Candidate>(std::size_t(mh_max) * 16);
    gathered_cand = alloc<head::Candidate>(std::size_t(2) * mh_max * 16);
  }
};

BatchDFlash2Head::BatchDFlash2Head(TpRuntime& tp, head::Nvfp4Matrix matrix, int max_seqs, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(tp, matrix, max_seqs, stream)) {}

BatchDFlash2Head::~BatchDFlash2Head() = default;

void BatchDFlash2Head::run(int n, const Bf* hidden_in) {
  if (n < 1 || n > impl_->max_seqs) throw std::invalid_argument("BatchDFlash2Head::run: n out of range");
  run_rows(n * GAMMA, hidden_in);
}

void BatchDFlash2Head::run_rows(int total_rows, const Bf* hidden_in) {
  auto& d = *impl_;
  if (total_rows < 1 || total_rows > d.max_seqs * GAMMA)
    throw std::invalid_argument("BatchDFlash2Head::run_rows: total_rows out of range");
  const std::size_t hidden_bytes = std::size_t(total_rows) * H * sizeof(Bf);
  Bf* input = hidden_in ? const_cast<Bf*>(hidden_in) : d.hidden;
  d.tp.exchange_draft_head(input, hidden_bytes, nullptr, d.stream);
  kernels::batch::head_topk16_nvfp4(input, H, d.matrix, total_rows, d.head_ws, d.top_vals, d.top_ids, d.stream);
  const int total_cands = total_rows * 16;
  pack_batch_candidates<<<(total_cands + 255) / 256, 256, 0, d.stream>>>(
      d.top_vals, d.top_ids, total_cands, d.local_cand);
  d.tp.exchange_draft_head(d.local_cand, std::size_t(total_cands) * sizeof(head::Candidate), d.gathered_cand,
                           d.stream);
  if (hidden_in) {
    merge_batch_head_candidates<<<(total_rows + 127) / 128, 128, 0, d.stream>>>(
        d.gathered_cand, total_rows, d.top_ids, d.top_vals);
  }
  CUDA_CHECK(cudaGetLastError());
}

const std::int32_t* BatchDFlash2Head::ids() const { return impl_->top_ids; }
const float* BatchDFlash2Head::values() const { return impl_->top_vals; }
std::size_t BatchDFlash2Head::device_bytes() const { return impl_->bytes; }

// =================================================================================================
// BatchDFlash2
// =================================================================================================

struct BatchDFlash2::Impl {
  cudaStream_t s;
  int slots, max_seqs;
  Precision precision;
  bool kv_fp8;
  bool cumulative = false;

  std::vector<DeviceBuffer> storage;
  std::size_t bytes = 0;

  kernels::draft::Bf16Gemm gemm;
  std::unique_ptr<kbd::MxGemm> mx_gemm;

  // Target embeddings and head
  const Bf* embed = nullptr;
  const float* embed_norm = nullptr;
  head::Nvfp4Matrix target_head{};
  BatchDFlash2Head* sharded_head = nullptr;

  // DFlash2 weights
  const Bf* fc_bf16 = nullptr;
  const Bf *hidden_norm = nullptr, *final_norm = nullptr;
  const Bf *selector_w = nullptr, *pred = nullptr, *succ = nullptr;
  std::array<LayerW, NumLayers> layers;

  // Per-slot KV rings: [slots, 5, W, KV] in FP8 E4M3 or BF16
  void* k_rings = nullptr;
  void* v_rings = nullptr;
  std::vector<int> valid_begin, valid_end;

  // Commit scratch
  Bf *ctx = nullptr, *commit_kv = nullptr;
  std::int32_t *d_commit_slot = nullptr, *d_commit_pos = nullptr;
  MappedHostBuffer pin_commit_slot, pin_commit_pos;

  // Propose scratch (M = max_seqs * 8, MH = max_seqs * 7)
  // Slot, anchor, position, valid bounds, proposal offset/count, then (sequence, step) per compacted row.
  static constexpr int ProposeFields = 7;
  std::int32_t* d_propose_params = nullptr;
  MappedHostBuffer pin_propose_params;
  cudaEvent_t propose_params_copied = nullptr;
  std::int64_t* ids = nullptr;
  float* embed_f32 = nullptr;
  Bf *x = nullptr, *normed = nullptr, *convolved = nullptr, *coeff = nullptr;
  Bf *qkv_buf = nullptr, *q = nullptr, *attn_out = nullptr, *projection = nullptr;
  Bf *gate_up_buf = nullptr, *act_buf = nullptr, *hidden_props = nullptr, *projected = nullptr;
  void *block_k = nullptr, *block_v = nullptr;
  kbd::Mx mx_act_h{}, mx_act_q{}, mx_act_i{};
  float *attn_partial_o = nullptr, *attn_partial_ml = nullptr;
  void* head_ws = nullptr;
  float *top_vals = nullptr, *lattice = nullptr, *conf = nullptr;
  std::int32_t *top_ids = nullptr, *tokens = nullptr;

  template <typename T = std::byte>
  T* alloc(std::size_t n) {
    storage.emplace_back(n * sizeof(T));
    bytes += storage.back().bytes;
    return static_cast<T*>(storage.back().p);
  }

  LinearW make_linear(const Bf* src_bf16, int rows, int cols) {
    LinearW lw;
    lw.rows = rows;
    lw.cols = cols;
    if (precision == Precision::MXFP8) {
      lw.mx.q = alloc<std::uint8_t>(std::size_t(rows) * cols);
      lw.mx.sf = alloc<std::uint8_t>(kbd::mx_sf_bytes(rows, cols));
      kbd::mx_quantize(src_bf16, rows, cols, lw.mx, s);
    } else {
      // Match solo DFlash2 default FP8-row quantized weight values via BF16 GEMM.
      DeviceBuffer q8(std::size_t(rows) * cols), sc(std::size_t(rows) * sizeof(float));
      kernels::dense::fp8_rows(src_bf16, rows, cols, q8.p, static_cast<float*>(sc.p), s);
      Bf* dst = alloc<Bf>(std::size_t(rows) * cols);
      kernels::dense::dequant_rows(q8.p, static_cast<const float*>(sc.p), rows, cols, dst, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      lw.bf16 = dst;
    }
    return lw;
  }

  void run_linear(const Bf* input, int rows, const LinearW& w, kbd::Mx act_mx, Bf* out) {
    if (precision == Precision::MXFP8) {
      kbd::mx_quantize(input, rows, w.cols, act_mx, s);
      mx_gemm->run(act_mx.q, act_mx.sf, w.mx.q, w.mx.sf, out, w.rows, rows, w.rows, w.cols, s);
    } else {
      gemm.run(input, rows, w.bf16, w.rows, w.cols, out, s);
    }
  }

  Impl(const std::filesystem::path& checkpoint, const Weights& w, int slots_in, int max_seqs_in,
       cudaStream_t stream, Precision prec, bool kv_fp8_in, const head::Nvfp4Matrix* head_override,
       BatchDFlash2Head* sharded)
      : s(stream),
        slots(slots_in),
        max_seqs(max_seqs_in),
        precision(prec),
        kv_fp8(kv_fp8_in),
        cumulative(std::getenv("INKLING_DFLASH2_CUMULATIVE_CONFIDENCE") != nullptr),
        embed(static_cast<const Bf*>(w.embed.data)),
        embed_norm(static_cast<const float*>(w.embed_norm.data)),
        sharded_head(sharded),
        valid_begin(std::size_t(slots_in), 0),
        valid_end(std::size_t(slots_in), 0) {
    if (slots < 1 || max_seqs < 1 || max_seqs > slots)
      throw std::invalid_argument("BatchDFlash2: invalid slots or max_seqs");
    if (!embed || !embed_norm) throw std::invalid_argument("BatchDFlash2: missing target embeddings");
    if (!sharded_head) {
      if (head_override && head_override->codes && head_override->scales) {
        target_head = *head_override;
      } else if (w.unembed_nvfp4.codes && w.unembed_nvfp4.scales) {
        target_head = w.unembed_nvfp4;
      } else {
        throw std::invalid_argument("BatchDFlash2 requires an NVFP4 target head (or BatchDFlash2Head)");
      }
      if (target_head.v_begin != 0 || target_head.rows < 16 || target_head.rows > Vocabulary)
        throw std::invalid_argument("BatchDFlash2 TP=1 requires the complete NVFP4 target head");
    }
    if (precision == Precision::MXFP8) mx_gemm = std::make_unique<kbd::MxGemm>();

    SafetensorsModel model(checkpoint);
    auto upload_tmp = [&](const std::string& name, std::vector<std::int64_t> shape, DeviceBuffer& buf) {
      const auto tensor = model.at(name);
      if (tensor->dtype != "BF16" || tensor->shape != shape)
        throw std::runtime_error("BatchDFlash2: unexpected tensor shape or dtype: " + name);
      buf = DeviceBuffer(tensor->bytes);
      const auto data = model.read(name);
      buf.copy_from_host(data.data(), data.size());
      return static_cast<Bf*>(buf.p);
    };
    auto load_perm = [&](const std::string& name, std::vector<std::int64_t> shape) {
      const auto tensor = model.at(name);
      if (tensor->dtype != "BF16" || tensor->shape != shape)
        throw std::runtime_error("BatchDFlash2: unexpected tensor shape or dtype: " + name);
      Bf* dst = alloc<Bf>(tensor->bytes / sizeof(Bf));
      const auto data = model.read(name);
      CUDA_CHECK(cudaMemcpyAsync(dst, data.data(), data.size(), cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      return static_cast<const Bf*>(dst);
    };

    fc_bf16 = load_perm("fc.weight", {H, 4 * H});
    hidden_norm = load_perm("hidden_norm.weight", {H});
    final_norm = load_perm("norm.weight", {H});
    selector_w = load_perm("candidate_selector.hidden_projection.weight", {Rank, H});
    pred = load_perm("candidate_selector.predecessor_codebook", {Vocabulary, Rank});
    succ = load_perm("candidate_selector.successor_codebook", {Vocabulary, Rank});

    for (int i = 0; i < NumLayers; ++i) {
      auto& l = layers[i];
      const std::string p = "layers." + std::to_string(i) + ".";
      l.pre = load_perm(p + "input_layernorm.weight", {H});
      l.post = load_perm(p + "post_attention_layernorm.weight", {H});
      l.qnorm = load_perm(p + "self_attn.q_norm.weight", {D});
      l.knorm = load_perm(p + "self_attn.k_norm.weight", {D});
      l.attn_conv_base = load_perm(p + "attention_conv.base_kernel", {2, 2, H});
      l.attn_conv_proj = load_perm(p + "attention_conv.kernel_projection.weight", {Coefficients, H});
      l.mlp_conv_base = load_perm(p + "mlp_conv.base_kernel", {2, 2, H});
      l.mlp_conv_proj = load_perm(p + "mlp_conv.kernel_projection.weight", {Coefficients, H});

      // Fused QKV [6144, H] and KV commit [2048, H]
      {
        DeviceBuffer fq, fk, fv;
        const Bf* q_tmp = upload_tmp(p + "self_attn.q_proj.weight", {Q, H}, fq);
        const Bf* k_tmp = upload_tmp(p + "self_attn.k_proj.weight", {KV, H}, fk);
        const Bf* v_tmp = upload_tmp(p + "self_attn.v_proj.weight", {KV, H}, fv);
        DeviceBuffer qkv_cat(std::size_t(Q + 2 * KV) * H * sizeof(Bf));
        auto* qp = static_cast<Bf*>(qkv_cat.p);
        CUDA_CHECK(cudaMemcpyAsync(qp, q_tmp, std::size_t(Q) * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(qp + std::size_t(Q) * H, k_tmp, std::size_t(KV) * H * sizeof(Bf),
                                   cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(qp + std::size_t(Q + KV) * H, v_tmp, std::size_t(KV) * H * sizeof(Bf),
                                   cudaMemcpyDeviceToDevice, s));
        l.qkv = make_linear(qp, Q + 2 * KV, H);

        // Commit K/V uses FP8-row quantized & dequantized mirror, matching solo DFlash2 commit!
        DeviceBuffer q8(std::size_t(2 * KV) * H), sc(std::size_t(2 * KV) * sizeof(float));
        kernels::dense::fp8_rows(qp + std::size_t(Q) * H, 2 * KV, H, q8.p, static_cast<float*>(sc.p), s);
        Bf* kv_c = alloc<Bf>(std::size_t(2 * KV) * H);
        kernels::dense::dequant_rows(q8.p, static_cast<const float*>(sc.p), 2 * KV, H, kv_c, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        l.kv_commit = kv_c;
      }
      {
        DeviceBuffer fo;
        const Bf* o_tmp = upload_tmp(p + "self_attn.o_proj.weight", {H, Q}, fo);
        l.o = make_linear(o_tmp, H, Q);
      }
      {
        DeviceBuffer fg, fu;
        const Bf* g_tmp = upload_tmp(p + "mlp.gate_proj.weight", {I, H}, fg);
        const Bf* u_tmp = upload_tmp(p + "mlp.up_proj.weight", {I, H}, fu);
        DeviceBuffer gu_cat(std::size_t(2 * I) * H * sizeof(Bf));
        auto* gp = static_cast<Bf*>(gu_cat.p);
        CUDA_CHECK(cudaMemcpyAsync(gp, g_tmp, std::size_t(I) * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(gp + std::size_t(I) * H, u_tmp, std::size_t(I) * H * sizeof(Bf),
                                   cudaMemcpyDeviceToDevice, s));
        l.gate_up = make_linear(gp, 2 * I, H);
      }
      {
        DeviceBuffer fd;
        const Bf* d_tmp = upload_tmp(p + "mlp.down_proj.weight", {H, I}, fd);
        l.down = make_linear(d_tmp, H, I);
      }
    }

    // Per-slot KV rings: [slots, 5, W, KV]
    const std::size_t es = kv_fp8 ? 1 : 2;
    const std::size_t ring_bytes = std::size_t(slots) * NumLayers * W * KV * es;
    k_rings = alloc<std::byte>(ring_bytes);
    v_rings = alloc<std::byte>(ring_bytes);
    CUDA_CHECK(cudaMemsetAsync(k_rings, 0, ring_bytes, s));
    CUDA_CHECK(cudaMemsetAsync(v_rings, 0, ring_bytes, s));

    // Commit scratch
    ctx = alloc<Bf>(std::size_t(COMMIT_PIECE) * H);
    commit_kv = alloc<Bf>(std::size_t(COMMIT_PIECE) * 2 * KV);
    d_commit_slot = alloc<std::int32_t>(COMMIT_PIECE);
    d_commit_pos = alloc<std::int32_t>(COMMIT_PIECE);
    pin_commit_slot = MappedHostBuffer(std::size_t(COMMIT_PIECE) * sizeof(std::int32_t));
    pin_commit_pos = MappedHostBuffer(std::size_t(COMMIT_PIECE) * sizeof(std::int32_t));

    // Propose scratch
    const int M = max_seqs * Block;
    const int MH = max_seqs * Proposals;
    const std::size_t propose_ints = std::size_t(ProposeFields) * max_seqs + std::size_t(2) * MH;
    d_propose_params = alloc<std::int32_t>(propose_ints);
    pin_propose_params = MappedHostBuffer(propose_ints * sizeof(std::int32_t));
    ids = alloc<std::int64_t>(M);
    embed_f32 = alloc<float>(std::size_t(M) * H);
    x = alloc<Bf>(std::size_t(M) * H);
    normed = alloc<Bf>(std::size_t(M) * H);
    convolved = alloc<Bf>(std::size_t(M) * H);
    coeff = alloc<Bf>(std::size_t(M) * Coefficients);
    qkv_buf = alloc<Bf>(std::size_t(M) * (Q + 2 * KV));
    q = alloc<Bf>(std::size_t(M) * Q);
    block_k = alloc<std::byte>(std::size_t(M) * KV * es);
    block_v = alloc<std::byte>(std::size_t(M) * KV * es);
    attn_out = alloc<Bf>(std::size_t(M) * Q);
    projection = alloc<Bf>(std::size_t(M) * H);
    gate_up_buf = alloc<Bf>(std::size_t(M) * 2 * I);
    act_buf = alloc<Bf>(std::size_t(M) * I);
    hidden_props = alloc<Bf>(std::size_t(MH) * H);
    projected = alloc<Bf>(std::size_t(MH) * Rank);

    if (precision == Precision::MXFP8) {
      mx_act_h.q = alloc<std::uint8_t>(std::size_t(M) * H);
      mx_act_h.sf = alloc<std::uint8_t>(kbd::mx_sf_bytes(M, H));
      mx_act_q.q = alloc<std::uint8_t>(std::size_t(M) * Q);
      mx_act_q.sf = alloc<std::uint8_t>(kbd::mx_sf_bytes(M, Q));
      mx_act_i.q = alloc<std::uint8_t>(std::size_t(M) * I);
      mx_act_i.sf = alloc<std::uint8_t>(kbd::mx_sf_bytes(M, I));
    }

    std::size_t max_seq_chunks = 0;
    for (int n = 1; n <= max_seqs; ++n) max_seq_chunks = std::max(max_seq_chunks, std::size_t(n) * chunk_target(n));
    attn_partial_o = alloc<float>(max_seq_chunks * KVHeads * GqaRows * D);
    attn_partial_ml = alloc<float>(max_seq_chunks * KVHeads * GqaRows * 2);

    if (!sharded_head) {
      head_ws = alloc<std::byte>(kernels::batch::head_topk16_nvfp4_workspace_bytes(MH, target_head.rows));
      top_vals = alloc<float>(std::size_t(MH) * 16);
      top_ids = alloc<std::int32_t>(std::size_t(MH) * 16);
    }
    lattice = alloc<float>(std::size_t(MH) * 16 * 16);
    tokens = alloc<std::int32_t>(MH);
    conf = alloc<float>(MH);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaEventCreateWithFlags(&propose_params_copied, cudaEventDisableTiming));
  }

  ~Impl() {
    cudaEventSynchronize(propose_params_copied);
    cudaEventDestroy(propose_params_copied);
  }

  void reset_slot(int slot) {
    if (slot < 0 || slot >= slots) throw std::invalid_argument("BatchDFlash2::reset_slot: slot out of range");
    valid_begin[std::size_t(slot)] = 0;
    valid_end[std::size_t(slot)] = 0;
  }

  void clone_slot(int src_slot, int dst_slot, int len) {
    if (src_slot < 0 || src_slot >= slots || dst_slot < 0 || dst_slot >= slots || len < 0)
      throw std::invalid_argument("BatchDFlash2::clone_slot: invalid args");
    const int src_begin = valid_begin[std::size_t(src_slot)];
    const int src_end = valid_end[std::size_t(src_slot)];
    const int dst_end = std::min(src_end, len);
    if (src_slot == dst_slot) {
      valid_end[std::size_t(dst_slot)] = dst_end;
      if (dst_end <= src_begin) valid_begin[std::size_t(dst_slot)] = 0, valid_end[std::size_t(dst_slot)] = 0;
      return;
    }
    if (dst_end <= src_begin || dst_end == 0) {
      valid_begin[std::size_t(dst_slot)] = 0;
      valid_end[std::size_t(dst_slot)] = 0;
      return;
    }
    const std::size_t es = kv_fp8 ? 1 : 2;
    const std::size_t layer_bytes = std::size_t(W) * KV * es;
    const std::size_t slot_bytes = std::size_t(NumLayers) * layer_bytes;
    auto* kb = static_cast<std::byte*>(k_rings);
    auto* vb = static_cast<std::byte*>(v_rings);
    if (src_end >= W || src_begin > 0) {
      // Ring may have wrapped or start at non-zero offset: copy all 5 layers in two contiguous D2D copies.
      CUDA_CHECK(cudaMemcpyAsync(kb + std::size_t(dst_slot) * slot_bytes, kb + std::size_t(src_slot) * slot_bytes,
                                 slot_bytes, cudaMemcpyDeviceToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(vb + std::size_t(dst_slot) * slot_bytes, vb + std::size_t(src_slot) * slot_bytes,
                                 slot_bytes, cudaMemcpyDeviceToDevice, s));
    } else {
      // Unwrapped prefix [0, dst_end): copy only valid rows per layer.
      const std::size_t copy_bytes = std::size_t(dst_end) * KV * es;
      for (int l = 0; l < NumLayers; ++l) {
        const std::size_t src_off = std::size_t(src_slot) * slot_bytes + std::size_t(l) * layer_bytes;
        const std::size_t dst_off = std::size_t(dst_slot) * slot_bytes + std::size_t(l) * layer_bytes;
        CUDA_CHECK(cudaMemcpyAsync(kb + dst_off, kb + src_off, copy_bytes, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(vb + dst_off, vb + src_off, copy_bytes, cudaMemcpyDeviceToDevice, s));
      }
    }
    valid_begin[std::size_t(dst_slot)] = src_begin;
    valid_end[std::size_t(dst_slot)] = dst_end;
  }

  void commit(int n, const std::int32_t* slot_ids, const std::int64_t* start, const std::int32_t* count,
              const Bf* taps) {
    if (n <= 0) return;
    if (n > slots) throw std::invalid_argument("BatchDFlash2::commit: n exceeds slots");

    // Fast skip for single-sequence oversized prefill commit (> W rows).
    std::int32_t single_slot = 0, single_count = 0;
    std::int64_t single_start = 0;
    if (n == 1 && count[0] > W) {
      const int drop = count[0] - W;
      single_slot = slot_ids[0];
      single_start = start[0] + drop;
      single_count = W;
      taps += std::size_t(drop) * 4 * H;
      // Still update logical valid range starting from original start[0]
      const int sl = single_slot;
      if (valid_end[sl] == 0 || start[0] != valid_end[sl]) valid_begin[sl] = int(start[0]);
      valid_end[sl] = int(start[0] + count[0]);
      slot_ids = &single_slot;
      start = &single_start;
      count = &single_count;
    } else {
      for (int i = 0; i < n; ++i) {
        if (count[i] < 0 || slot_ids[i] < 0 || slot_ids[i] >= slots || start[i] < 0)
          throw std::invalid_argument("BatchDFlash2::commit: invalid slot/start/count");
        if (count[i] == 0) continue;
        const int sl = slot_ids[i];
        if (valid_end[sl] == 0 || int(start[i]) != valid_end[sl]) valid_begin[sl] = int(start[i]);
        valid_end[sl] = int(start[i] + count[i]);
      }
    }

    int total = 0;
    for (int i = 0; i < n; ++i) total += count[i];
    if (total == 0) return;

    std::vector<std::int32_t> all_slot(static_cast<std::size_t>(total));
    std::vector<std::int32_t> all_pos(static_cast<std::size_t>(total));
    int cursor = 0;
    for (int i = 0; i < n; ++i) {
      for (int r = 0; r < count[i]; ++r) {
        all_slot[std::size_t(cursor)] = slot_ids[i];
        all_pos[std::size_t(cursor)] = std::int32_t(start[i] + r);
        ++cursor;
      }
    }

    for (int r0 = 0; r0 < total; r0 += COMMIT_PIECE) {
      const int rows = std::min(COMMIT_PIECE, total - r0);
      // Pageable <= 8 KB source: CUDA stages synchronously before returning, safe for back-to-back async commits.
      CUDA_CHECK(cudaMemcpyAsync(d_commit_slot, all_slot.data() + r0, std::size_t(rows) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(d_commit_pos, all_pos.data() + r0, std::size_t(rows) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, s));

      gemm.run(taps + std::size_t(r0) * 4 * H, rows, fc_bf16, H, 4 * H, ctx, s);
      rmsnorm_rows<<<rows, Threads, H * sizeof(Bf), s>>>(ctx, hidden_norm, H, ctx);
      for (int l = 0; l < NumLayers; ++l) {
        gemm.run(ctx, rows, layers[l].kv_commit, 2 * KV, H, commit_kv, s);
        if (kv_fp8) {
          commit_store_kernel<true><<<rows * KVHeads, Threads, 0, s>>>(
              commit_kv, layers[l].knorm, d_commit_slot, d_commit_pos, l, k_rings, v_rings);
        } else {
          commit_store_kernel<false><<<rows * KVHeads, Threads, 0, s>>>(
              commit_kv, layers[l].knorm, d_commit_slot, d_commit_pos, l, k_rings, v_rings);
        }
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }

  Drafts propose(int n, const std::int32_t* slot_ids, const std::int32_t* anchors, const std::int64_t* pos,
                 const std::int32_t* max_drafts) {
    if (n < 1 || n > max_seqs) throw std::invalid_argument("BatchDFlash2::propose: n out of range");
    // Callers may enqueue back-to-back proposals with different row maps.
    CUDA_CHECK(cudaEventSynchronize(propose_params_copied));
    auto* p = static_cast<std::int32_t*>(pin_propose_params.data());
    auto* row_map = p + ProposeFields * n;
    int total_rows = 0;
    int max_keys = Block;
    for (int i = 0; i < n; ++i) {
      const int sl = slot_ids[i];
      if (sl < 0 || sl >= slots || anchors[i] < 0 || anchors[i] >= Vocabulary || pos[i] < 0)
        throw std::invalid_argument("BatchDFlash2::propose: invalid slot/anchor/pos");
      const int P = int(pos[i]);
      const int vb = valid_begin[std::size_t(sl)];
      const int ve = std::min(valid_end[std::size_t(sl)], P);
      const int ring_count = std::max(0, ve - std::max(vb, P - W + 1));
      max_keys = std::max(max_keys, ring_count + Block);
      const int count = max_drafts ? std::clamp(int(max_drafts[i]), 1, Proposals) : Proposals;
      p[i] = sl;
      p[n + i] = anchors[i];
      p[2 * n + i] = P;
      p[3 * n + i] = vb;
      p[4 * n + i] = ve;
      p[5 * n + i] = total_rows;
      p[6 * n + i] = count;
      for (int step = 0; step < count; ++step) {
        row_map[2 * (total_rows + step)] = i;
        row_map[2 * (total_rows + step) + 1] = step;
      }
      total_rows += count;
    }
    const std::size_t param_bytes = std::size_t(ProposeFields * n + 2 * total_rows) * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(d_propose_params, p, param_bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaEventRecord(propose_params_copied, s));
    const std::int32_t* d_slots = d_propose_params;
    const std::int32_t* d_anchors = d_propose_params + n;
    const std::int32_t* d_pos = d_propose_params + 2 * n;
    const std::int32_t* d_vbegin = d_propose_params + 3 * n;
    const std::int32_t* d_vend = d_propose_params + 4 * n;
    const std::int32_t* d_prop_off = d_propose_params + 5 * n;
    const std::int32_t* d_prop_cnt = d_propose_params + 6 * n;
    const std::int32_t* d_row_map = d_propose_params + ProposeFields * n;

    const int M = n * Block;

    // 1. Embeddings: bit-exact with solo DFlash2 via kernels::dense::embed
    build_block_ids<<<n, Block, 0, s>>>(d_anchors, ids);
    kernels::dense::embed(ids, M, embed, embed_norm, embed_f32, false, s);
    f32_to_bf16<<<(M * H + 4 * Threads - 1) / (4 * Threads), Threads, 0, s>>>(embed_f32, x, M * H);

    // Attention chunking
    const int target = chunk_target(n);
    const int chunk_keys = std::max(GqaTile, ((max_keys + target - 1) / target + GqaTile - 1) / GqaTile * GqaTile);
    const int chunks = (max_keys + chunk_keys - 1) / chunk_keys;

    // 2. Five DFlash2 layers batched across all M = n * 8 rows
    const dim3 conv_grid((H + Threads - 1) / Threads, M);
    for (int l = 0; l < NumLayers; ++l) {
      const LayerW& L = layers[l];

      // Pre-norm + attention conv (part 0)
      rmsnorm_rows<<<M, Threads, H * sizeof(Bf), s>>>(x, L.pre, H, normed);
      gemm.run(normed, M, L.attn_conv_proj, Coefficients, H, coeff, s);
      batch_convolution<<<conv_grid, Threads, 0, s>>>(normed, coeff, L.attn_conv_base, 0, convolved, false);

      // Fused QKV projection + Q/K norm + RoPE
      run_linear(convolved, M, L.qkv, mx_act_h, qkv_buf);
      if (kv_fp8) {
        batch_qkv_rope<true><<<M * (QHeads + KVHeads), Threads, 0, s>>>(
            qkv_buf, d_pos, L.qnorm, L.knorm, q, block_k, block_v);
        batch_gqa_attention<true><<<dim3(chunks, KVHeads, n), Threads, 0, s>>>(
            q, k_rings, v_rings, l, block_k, block_v, d_slots, d_pos, d_vbegin, d_vend, chunks, chunk_keys,
            attn_partial_o, attn_partial_ml);
      } else {
        batch_qkv_rope<false><<<M * (QHeads + KVHeads), Threads, 0, s>>>(
            qkv_buf, d_pos, L.qnorm, L.knorm, q, block_k, block_v);
        batch_gqa_attention<false><<<dim3(chunks, KVHeads, n), Threads, 0, s>>>(
            q, k_rings, v_rings, l, block_k, block_v, d_slots, d_pos, d_vbegin, d_vend, chunks, chunk_keys,
            attn_partial_o, attn_partial_ml);
      }
      batch_merge_attention<<<dim3(Block * QHeads, n), Threads, 0, s>>>(
          attn_partial_o, attn_partial_ml, d_pos, d_vbegin, d_vend, chunks, chunk_keys, attn_out);

      // O projection + attention conv residual (part 1)
      run_linear(attn_out, M, L.o, mx_act_q, projection);
      batch_convolution<<<conv_grid, Threads, 0, s>>>(projection, coeff, L.attn_conv_base, 1, x, true);

      // Post-norm + MLP conv (part 0) + fused gate_up + SwiGLU + down + MLP conv residual (part 1)
      rmsnorm_rows<<<M, Threads, H * sizeof(Bf), s>>>(x, L.post, H, normed);
      gemm.run(normed, M, L.mlp_conv_proj, Coefficients, H, coeff, s);
      batch_convolution<<<conv_grid, Threads, 0, s>>>(normed, coeff, L.mlp_conv_base, 0, convolved, false);

      run_linear(convolved, M, L.gate_up, mx_act_h, gate_up_buf);
      batch_swiglu<<<(M * I + 8 * Threads - 1) / (8 * Threads), Threads, 0, s>>>(gate_up_buf, act_buf, M * I);
      run_linear(act_buf, M, L.down, mx_act_i, projection);
      batch_convolution<<<conv_grid, Threads, 0, s>>>(projection, coeff, L.mlp_conv_base, 1, x, true);
    }

    // 3. Final norm + compact proposal rows into hidden_props [total_rows, H]
    final_norm_compact<<<total_rows, Threads, H * sizeof(Bf), s>>>(x, final_norm, d_row_map, hidden_props);

    // 4. Target NVFP4 head + top-16 (TP=1 or TP=2 sharded)
    const std::int32_t* cand_ids = nullptr;
    const float* cand_vals = nullptr;
    if (sharded_head) {
      sharded_head->run_rows(total_rows, hidden_props);
      cand_ids = sharded_head->ids();
      cand_vals = sharded_head->values();
    } else {
      kernels::batch::head_topk16_nvfp4(hidden_props, H, target_head, total_rows, head_ws, top_vals, top_ids, s);
      cand_ids = top_ids;
      cand_vals = top_vals;
    }

    // 5. Batched DFlash2 path selector
    gemm.run(hidden_props, total_rows, selector_w, Rank, H, projected, s);
    batch_score_edges<<<dim3(total_rows, 16, 16), 32, 0, s>>>(
        cand_ids, cand_vals, projected, pred, succ, d_anchors, d_row_map, lattice);
    batch_walk<<<n, 1, 0, s>>>(cand_ids, lattice, d_prop_off, d_prop_cnt, tokens, conf, cumulative);
    CUDA_CHECK(cudaGetLastError());
    return {tokens, conf};
  }
};

BatchDFlash2::BatchDFlash2(const std::filesystem::path& checkpoint, const Weights& w, int slots, int max_seqs,
                           cudaStream_t stream, Precision precision, bool kv_fp8,
                           const head::Nvfp4Matrix* head_override, BatchDFlash2Head* sharded_head)
    : impl_(std::make_unique<Impl>(checkpoint, w, slots, max_seqs, stream, precision, kv_fp8, head_override,
                                   sharded_head)) {}

BatchDFlash2::~BatchDFlash2() = default;

void BatchDFlash2::reset_slot(int slot) { impl_->reset_slot(slot); }
void BatchDFlash2::clone_slot(int src_slot, int dst_slot, int len) { impl_->clone_slot(src_slot, dst_slot, len); }
void BatchDFlash2::commit(int n, const std::int32_t* slots, const std::int64_t* start, const std::int32_t* count,
                          const Bf* taps) {
  impl_->commit(n, slots, start, count, taps);
}
BatchDFlash2::Drafts BatchDFlash2::propose(int n, const std::int32_t* slots, const std::int32_t* anchors,
                                           const std::int64_t* pos, const std::int32_t* max_drafts) {
  return impl_->propose(n, slots, anchors, pos, max_drafts);
}
int BatchDFlash2::slots() const { return impl_->slots; }
int BatchDFlash2::max_seqs() const { return impl_->max_seqs; }
std::size_t BatchDFlash2::device_bytes() const {
  return impl_->bytes + (impl_->mx_gemm ? kbd::MxGemm::WORKSPACE_BYTES : 0) + (std::size_t(32) << 20);
}

BatchDFlash2::Precision BatchDFlash2::default_precision() {
  if (const char* e = std::getenv("INGINE_BATCH_DFLASH2_PRECISION")) {
    const std::string_view v(e);
    if (v == "bf16" || v == "BF16") return Precision::BF16;
    if (v == "mxfp8" || v == "MXFP8") return Precision::MXFP8;
    throw std::invalid_argument("INGINE_BATCH_DFLASH2_PRECISION must be mxfp8 or bf16");
  }
  return Precision::MXFP8;
}

bool BatchDFlash2::default_kv_fp8() {
  if (const char* e = std::getenv("INGINE_BATCH_DFLASH2_KV_BF16"); e && *e && std::atoi(e) != 0) return false;
  return true;
}

}  // namespace ingine::model
