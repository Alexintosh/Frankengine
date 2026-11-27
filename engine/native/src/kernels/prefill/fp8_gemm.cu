// Native Blackwell (sm_120a / sm_121a) FP8 E4M3 dense GEMM with fused OUTER_VEC_32F per-row scaling:
//   C[M, N] = (A_q[M, K] * a_scale[M]) . (B_q[N, K] * b_scale[N])^T
// Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 directly on FP8 Tensor Cores with
// a_scale[m] * b_scale[n] applied in FP32 registers before a single final store (BF16 or FP32).
// Eliminates dequant_rows weight expansion, runs at ~150 TFLOP/s on GB10 (zero K-loop scale traffic),
// and supports every TP=1 and TP=2 shape (N = 8600, 6552, 4300, 3276, 2304, 1024, 512, 32, 16; any M >= 1).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ uint16_t f2e4m3_x2(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint16_t out;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(a), "f"(b));
  return out;
#else
  return uint16_t(f2e4m3(a)) | (uint16_t(f2e4m3(b)) << 8);
#endif
}

__device__ __forceinline__ uint32_t pack_e4m3_x4(float a, float b, float c, float d) {
  return uint32_t(f2e4m3_x2(a, b)) | (uint32_t(f2e4m3_x2(c, d)) << 16);
}

// Single-pass coalesced per-row BF16 -> FP8 E4M3 quantization for K <= 4096 (K % 8 == 0):
// 256 threads/CTA, 1 CTA per row, coalesced uint4 reads and uint2 writes, values kept in registers.
__global__ void __launch_bounds__(256)
quantize_fp8_rows_1pass_kernel(const uint16_t* __restrict__ src, int ldx, int K,
                               uint8_t* __restrict__ dst, float* __restrict__ scale_out) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const uint16_t* in = src + size_t(row) * ldx;
  uint8_t* out = dst + size_t(row) * K;

  float v[2][8];
  float amax = 0.f;
  const int k0 = tid * 8;
  const int k1 = k0 + 256 * 8;
  if (k0 < K) {
    const uint4 u = *reinterpret_cast<const uint4*>(in + k0);
    v[0][0] = bf_lo(u.x); v[0][1] = bf_hi(u.x);
    v[0][2] = bf_lo(u.y); v[0][3] = bf_hi(u.y);
    v[0][4] = bf_lo(u.z); v[0][5] = bf_hi(u.z);
    v[0][6] = bf_lo(u.w); v[0][7] = bf_hi(u.w);
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[0][i]));
  }
  if (k1 < K) {
    const uint4 u = *reinterpret_cast<const uint4*>(in + k1);
    v[1][0] = bf_lo(u.x); v[1][1] = bf_hi(u.x);
    v[1][2] = bf_lo(u.y); v[1][3] = bf_hi(u.y);
    v[1][4] = bf_lo(u.z); v[1][5] = bf_hi(u.z);
    v[1][6] = bf_lo(u.w); v[1][7] = bf_hi(u.w);
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[1][i]));
  }

  __shared__ float smax[8];
  amax = warp_max(amax);
  if (lane == 0) smax[warp] = amax;
  __syncthreads();
  float row_max = lane < 8 ? smax[lane] : 0.f;
  row_max = warp_max(row_max);

  const float sc = row_max > 0.f ? (row_max * (1.f / 448.f)) : 1.f;
  const float inv = row_max > 0.f ? (448.f / row_max) : 0.f;
  if (tid == 0) scale_out[row] = sc;

  if (k0 < K) {
    uint2 p = make_uint2(pack_e4m3_x4(v[0][0] * inv, v[0][1] * inv, v[0][2] * inv, v[0][3] * inv),
                         pack_e4m3_x4(v[0][4] * inv, v[0][5] * inv, v[0][6] * inv, v[0][7] * inv));
    *reinterpret_cast<uint2*>(out + k0) = p;
  }
  if (k1 < K) {
    uint2 p = make_uint2(pack_e4m3_x4(v[1][0] * inv, v[1][1] * inv, v[1][2] * inv, v[1][3] * inv),
                         pack_e4m3_x4(v[1][4] * inv, v[1][5] * inv, v[1][6] * inv, v[1][7] * inv));
    *reinterpret_cast<uint2*>(out + k1) = p;
  }
}

__global__ void __launch_bounds__(256)
quantize_fp8_rows_generic_kernel(const uint16_t* __restrict__ src, int ldx, int K,
                                 uint8_t* __restrict__ dst, float* __restrict__ scale_out) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const uint16_t* in = src + size_t(row) * ldx;
  uint8_t* out = dst + size_t(row) * K;

  float amax = 0.f;
  for (int k = tid * 8; k < K; k += 256 * 8) {
    const uint4 u = *reinterpret_cast<const uint4*>(in + k);
    amax = fmaxf(amax, fmaxf(fabsf(bf_lo(u.x)), fabsf(bf_hi(u.x))));
    amax = fmaxf(amax, fmaxf(fabsf(bf_lo(u.y)), fabsf(bf_hi(u.y))));
    amax = fmaxf(amax, fmaxf(fabsf(bf_lo(u.z)), fabsf(bf_hi(u.z))));
    amax = fmaxf(amax, fmaxf(fabsf(bf_lo(u.w)), fabsf(bf_hi(u.w))));
  }
  __shared__ float smax[8];
  amax = warp_max(amax);
  if (lane == 0) smax[warp] = amax;
  __syncthreads();
  float row_max = lane < 8 ? smax[lane] : 0.f;
  row_max = warp_max(row_max);

  const float sc = row_max > 0.f ? (row_max * (1.f / 448.f)) : 1.f;
  const float inv = row_max > 0.f ? (448.f / row_max) : 0.f;
  if (tid == 0) scale_out[row] = sc;

  for (int k = tid * 8; k < K; k += 256 * 8) {
    const uint4 u = *reinterpret_cast<const uint4*>(in + k);
    uint2 p = make_uint2(
        pack_e4m3_x4(bf_lo(u.x) * inv, bf_hi(u.x) * inv, bf_lo(u.y) * inv, bf_hi(u.y) * inv),
        pack_e4m3_x4(bf_lo(u.z) * inv, bf_hi(u.z) * inv, bf_lo(u.w) * inv, bf_hi(u.w) * inv));
    *reinterpret_cast<uint2*>(out + k) = p;
  }
}

__device__ __forceinline__ void cp_async_16_z(void* dst, const void* src, bool valid) {
  const int src_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_addr(dst)), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void mma_fp8_m16n8k32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ int swz64(int r, int c16) { return (r << 6) + ((c16 ^ (r & 3)) << 4); }

template <bool C_FP32>
__device__ __forceinline__ void store_pair(void* C, int ldc, int r, int c0, int N, float v0, float v1) {
  if (c0 + 1 < N) {
    if constexpr (C_FP32) {
      float* dst = static_cast<float*>(C) + size_t(r) * ldc + c0;
      if ((reinterpret_cast<uintptr_t>(dst) & 7u) == 0u) {
        *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);
      } else {
        dst[0] = v0;
        dst[1] = v1;
      }
    } else {
      uint16_t* dst = static_cast<uint16_t*>(C) + size_t(r) * ldc + c0;
      if ((reinterpret_cast<uintptr_t>(dst) & 3u) == 0u) {
        *reinterpret_cast<uint32_t*>(dst) = pack_bf16x2(v0, v1);
      } else {
        dst[0] = f2bf_bits(v0);
        dst[1] = f2bf_bits(v1);
      }
    }
  } else if (c0 < N) {
    if constexpr (C_FP32) {
      static_cast<float*>(C)[size_t(r) * ldc + c0] = v0;
    } else {
      static_cast<uint16_t*>(C)[size_t(r) * ldc + c0] = f2bf_bits(v0);
    }
  }
}

// Large tile: BM=128, BN=128, BK=64, STAGES=3, 256 threads (4x2 warps, 32x64 per warp), 2 CTAs/SM.
template <bool C_FP32>
__global__ void __launch_bounds__(256, 2)
fp8_gemm_128x128_kernel(const uint8_t* __restrict__ A, const float* __restrict__ a_scale, int lda,
                        const uint8_t* __restrict__ B, const float* __restrict__ b_scale, int ldb,
                        void* __restrict__ C, int ldc, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, GROUP_M = 8;
  __shared__ __align__(128) uint8_t sA[STAGES][BM * BK];
  __shared__ __align__(128) uint8_t sB[STAGES][BN * BK];

  const int m_tiles = (M + BM - 1) / BM;
  const int n_tiles = (N + BN - 1) / BN;
  const int linear = blockIdx.x;
  const int gm = min(GROUP_M, m_tiles);
  const int tpg = gm * n_tiles;
  const int gid = linear / tpg;
  const int first_m = gid * gm;
  const int cur_gm = min(gm, m_tiles - first_m);
  const int rem = linear - gid * tpg;
  const int m0 = (first_m + (rem % cur_gm)) * BM;
  const int n0 = (rem / cur_gm) * BN;
  if (m0 >= M || n0 >= N) return;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1;

  float acc[2][8][4] = {};
  const int n_stages = K / BK;

  auto load_stage = [&](int stage, int ks) {
    const int k_off = ks * BK;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = tid + u * 256;
      const int r = idx >> 2, c16 = idx & 3;
      const bool va = (m0 + r) < M;
      const bool vb = (n0 + r) < N;
      cp_async_16_z(&sA[stage][swz64(r, c16)], va ? (A + size_t(m0 + r) * lda + k_off + c16 * 16) : A, va);
      cp_async_16_z(&sB[stage][swz64(r, c16)], vb ? (B + size_t(n0 + r) * ldb + k_off + c16 * 16) : B, vb);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  const int preload = min(STAGES - 1, n_stages);
  for (int s = 0; s < preload; ++s) load_stage(s, s);

  for (int ks = 0; ks < n_stages; ++ks) {
    const int rem_in_flight = min(preload, n_stages - ks) - 1;
    if (rem_in_flight == 1) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    if (ks + preload < n_stages) load_stage((ks + preload) % STAGES, ks + preload);

    const uint8_t* sa = sA[ks % STAGES];
    const uint8_t* sb = sB[ks % STAGES];

    uint32_t a_reg[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int c_off = kk * 2;
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16 + (lane & 15);
        ldmatrix_x4(a_reg[mi][0], a_reg[mi][1], a_reg[mi][2], a_reg[mi][3],
                    smem_addr(sa + swz64(r, c_off + (lane >> 4))));
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        uint32_t b_reg[4];
        const int r = wn * 64 + ni * 16 + (lane & 15);
        ldmatrix_x4(b_reg[0], b_reg[1], b_reg[2], b_reg[3],
                    smem_addr(sb + swz64(r, c_off + (lane >> 4))));
        mma_fp8_m16n8k32(acc[0][2 * ni],     a_reg[0], b_reg[0], b_reg[2]);
        mma_fp8_m16n8k32(acc[0][2 * ni + 1], a_reg[0], b_reg[1], b_reg[3]);
        mma_fp8_m16n8k32(acc[1][2 * ni],     a_reg[1], b_reg[0], b_reg[2]);
        mma_fp8_m16n8k32(acc[1][2 * ni + 1], a_reg[1], b_reg[1], b_reg[3]);
      }
    }
  }

  const int grp = lane >> 2, t4 = lane & 3;
  float2 sb[8];
#pragma unroll
  for (int ni = 0; ni < 8; ++ni) {
    const int c0 = n0 + wn * 64 + ni * 8 + t4 * 2;
    if (c0 + 1 < N) sb[ni] = __ldg(reinterpret_cast<const float2*>(b_scale + c0));
    else sb[ni] = make_float2(c0 < N ? __ldg(b_scale + c0) : 0.f, 0.f);
  }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int r0 = m0 + wm * 32 + mi * 16 + grp;
    const int r1 = r0 + 8;
    const float sa0 = r0 < M ? __ldg(a_scale + r0) : 0.f;
    const float sa1 = r1 < M ? __ldg(a_scale + r1) : 0.f;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      const int c0 = n0 + wn * 64 + ni * 8 + t4 * 2;
      if (c0 < N) {
        if (r0 < M) store_pair<C_FP32>(C, ldc, r0, c0, N, acc[mi][ni][0] * sa0 * sb[ni].x, acc[mi][ni][1] * sa0 * sb[ni].y);
        if (r1 < M) store_pair<C_FP32>(C, ldc, r1, c0, N, acc[mi][ni][2] * sa1 * sb[ni].x, acc[mi][ni][3] * sa1 * sb[ni].y);
      }
    }
  }
}

// Small tile: BM=64, BN=64, BK=64, STAGES=3, 128 threads (2x2 warps, 32x32 per warp), 4 CTAs/SM.
template <bool C_FP32>
__global__ void __launch_bounds__(128, 4)
fp8_gemm_64x64_kernel(const uint8_t* __restrict__ A, const float* __restrict__ a_scale, int lda,
                      const uint8_t* __restrict__ B, const float* __restrict__ b_scale, int ldb,
                      void* __restrict__ C, int ldc, int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 64, STAGES = 3;
  __shared__ __align__(128) uint8_t sA[STAGES][BM * BK];
  __shared__ __align__(128) uint8_t sB[STAGES][BN * BK];

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  if (m0 >= M || n0 >= N) return;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1;

  float acc[2][4][4] = {};
  const int n_stages = K / BK;

  auto load_stage = [&](int stage, int ks) {
    const int k_off = ks * BK;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = tid + u * 128;
      const int r = idx >> 2, c16 = idx & 3;
      const bool va = (m0 + r) < M;
      const bool vb = (n0 + r) < N;
      cp_async_16_z(&sA[stage][swz64(r, c16)], va ? (A + size_t(m0 + r) * lda + k_off + c16 * 16) : A, va);
      cp_async_16_z(&sB[stage][swz64(r, c16)], vb ? (B + size_t(n0 + r) * ldb + k_off + c16 * 16) : B, vb);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  const int preload = min(STAGES - 1, n_stages);
  for (int s = 0; s < preload; ++s) load_stage(s, s);

  for (int ks = 0; ks < n_stages; ++ks) {
    const int rem_in_flight = min(preload, n_stages - ks) - 1;
    if (rem_in_flight == 1) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    if (ks + preload < n_stages) load_stage((ks + preload) % STAGES, ks + preload);

    const uint8_t* sa = sA[ks % STAGES];
    const uint8_t* sb = sB[ks % STAGES];

    uint32_t a_reg[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int c_off = kk * 2;
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16 + (lane & 15);
        ldmatrix_x4(a_reg[mi][0], a_reg[mi][1], a_reg[mi][2], a_reg[mi][3],
                    smem_addr(sa + swz64(r, c_off + (lane >> 4))));
      }
#pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        uint32_t b_reg[4];
        const int r = wn * 32 + ni * 16 + (lane & 15);
        ldmatrix_x4(b_reg[0], b_reg[1], b_reg[2], b_reg[3],
                    smem_addr(sb + swz64(r, c_off + (lane >> 4))));
        mma_fp8_m16n8k32(acc[0][2 * ni],     a_reg[0], b_reg[0], b_reg[2]);
        mma_fp8_m16n8k32(acc[0][2 * ni + 1], a_reg[0], b_reg[1], b_reg[3]);
        mma_fp8_m16n8k32(acc[1][2 * ni],     a_reg[1], b_reg[0], b_reg[2]);
        mma_fp8_m16n8k32(acc[1][2 * ni + 1], a_reg[1], b_reg[1], b_reg[3]);
      }
    }
  }

  const int grp = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int r0 = m0 + wm * 32 + mi * 16 + grp;
    const int r1 = r0 + 8;
    const float sa0 = r0 < M ? a_scale[r0] : 0.f;
    const float sa1 = r1 < M ? a_scale[r1] : 0.f;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int c0 = n0 + wn * 32 + ni * 8 + t4 * 2;
      if (c0 < N) {
        const float sb0 = b_scale[c0];
        const float sb1 = (c0 + 1 < N) ? b_scale[c0 + 1] : 0.f;
        if (r0 < M) store_pair<C_FP32>(C, ldc, r0, c0, N, acc[mi][ni][0] * sa0 * sb0, acc[mi][ni][1] * sa0 * sb1);
        if (r1 < M) store_pair<C_FP32>(C, ldc, r1, c0, N, acc[mi][ni][2] * sa1 * sb0, acc[mi][ni][3] * sa1 * sb1);
      }
    }
  }
}

// Each warp owns up to 16 output columns and one or two m16 fragments.
template <int BM, int BN, bool C_FP32>
__global__ void __launch_bounds__(((BN + 15) / 16) * 32, 8)
fp8_gemm_small_kernel(const uint8_t* __restrict__ A, const float* __restrict__ a_scale, int lda,
                      const uint8_t* __restrict__ B, const float* __restrict__ b_scale, int ldb,
                      void* __restrict__ C, int ldc, int M, int N, int K) {
  constexpr int BK = 64, STAGES = 3, THREADS = ((BN + 15) / 16) * 32;
  constexpr int M_FRAGMENTS = BM / 16, N_FRAGMENTS = BN == 8 ? 1 : 2;
  __shared__ __align__(128) uint8_t sA[STAGES][BM * BK];
  __shared__ __align__(128) uint8_t sB[STAGES][BN * BK];

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_warp = warp * 16;
  const int n_stages = K / BK;
  float acc[M_FRAGMENTS][N_FRAGMENTS][4] = {};

  auto load_stage = [&](int stage, int ks) {
    const int k_off = ks * BK;
#pragma unroll
    for (int u = 0; u < BM * BK / (16 * THREADS); ++u) {
      const int idx = tid + u * THREADS;
      const int r = idx >> 2, c16 = idx & 3;
      const bool valid = m0 + r < M;
      cp_async_16_z(&sA[stage][swz64(r, c16)], valid ? A + size_t(m0 + r) * lda + k_off + c16 * 16 : A, valid);
    }
#pragma unroll
    for (int u = 0; u < BN * BK / (16 * THREADS); ++u) {
      const int idx = tid + u * THREADS;
      const int r = idx >> 2, c16 = idx & 3;
      const bool valid = n0 + r < N;
      cp_async_16_z(&sB[stage][swz64(r, c16)], valid ? B + size_t(n0 + r) * ldb + k_off + c16 * 16 : B, valid);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  const int preload = min(STAGES - 1, n_stages);
  for (int s = 0; s < preload; ++s) load_stage(s, s);

  for (int ks = 0; ks < n_stages; ++ks) {
    const int rem_in_flight = min(preload, n_stages - ks) - 1;
    if (rem_in_flight == 1) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    if constexpr (THREADS == 32) __syncwarp();
    else __syncthreads();

    if (ks + preload < n_stages) load_stage((ks + preload) % STAGES, ks + preload);

    const uint8_t* sa = sA[ks % STAGES];
    const uint8_t* sb = sB[ks % STAGES];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int c16 = kk * 2 + (lane >> 4);
      uint32_t b_reg[2 * N_FRAGMENTS];
      if constexpr (BN == 8) {
        // The first 16 lanes supply eight row addresses per half of K=32.
        const unsigned addr = smem_addr(sb + swz64(lane & 7, kk * 2 + ((lane >> 3) & 1)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(b_reg[0]), "=r"(b_reg[1]) : "r"(addr));
      } else {
        ldmatrix_x4(b_reg[0], b_reg[1], b_reg[2], b_reg[3],
                    smem_addr(sb + swz64(n_warp + (lane & 15), c16)));
      }
#pragma unroll
      for (int mi = 0; mi < M_FRAGMENTS; ++mi) {
        uint32_t a_reg[4];
        ldmatrix_x4(a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                    smem_addr(sa + swz64(mi * 16 + (lane & 15), c16)));
#pragma unroll
        for (int ni = 0; ni < N_FRAGMENTS; ++ni) {
          mma_fp8_m16n8k32(acc[mi][ni], a_reg, b_reg[ni], b_reg[N_FRAGMENTS + ni]);
        }
      }
    }
  }

#pragma unroll
  for (int mi = 0; mi < M_FRAGMENTS; ++mi) {
    const int r0 = m0 + mi * 16 + (lane >> 2);
    const int r1 = r0 + 8;
    const float sa0 = r0 < M ? a_scale[r0] : 0.f;
    const float sa1 = r1 < M ? a_scale[r1] : 0.f;
#pragma unroll
    for (int ni = 0; ni < N_FRAGMENTS; ++ni) {
      const int c0 = n0 + n_warp + ni * 8 + (lane & 3) * 2;
      if (c0 < N) {
        const float sb0 = b_scale[c0];
        const float sb1 = c0 + 1 < N ? b_scale[c0 + 1] : 0.f;
        if (r0 < M) store_pair<C_FP32>(C, ldc, r0, c0, N, acc[mi][ni][0] * sa0 * sb0, acc[mi][ni][1] * sa0 * sb1);
        if (r1 < M) store_pair<C_FP32>(C, ldc, r1, c0, N, acc[mi][ni][2] * sa1 * sb0, acc[mi][ni][3] * sa1 * sb1);
      }
    }
  }
}

template <bool C_FP32>
void launch_fp8_dense_gemm(const uint8_t* A, const float* a_scale, int lda,
                           const uint8_t* B, const float* b_scale, int ldb,
                           void* C, int ldc, int M, int N, int K, cudaStream_t stream, bool legacy) {
  constexpr int GB10_SMS = 48;
  const bool keep_legacy_tiles = legacy || M >= 512 || M < 16 || N <= 64;
  const int tiles_128 = ((M + 127) / 128) * ((N + 127) / 128);
  if (keep_legacy_tiles ? (M > 64 && N > 64) : (tiles_128 >= GB10_SMS)) {
    fp8_gemm_128x128_kernel<C_FP32><<<tiles_128, 256, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
    return;
  }

  const dim3 grid64((N + 63) / 64, (M + 63) / 64);
  if (keep_legacy_tiles || grid64.x * grid64.y >= GB10_SMS) {
    fp8_gemm_64x64_kernel<C_FP32><<<grid64, 128, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
    return;
  }

  // Use the largest tile that fills a wave, so fewer CTAs reload the same weights.
  const dim3 grid32((N + 31) / 32, (M + 31) / 32);
  const dim3 grid16x32(grid32.x, (M + 15) / 16);
  const dim3 grid16((N + 15) / 16, grid16x32.y);
  if (grid32.x * grid32.y >= GB10_SMS) {
    fp8_gemm_small_kernel<32, 32, C_FP32><<<grid32, 64, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
  } else if (grid16x32.x * grid16x32.y >= GB10_SMS) {
    fp8_gemm_small_kernel<16, 32, C_FP32><<<grid16x32, 64, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
  } else if (grid16.x * grid16.y >= GB10_SMS) {
    fp8_gemm_small_kernel<16, 16, C_FP32><<<grid16, 32, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
  } else {
    const dim3 grid16x8((N + 7) / 8, grid16x32.y);
    fp8_gemm_small_kernel<16, 8, C_FP32><<<grid16x8, 32, 0, stream>>>(A, a_scale, lda, B, b_scale, ldb, C, ldc, M, N, K);
  }
}

}  // namespace

void quantize_fp8_rows(const __nv_bfloat16* x, int ldx, int M, int K, uint8_t* q, float* scale, cudaStream_t stream) {
  if (M <= 0 || K <= 0 || K % 8 != 0 || ldx < K)
    throw std::invalid_argument("quantize_fp8_rows: invalid shape");
  const auto* src = reinterpret_cast<const uint16_t*>(x);
  if (K <= 4096)
    quantize_fp8_rows_1pass_kernel<<<M, 256, 0, stream>>>(src, ldx, K, q, scale);
  else
    quantize_fp8_rows_generic_kernel<<<M, 256, 0, stream>>>(src, ldx, K, q, scale);
}

void fp8_dense_gemm(const uint8_t* A_q, const float* a_scale, int lda,
                    const void* B_q, const float* b_scale, int ldb,
                    void* C, int ldc, bool c_fp32, int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || K % 64 != 0 || lda % 16 != 0 || ldb % 16 != 0)
    throw std::invalid_argument("fp8_dense_gemm: unsupported shape or alignment");
  static const bool legacy = [] {
    const char* value = std::getenv("INGINE_FP8_GEMM_LEGACY");
    return value && value[0] == '1' && value[1] == '\0';
  }();
  const auto* b = static_cast<const uint8_t*>(B_q);
  if (c_fp32)
    launch_fp8_dense_gemm<true>(A_q, a_scale, lda, b, b_scale, ldb, C, ldc, M, N, K, stream, legacy);
  else
    launch_fp8_dense_gemm<false>(A_q, a_scale, lda, b, b_scale, ldb, C, ldc, M, N, K, stream, legacy);
}

}  // namespace ingine::kernels::prefill
