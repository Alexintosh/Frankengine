// Grouped expert GEMMs of the prefill (model._moe_grouped) over NVFP4 / MXFP8 experts.
//
// Two backends:
// 1. Native Blackwell block-scaled Tensor Core MMA (default on sm_120a / sm_121a when MoeWork scratch is supplied):
//    - Pre-quantizes BF16 activations once per call (exact per-row outer scale for NVFP4, per-32 UE8M0 for MXFP8)
//      so quantized activations fit in L2 and are shared across all N-tiles via 100% asynchronous cp.async.
//    - NVFP4 (FMT=0): mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//      3-stage async A pipeline (AS=3) + 2-stage 128B cache-line W pipeline (WK=256), W scales loaded directly
//      into registers (uint4 per thread -> 0 SMEM), <= 46 KB/CTA -> 2 CTAs/SM on GB10.
//      Supports TM=32xTN=128 (T<=512), TM=128xTN=128 (512<T<=1024), and TM=256xTN=64 (T>1024, 2x4 warps,
//      eliminating 2-tile weight re-reads at T=2048).
//    - MXFP8 (FMT=1): mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
//      Checkpoint e4m3 codes (WK=128) and ue8m0 scales stream straight into MMA with zero weight dequantization.
// 2. BF16 dequant-on-the-fly fallback (INGINE_PREFILL_MMA=bf16 or when scratch is null).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

enum class Mode { Up, Down, DownPartial };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1;

constexpr int TN_DEFAULT = 128, BK = 64;
constexpr int AS_BF16 = 3;
constexpr int NT = 256;
constexpr int KCH = BK / 8;

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

enum class MmaBackend { Native, Bf16, Nvfp4Only, Mxfp8Only };

MmaBackend select_backend() {
  static const MmaBackend b = [] {
    const char* e = std::getenv("INGINE_PREFILL_MMA");
    if (!e || !*e) return MmaBackend::Native;
    if (std::strcmp(e, "bf16") == 0) return MmaBackend::Bf16;
    if (std::strcmp(e, "nvfp4") == 0) return MmaBackend::Nvfp4Only;
    if (std::strcmp(e, "mxfp8") == 0) return MmaBackend::Mxfp8Only;
    return MmaBackend::Native;
  }();
  return b;
}

struct GemmArgs {
  const uint8_t* codes = nullptr;
  const uint8_t* scales = nullptr;
  const float* factors = nullptr;
  const uint16_t* a = nullptr;
  MoeWork w;
  uint16_t* out = nullptr;
};

__device__ __forceinline__ void cp_async4(void* smem_dst, const void* gmem_src, int src_bytes = 4) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src),
               "r"(src_bytes));
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// =================================================================================================
// Native Blackwell pre-quantize + block-scaled MMA kernels (sm_120a / sm_121a)
// =================================================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200

__device__ __forceinline__ uint32_t cvt_e2m1x8(const float (&v)[8]) {
  uint32_t r;
  asm("{\n"
      ".reg .b8 t0, t1, t2, t3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 t0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 t1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 t2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 t3, %8, %7;\n"
      "mov.b32 %0, {t0, t1, t2, t3};\n"
      "}"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}

__device__ __forceinline__ uint32_t cvt_e4m3x4(float v0, float v1, float v2, float v3) {
  const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v2, v3), __NV_SATFINITE, __NV_E4M3);
  return uint32_t(lo) | (uint32_t(hi) << 16);
}

__device__ __forceinline__ int mx_exp(float amax) {
  if (!(amax > 0.f)) return -127;
  int ex;
  const float m = frexpf(__fdiv_rn(amax, 448.f), &ex);
  return max(m == 0.5f ? ex - 1 : ex, -127);
}

__device__ __forceinline__ float mx_inv(int e) { return __int_as_float((127 - e) << 23); }

template <int TidB>
__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                          unsigned sfa, unsigned sfb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, %12};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(TidB));
}

template <int ByteId, int TidB>
__device__ __forceinline__ void mma_mxfp8(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                          unsigned sfa, unsigned sfb) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, 0}, %11, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(ByteId), "n"(TidB));
}

__device__ __forceinline__ int swz_w128(int r, int c16) { return (r << 7) + ((c16 ^ (r & 7)) << 4); }
__device__ __forceinline__ int swz_a32(int r, int c16) { return (r << 5) + ((c16 ^ ((r >> 2) & 1)) << 4); }
__device__ __forceinline__ int swz_a64(int r, int c16) { return (r << 6) + ((c16 ^ ((r >> 1) & 3)) << 4); }

template <int MT, int WM_ROWS>
__device__ __forceinline__ void nvfp4_mma_step(float (&acc)[WM_ROWS / 16][2][4], const uint8_t* a_smem,
                                               const uint8_t* asc_smem, int m_base, const unsigned (&b)[4],
                                               unsigned sfb, int lane) {
  const int grp = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int mr = m_base + mt * 16;
    unsigned a[4];
    ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(a_smem + swz_a32(mr + (lane & 15), lane >> 4)));
    const unsigned sfa = (t4 < 2) ? *reinterpret_cast<const uint32_t*>(asc_smem + (mr + grp + t4 * 8) * 4) : 0u;
    mma_nvfp4<0>(acc[mt][0], a, b[0], b[1], sfa, sfb);
    mma_nvfp4<1>(acc[mt][1], a, b[2], b[3], sfa, sfb);
  }
}

#endif  // __CUDA_ARCH__ >= 1200

// Single-pass per-row BF16 -> NVFP4 quantization (exact per-row outer scale = row_amax / (448 * 6)).
template <int K>
__global__ void quantize_nvfp4_rows_kernel(const uint16_t* __restrict__ src, uint8_t* __restrict__ codes,
                                           uint8_t* __restrict__ scales, float* __restrict__ factors) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int NB = K / 16;
  constexpr int WARPS = (NB + 31) / 32;
  __shared__ float wmax[WARPS];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float f0[8], f1[8];
  float bmax = 0.f;
  if (tid < NB) {
    const uint16_t* p = src + size_t(row) * K + tid * 16;
    const uint4 v0 = *reinterpret_cast<const uint4*>(p);
    const uint4 v1 = *reinterpret_cast<const uint4*>(p + 8);
    f0[0] = bf_lo(v0.x); f0[1] = bf_hi(v0.x); f0[2] = bf_lo(v0.y); f0[3] = bf_hi(v0.y);
    f0[4] = bf_lo(v0.z); f0[5] = bf_hi(v0.z); f0[6] = bf_lo(v0.w); f0[7] = bf_hi(v0.w);
    f1[0] = bf_lo(v1.x); f1[1] = bf_hi(v1.x); f1[2] = bf_lo(v1.y); f1[3] = bf_hi(v1.y);
    f1[4] = bf_lo(v1.z); f1[5] = bf_hi(v1.z); f1[6] = bf_lo(v1.w); f1[7] = bf_hi(v1.w);
#pragma unroll
    for (int i = 0; i < 8; ++i) bmax = fmaxf(bmax, fmaxf(fabsf(f0[i]), fabsf(f1[i])));
  }
  float wm = bmax;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, o));
  if (lane == 0) wmax[warp] = wm;
  __syncthreads();
  float rmax = 0.f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) rmax = fmaxf(rmax, wmax[w]);
  const float rfac = rmax * (1.f / (448.f * 6.f));
  if (tid == 0) factors[row] = rfac;
  if (tid < NB) {
    uint2 q = make_uint2(0, 0);
    uint8_t sc_b = 0;
    if (rfac > 0.f) {
      sc_b = f2e4m3(__fdiv_rn(bmax, __fmul_rn(6.f, rfac)));
      const float sc_f = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&sc_b));
      if (sc_f > 0.f) {
        const float inv = __frcp_rn(__fmul_rn(sc_f, rfac));
#pragma unroll
        for (int i = 0; i < 8; ++i) f0[i] *= inv, f1[i] *= inv;
        q.x = cvt_e2m1x8(f0);
        q.y = cvt_e2m1x8(f1);
      }
    }
    *reinterpret_cast<uint2*>(codes + size_t(row) * (K / 2) + tid * 8) = q;
    scales[size_t(row) * NB + tid] = sc_b;
  }
#endif
}

template <int K>
__global__ void quantize_mxfp8_rows_kernel(const uint16_t* __restrict__ src, uint8_t* __restrict__ codes,
                                           uint8_t* __restrict__ scales) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int NB = K / 32;
  constexpr int SC_STRIDE = (NB + 3) / 4 * 4;
  const int row = blockIdx.x, tid = threadIdx.x;
  if (tid >= NB) return;
  const uint16_t* p = src + size_t(row) * K + tid * 32;
  const uint4 v[4] = {
      *reinterpret_cast<const uint4*>(p),
      *reinterpret_cast<const uint4*>(p + 8),
      *reinterpret_cast<const uint4*>(p + 16),
      *reinterpret_cast<const uint4*>(p + 24),
  };
  float f[32];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    f[8 * j + 0] = bf_lo(v[j].x); f[8 * j + 1] = bf_hi(v[j].x);
    f[8 * j + 2] = bf_lo(v[j].y); f[8 * j + 3] = bf_hi(v[j].y);
    f[8 * j + 4] = bf_lo(v[j].z); f[8 * j + 5] = bf_hi(v[j].z);
    f[8 * j + 6] = bf_lo(v[j].w); f[8 * j + 7] = bf_hi(v[j].w);
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  const int ex = mx_exp(amax);
  const float inv = mx_inv(ex);
  uint4 q0, q1;
  q0.x = cvt_e4m3x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
  q0.y = cvt_e4m3x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
  q0.z = cvt_e4m3x4(f[8] * inv, f[9] * inv, f[10] * inv, f[11] * inv);
  q0.w = cvt_e4m3x4(f[12] * inv, f[13] * inv, f[14] * inv, f[15] * inv);
  q1.x = cvt_e4m3x4(f[16] * inv, f[17] * inv, f[18] * inv, f[19] * inv);
  q1.y = cvt_e4m3x4(f[20] * inv, f[21] * inv, f[22] * inv, f[23] * inv);
  q1.z = cvt_e4m3x4(f[24] * inv, f[25] * inv, f[26] * inv, f[27] * inv);
  q1.w = cvt_e4m3x4(f[28] * inv, f[29] * inv, f[30] * inv, f[31] * inv);
  uint8_t* dst = codes + size_t(row) * K + tid * 32;
  *reinterpret_cast<uint4*>(dst) = q0;
  *reinterpret_cast<uint4*>(dst + 16) = q1;
  scales[size_t(row) * SC_STRIDE + tid] = uint8_t(ex + 127);
#endif
}

// NVFP4 config: supports (TM=32,TN=128,WM=1,WN=8), (TM=128,TN=128,WM=1,WN=8), (TM=256,TN=64,WM=2,WN=4).
// 3-stage A pipeline (AS=3) + 2-stage W pipeline (WS=2), W scales loaded directly into registers (0 SMEM!).
template <int K, int TM, int TN_>
struct Nvfp4Cfg {
  static constexpr int TN = TN_;
  static constexpr int WN = TN / 16;          // 8 or 4 warps across N
  static constexpr int WM = 8 / WN;           // 1 or 2 warps across M
  static constexpr int WM_ROWS = TM / WM;     // 32 or 128 rows per warp
  static constexpr int CODE_ROW = K / 2;
  static constexpr int SCALE_ROW = K / 16;
  static constexpr int WK = 256;              // 128 code bytes per weight row (full L2 line)
  static constexpr int KC = (K + WK - 1) / WK;
  static constexpr int WS = 2;
  static constexpr int AS_N = 3;
  static constexpr int W_BYTES = TN * 128;    // 16384 B (TN=128) or 8192 B (TN=64)
  static constexpr int A_BYTES = TM * 32;     // 1024, 4096, or 8192 B
  static constexpr int ASC_BYTES = TM * 4;    // 128, 512, or 1024 B
  static constexpr int OFF_W = 0;
  static constexpr int OFF_A = WS * W_BYTES;
  static constexpr int OFF_ASC = OFF_A + AS_N * A_BYTES;
  static constexpr int SMEM = (OFF_ASC + AS_N * ASC_BYTES + 15) / 16 * 16;
  static_assert(K % BK == 0 && SCALE_ROW % 4 == 0 && SMEM <= 49152);  // <= 48 KB -> 2 CTAs/SM
};

template <Mode MODE, int K, int TM, int TN_, int I>
__global__ void __launch_bounds__(NT, 2) wgemm_nvfp4_kernel(const GemmArgs g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * I : kHidden;
  using C = Nvfp4Cfg<K, TM, TN_>;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* w_smem = smem + C::OFF_W;
  uint8_t* a_smem = smem + C::OFF_A;
  uint8_t* asc_smem = smem + C::OFF_ASC;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int grp = lane >> 2, t4 = lane & 3;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int m_base = warp_m * C::WM_ROWS;

  constexpr int NTILES_N = NE / C::TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * C::TN;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW;

  // Per-thread A pointers for load_a:
  // TM=32:  64 chunks (tid < 64), 32 scales (tid < 32)
  // TM=128: 256 chunks (1/thread), 128 scales (tid < 128)
  // TM=256: 512 chunks (2/thread: same row r=tid, both c16=0 and c16=1!), 256 scales (1/thread: r=tid)
  const int a_r = (TM == 256) ? tid : (tid >> 1);
  const bool a_code_active = (TM == 32) ? (tid < 64) : true;
  const bool a_code_valid = a_code_active && (a_r < n);
  const int a_row_idx = a_code_valid ? (MODE == Mode::Up ? g.w.pair_token[pb + a_r] : (pb + a_r)) : 0;
  const uint8_t* a_code_row = g.w.a_codes + int64_t(a_row_idx) * C::CODE_ROW;

  const bool a_sc_valid = (tid < TM) && (tid < n);
  const int a_sc_idx = a_sc_valid ? (MODE == Mode::Up ? g.w.pair_token[pb + tid] : (pb + tid)) : 0;
  const uint8_t* a_sc_row = g.w.a_scales + int64_t(a_sc_idx) * C::SCALE_ROW;

  auto load_w = [&](int chunk) {
    uint8_t* ws = w_smem + (chunk & 1) * C::W_BYTES;
    constexpr int W_ITERS = (C::TN * 8) / NT;  // 4 for TN=128, 2 for TN=64
#pragma unroll
    for (int i = 0; i < W_ITERS; ++i) {
      const int id = tid + i * NT, row = id >> 3, c16 = id & 7;
      const int start = chunk * 128 + (c16 << 4);
      const bool valid = (K % C::WK == 0) || (start < C::CODE_ROW);
      cp_async16(ws + swz_w128(row, c16), code_tile + int64_t(row) * C::CODE_ROW + (valid ? start : 0),
                 valid ? 16 : 0);
    }
  };

  auto load_a = [&](int ks) {
    uint8_t* ad = a_smem + (ks % C::AS_N) * C::A_BYTES;
    uint8_t* scd = asc_smem + (ks % C::AS_N) * C::ASC_BYTES;
    if constexpr (TM == 256) {
      cp_async16(ad + swz_a32(a_r, 0), a_code_row + ks * 32, a_code_valid ? 16 : 0);
      cp_async16(ad + swz_a32(a_r, 1), a_code_row + ks * 32 + 16, a_code_valid ? 16 : 0);
    } else if (a_code_active) {
      const int a_c16 = tid & 1;
      cp_async16(ad + swz_a32(a_r, a_c16), a_code_row + ks * 32 + (a_c16 << 4), a_code_valid ? 16 : 0);
    }
    if (tid < TM) {
      cp_async4(scd + tid * 4, a_sc_row + ks * 4, a_sc_valid ? 4 : 0);
    }
  };

  // Async prologue: group 0 = W[0]+A[0], group 1 = A[1]+W[1] (AS_N=3 guarantees A[2] goes into disjoint slot 2!)
  load_w(0);
  load_a(0);
  cp_commit();
  if constexpr (KS > 1) {
    load_a(1);
    if constexpr (C::KC > 1) load_w(1);
    cp_commit();
  }

  const int wm_n = max(0, min(C::WM_ROWS, n - m_base));
  const int mact = (wm_n + 15) / 16;
  float acc[C::WM_ROWS / 16][2][4];
#pragma unroll
  for (int a = 0; a < C::WM_ROWS / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  const int brow = warp_n * 16 + (lane & 7) + ((lane >> 4) ? 8 : 0);
  const int bhalf = (lane >> 3) & 1;
  const uint8_t* wsc_row = scale_tile + int64_t(warp_n * 16 + grp + t4 * 8) * C::SCALE_ROW;

#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    if (ks + 1 < KS) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();
    const int chunk = ks >> 2, sub = ks & 3;
    if (ks + 2 < KS) {
      load_a(ks + 2);
      if (sub == 0 && chunk >= 1 && chunk + 1 < C::KC) load_w(chunk + 1);
      cp_commit();
    }

    if (mact > 0) {
      const uint8_t* ws = w_smem + (chunk & 1) * C::W_BYTES;
      const uint8_t* A = a_smem + (ks % C::AS_N) * C::A_BYTES;
      const uint8_t* Asc = asc_smem + (ks % C::AS_N) * C::ASC_BYTES;

      unsigned b[4];
      ldmatrix_x4(b[0], b[1], b[2], b[3], smem_addr(ws + swz_w128(brow, 2 * sub + bhalf)));
      const unsigned sfb = (t4 < 2) ? __ldg(reinterpret_cast<const uint32_t*>(wsc_row + ks * 4)) : 0u;

      switch (mact) {
        case 1: if constexpr (1 <= C::WM_ROWS / 16) nvfp4_mma_step<1, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 2: if constexpr (2 <= C::WM_ROWS / 16) nvfp4_mma_step<2, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 3: if constexpr (3 <= C::WM_ROWS / 16) nvfp4_mma_step<3, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 4: if constexpr (4 <= C::WM_ROWS / 16) nvfp4_mma_step<4, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 5: if constexpr (5 <= C::WM_ROWS / 16) nvfp4_mma_step<5, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 6: if constexpr (6 <= C::WM_ROWS / 16) nvfp4_mma_step<6, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        case 7: if constexpr (7 <= C::WM_ROWS / 16) nvfp4_mma_step<7, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
        default: nvfp4_mma_step<C::WM_ROWS / 16, C::WM_ROWS>(acc, A, Asc, m_base, b, sfb, lane); break;
      }
    }
  }
  cp_wait<0>();

  const float w_fac = __ldg(g.factors + e);
#pragma unroll
  for (int mt = 0; mt < C::WM_ROWS / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = m_base + mt * 16 + grp + half * 8;
      if (row >= n) continue;
      const int a_id = (MODE == Mode::Up) ? g.w.pair_token[pb + row] : (pb + row);
      const float scale_out = w_fac * __ldg(g.w.a_factors + a_id);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp_n * 16 + nt * 8 + t4 * 2;
        const float c0 = acc[mt][nt][2 * half] * scale_out;
        const float c1 = acc[mt][nt][2 * half + 1] * scale_out;
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * I + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          const auto offset = size_t(dst) * kHidden + n0 + cl;
          if constexpr (MODE == Mode::DownPartial)
            *reinterpret_cast<float2*>(reinterpret_cast<float*>(g.out) + offset) = make_float2(c0, c1);
          else
            *reinterpret_cast<uint32_t*>(g.out + offset) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
#endif
}

// MXFP8 config: W scales loaded via __ldg directly into registers (0 SMEM for WSC!),
// freeing SMEM for AS_M=2 (TM=128) or AS_M=3 (TM=32) and 2 CTAs/SM!
template <int K, int TM, int TN_>
struct Mxfp8Cfg {
  static constexpr int TN = TN_;
  static constexpr int WN = TN / 16;
  static constexpr int WM = 8 / WN;
  static constexpr int WM_ROWS = TM / WM;
  static constexpr int CODE_ROW = K;
  static constexpr int SCALE_ROW = K / 32;
  static constexpr int A_SCALE_STRIDE = (SCALE_ROW + 3) / 4 * 4;
  static constexpr int WK = 128;
  static constexpr int KC = (K + WK - 1) / WK;
  static constexpr int WS = 2;
  static constexpr int AS_M = (TM == 32) ? 3 : 1;
  static constexpr int W_BYTES = TN * 128;
  static constexpr int A_BYTES = TM * 64;
  static constexpr int ASC_BYTES = TM * 4;
  static constexpr int OFF_W = 0;
  static constexpr int OFF_A = WS * W_BYTES;
  static constexpr int OFF_ASC = OFF_A + AS_M * A_BYTES;
  static constexpr int SMEM = (OFF_ASC + AS_M * ASC_BYTES + 15) / 16 * 16;
  static_assert(K % BK == 0 && SMEM <= 49152);  // <= 48 KB -> 2 CTAs/SM
};

template <Mode MODE, int K, int TM, int TN_, int I>
__global__ void __launch_bounds__(NT, 2) wgemm_mxfp8_kernel(const GemmArgs g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * I : kHidden;
  using C = Mxfp8Cfg<K, TM, TN_>;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* w_smem = smem + C::OFF_W;
  uint8_t* a_smem = smem + C::OFF_A;
  uint8_t* asc_smem = smem + C::OFF_ASC;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int grp = lane >> 2, t4 = lane & 3;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int m_base = warp_m * C::WM_ROWS;

  constexpr int NTILES_N = NE / C::TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * C::TN;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW;

  // Per-thread A row pointers:
  // TM=32:  128 chunks (tid < 128: r=tid>>2, c16=tid&3)
  // TM=128: 512 chunks (2/thread: r=tid>>1, c16=(tid&1) and +2)
  // TM=256: 1024 chunks (4/thread: r=tid, c16=0..3)
  const int a_r = (TM == 256) ? tid : (TM == 128) ? (tid >> 1) : (tid >> 2);
  const bool a_active = (TM == 32) ? (tid < 128) : true;
  const bool a_valid = a_active && (a_r < n);
  const int a_idx = a_valid ? (MODE == Mode::Up ? g.w.pair_token[pb + a_r] : (pb + a_r)) : 0;
  const uint8_t* a_code_row = g.w.a_codes + int64_t(a_idx) * C::CODE_ROW;

  const bool asc_valid = (tid < TM) && (tid < n);
  const int asc_idx = asc_valid ? (MODE == Mode::Up ? g.w.pair_token[pb + tid] : (pb + tid)) : 0;
  const uint8_t* a_sc_row = g.w.a_scales + int64_t(asc_idx) * C::A_SCALE_STRIDE;

  auto load_w = [&](int chunk) {
    uint8_t* ws = w_smem + (chunk & 1) * C::W_BYTES;
    constexpr int W_ITERS = (C::TN * 8) / NT;
#pragma unroll
    for (int i = 0; i < W_ITERS; ++i) {
      const int id = tid + i * NT, row = id >> 3, c16 = id & 7;
      const int start = chunk * 128 + (c16 << 4);
      const bool valid = (K % C::WK == 0) || (start < C::CODE_ROW);
      cp_async16(ws + swz_w128(row, c16), code_tile + int64_t(row) * C::CODE_ROW + (valid ? start : 0),
                 valid ? 16 : 0);
    }
  };

  auto load_a = [&](int ks) {
    uint8_t* ad = a_smem + (ks % C::AS_M) * C::A_BYTES;
    uint8_t* scd = asc_smem + (ks % C::AS_M) * C::ASC_BYTES;
    if constexpr (TM == 256) {
#pragma unroll
      for (int c0 = 0; c0 < 4; ++c0)
        cp_async16(ad + swz_a64(a_r, c0), a_code_row + ks * 64 + (c0 << 4), a_valid ? 16 : 0);
    } else if constexpr (TM == 128) {
      const int c0 = tid & 1;
      cp_async16(ad + swz_a64(a_r, c0), a_code_row + ks * 64 + (c0 << 4), a_valid ? 16 : 0);
      cp_async16(ad + swz_a64(a_r, c0 + 2), a_code_row + ks * 64 + ((c0 + 2) << 4), a_valid ? 16 : 0);
    } else if (a_active) {
      const int c0 = tid & 3;
      cp_async16(ad + swz_a64(a_r, c0), a_code_row + ks * 64 + (c0 << 4), a_valid ? 16 : 0);
    }
    if (tid < TM) {
      const int sc_off = (ks & ~1) * 2;
      cp_async4(scd + tid * 4, a_sc_row + sc_off, asc_valid ? 4 : 0);
    }
  };

  load_w(0);
  load_a(0);
  cp_commit();
  if constexpr (C::AS_M == 3 && KS > 1) {
    load_a(1);
    if constexpr (C::KC > 1) load_w(1);
    cp_commit();
  }

  const int wm_n = max(0, min(C::WM_ROWS, n - m_base));
  const int mact = (wm_n + 15) / 16;
  float acc[C::WM_ROWS / 16][2][4];
#pragma unroll
  for (int a = 0; a < C::WM_ROWS / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  const int brow = warp_n * 16 + (lane & 7) + ((lane >> 4) ? 8 : 0);
  const int bhalf = (lane >> 3) & 1;
  const uint8_t* wsc_row = scale_tile + int64_t(warp_n * 16 + grp + t4 * 8) * C::SCALE_ROW;

#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    const int chunk = ks >> 1, sub = ks & 1;
    if constexpr (C::AS_M == 3) {
      if (ks + 1 < KS) cp_wait<1>();
      else cp_wait<0>();
      __syncthreads();
      if (ks + 2 < KS) {
        load_a(ks + 2);
        if (sub == 0 && chunk >= 1 && chunk + 1 < C::KC) load_w(chunk + 1);
        cp_commit();
      }
    } else {
      cp_wait<0>();
      __syncthreads();
      if (sub == 0 && chunk + 1 < C::KC) {
        load_w(chunk + 1);
        cp_commit();
      }
    }

    if (mact > 0) {
      const uint8_t* ws = w_smem + (chunk & 1) * C::W_BYTES;
      const uint8_t* A = a_smem + (ks % C::AS_M) * C::A_BYTES;
      const uint8_t* Asc_sub = asc_smem + (ks % C::AS_M) * C::ASC_BYTES + sub * 2;

      unsigned b0[4], b1[4];
      ldmatrix_x4(b0[0], b0[1], b0[2], b0[3], smem_addr(ws + swz_w128(brow, 4 * sub + bhalf)));
      ldmatrix_x4(b1[0], b1[1], b1[2], b1[3], smem_addr(ws + swz_w128(brow, 4 * sub + 2 + bhalf)));
      const unsigned sfb = (t4 < 2) ? uint32_t(__ldg(reinterpret_cast<const uint16_t*>(wsc_row + ks * 2))) : 0u;

      auto run_mma = [&]<int MT>() {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int mr = m_base + mt * 16;
          unsigned a0[4], a1[4];
          ldmatrix_x4(a0[0], a0[1], a0[2], a0[3], smem_addr(A + swz_a64(mr + (lane & 15), lane >> 4)));
          ldmatrix_x4(a1[0], a1[1], a1[2], a1[3], smem_addr(A + swz_a64(mr + (lane & 15), 2 + (lane >> 4))));
          const unsigned sfa =
              (t4 < 2) ? uint32_t(*reinterpret_cast<const uint16_t*>(Asc_sub + (mr + grp + t4 * 8) * 4)) : 0u;
          mma_mxfp8<0, 0>(acc[mt][0], a0, b0[0], b0[1], sfa, sfb);
          mma_mxfp8<0, 1>(acc[mt][1], a0, b0[2], b0[3], sfa, sfb);
          mma_mxfp8<1, 0>(acc[mt][0], a1, b1[0], b1[1], sfa, sfb);
          mma_mxfp8<1, 1>(acc[mt][1], a1, b1[2], b1[3], sfa, sfb);
        }
      };
      switch (mact) {
        case 1: if constexpr (1 <= C::WM_ROWS / 16) run_mma.template operator()<1>(); break;
        case 2: if constexpr (2 <= C::WM_ROWS / 16) run_mma.template operator()<2>(); break;
        case 3: if constexpr (3 <= C::WM_ROWS / 16) run_mma.template operator()<3>(); break;
        case 4: if constexpr (4 <= C::WM_ROWS / 16) run_mma.template operator()<4>(); break;
        case 5: if constexpr (5 <= C::WM_ROWS / 16) run_mma.template operator()<5>(); break;
        case 6: if constexpr (6 <= C::WM_ROWS / 16) run_mma.template operator()<6>(); break;
        case 7: if constexpr (7 <= C::WM_ROWS / 16) run_mma.template operator()<7>(); break;
        default: run_mma.template operator()<C::WM_ROWS / 16>(); break;
      }
    }

    if constexpr (C::AS_M == 1) {
      if (ks + 1 < KS) {
        __syncthreads();
        load_a(ks + 1);
        cp_commit();
      }
    }
  }
  cp_wait<0>();

#pragma unroll
  for (int mt = 0; mt < C::WM_ROWS / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = m_base + mt * 16 + grp + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp_n * 16 + nt * 8 + t4 * 2;
        const float c0 = acc[mt][nt][2 * half];
        const float c1 = acc[mt][nt][2 * half + 1];
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * I + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          const auto offset = size_t(dst) * kHidden + n0 + cl;
          if constexpr (MODE == Mode::DownPartial)
            *reinterpret_cast<float2*>(reinterpret_cast<float*>(g.out) + offset) = make_float2(c0, c1);
          else
            *reinterpret_cast<uint32_t*>(g.out + offset) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
#endif
}

// =================================================================================================
// BF16 dequant-on-the-fly fallback
// =================================================================================================
template <int FMT, int K, int TM>
struct Bf16Cfg {
  static constexpr int TN = TN_DEFAULT;
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  static constexpr int WK = FMT == FMT_NVFP4 ? 128 : 64;
  static constexpr int R = WK / BK;
  static constexpr int KC = (K + WK - 1) / WK;
  static constexpr int WS = TM <= 32 ? (FMT == FMT_NVFP4 ? 6 : 7) : (FMT == FMT_NVFP4 ? 3 : 4);
  static_assert(R * (WS - 1) >= AS_BF16);
  static constexpr int WCODE = FMT == FMT_NVFP4 ? WK / 2 : WK;
  static constexpr int WSTRIDE = WCODE + 16;
  static constexpr int W_BYTES = TN * WSTRIDE;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int SC_BYTES = TN * SCALE_ROW;
  static constexpr int OFF_W = AS_BF16 * A_BYTES, OFF_SC = OFF_W + WS * W_BYTES;
  static constexpr int OFF_LUT = (OFF_SC + SC_BYTES + 15) / 16 * 16;
  static constexpr int SMEM = OFF_LUT + 64;
  static_assert(K % BK == 0 && SMEM <= 101376);
};

template <int MT, int TM, class After>
__device__ __forceinline__ void mma_step_bf16(float (&acc)[TM / 16][2][4], const uint16_t* A,
                                              const unsigned (&b)[BK / 16][2][2], int lane, After&& after) {
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                  smem_addr(A + swz<KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      mma_bf16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
      mma_bf16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
    }
    after(kk);
  }
}

template <int FMT, Mode MODE, int K, int TM, int I>
__global__ void __launch_bounds__(NT, 1) wgemm_bf16_kernel(const GemmArgs g) {
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * I : kHidden;
  using C = Bf16Cfg<FMT, K, TM>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* a_smem = smem;
  unsigned char* w_smem = smem + C::OFF_W;
  uint8_t* sc_smem = smem + C::OFF_SC;
  float* lut = reinterpret_cast<float*>(smem + C::OFF_LUT);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  constexpr int NTILES_N = NE / C::TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * C::TN;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW;
  if (tid < 16) lut[tid] = kE2M1[tid];

  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + rr]) * K;
    else arow[i] = g.a + size_t(pb + rr) * K;
  }

  auto load_a = [&](int ks) {
    uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % AS_BF16) * C::A_BYTES);
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      const int r = (tid >> 3) + 32 * i, c = tid & 7;
      cp_async16(A + swz<KCH>(r, c), arow[i] + ks * BK + c * 8, r < n ? 16 : 0);
    }
  };
  auto load_w = [&](int chunk) {
    unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
    constexpr int CPR = C::WCODE / 16;
#pragma unroll
    for (int i = 0; i < C::TN * CPR / NT; ++i) {
      const int id = tid + i * NT, row = id / CPR, ch = id % CPR;
      const int start = chunk * C::WCODE + ch * 16;
      const bool valid = K % C::WK == 0 || start < C::CODE_ROW;
      cp_async16(ws + row * C::WSTRIDE + ch * 16, code_tile + int64_t(row) * C::CODE_ROW + (valid ? start : 0),
                 valid ? 16 : 0);
    }
  };
  auto issue = [&](int j) {
    if (j < KS) load_a(j);
    cp_commit();
  };

  for (int id = tid; id < C::SC_BYTES / 4; id += NT)
    cp_async4(sc_smem + id * 4, scale_tile + (id * 4 / C::SCALE_ROW) * int64_t(C::SCALE_ROW) + (id * 4) % C::SCALE_ROW);
#pragma unroll
  for (int j = 0; j < C::WS - 1; ++j)
    if (j < C::KC) load_w(j);
#pragma unroll
  for (int j = 0; j < AS_BF16 - 1; ++j) issue(j);

  const int c = lane & 3;
  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + e) : 1.f;
  const int mact = min((n + 15) / 16, TM / 16);
  float acc[TM / 16][2][4];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  auto scales_of = [&](int ks, float (&sc)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT == FMT_NVFP4) {
        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
        const float2 lo = e4m3x2_to_float2(uint16_t(s4 & 0xffff)), hi = e4m3x2_to_float2(uint16_t(s4 >> 16));
        sc[nt][0] = lo.x, sc[nt][1] = lo.y, sc[nt][2] = hi.x, sc[nt][3] = hi.y;
      } else {
        const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
        sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
        sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
      }
    }
  };
  auto decode = [&](int ks, int kk, const float (&sc)[2][4], unsigned (&b)[BK / 16][2][2]) {
    const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
    const int kc = (ks % C::R) * BK + kk * 16;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      const float s = sc[nt][kk];
      if constexpr (FMT == FMT_NVFP4) {
        const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
        const uint32_t b0 = (wd.x >> (8 * c)) & 0xffu, b1 = (wd.y >> (8 * c)) & 0xffu;
        b[kk][nt][0] = pack_bf16x2((lut[b0 & 0xf] * s) * factor, (lut[b0 >> 4] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[b1 & 0xf] * s) * factor, (lut[b1 >> 4] * s) * factor);
      } else {
        const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
        const uint32_t w0 = (c >> 1) ? q.y : q.x, w1 = (c >> 1) ? q.w : q.z;
        const float2 f0 = e4m3x2_to_float2(uint16_t(w0 >> (16 * (c & 1))));
        const float2 f1 = e4m3x2_to_float2(uint16_t(w1 >> (16 * (c & 1))));
        b[kk][nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
        b[kk][nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
      }
      asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
    }
  };

  unsigned bcur[BK / 16][2][2], bnext[BK / 16][2][2];
#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    cp_wait<AS_BF16 - 2>();
    __syncthreads();
    const int chunk = ks / C::R;
    if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
    issue(ks + AS_BF16 - 1);
    if (ks == 0) {
      float sc0[2][4];
      scales_of(0, sc0);
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) decode(0, kk, sc0, bcur);
    }
    const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % AS_BF16) * C::A_BYTES);
    const bool more = ks + 1 < KS;
    float scn[2][4];
    if (more) scales_of(ks + 1, scn);
    auto after = [&](int kk) {
      if (more) decode(ks + 1, kk, scn, bnext);
    };
    switch (mact) {
      case 1: if constexpr (1 <= TM / 16) mma_step_bf16<1, TM>(acc, A, bcur, lane, after); break;
      case 2: if constexpr (2 <= TM / 16) mma_step_bf16<2, TM>(acc, A, bcur, lane, after); break;
      case 3: if constexpr (3 <= TM / 16) mma_step_bf16<3, TM>(acc, A, bcur, lane, after); break;
      case 4: if constexpr (4 <= TM / 16) mma_step_bf16<4, TM>(acc, A, bcur, lane, after); break;
      case 5: if constexpr (5 <= TM / 16) mma_step_bf16<5, TM>(acc, A, bcur, lane, after); break;
      case 6: if constexpr (6 <= TM / 16) mma_step_bf16<6, TM>(acc, A, bcur, lane, after); break;
      case 7: if constexpr (7 <= TM / 16) mma_step_bf16<7, TM>(acc, A, bcur, lane, after); break;
      default: mma_step_bf16<TM / 16, TM>(acc, A, bcur, lane, after); break;
    }
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) bcur[kk][nt][0] = bnext[kk][nt][0], bcur[kk][nt][1] = bnext[kk][nt][1];
  }
  cp_wait<0>();

#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp * 16 + nt * 8 + c * 2;
        const float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * I + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          const auto offset = size_t(dst) * kHidden + n0 + cl;
          if constexpr (MODE == Mode::DownPartial)
            *reinterpret_cast<float2*>(reinterpret_cast<float*>(g.out) + offset) = make_float2(c0, c1);
          else
            *reinterpret_cast<uint32_t*>(g.out + offset) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, Mode MODE, int K, int TM, int TN_, int I>
void launch_native(const GemmArgs& g, int a_rows, int row_tiles, cudaStream_t stream) {
  constexpr int NE = MODE == Mode::Up ? 2 * I : kHidden;
  const int grid = (NE / TN_) * row_tiles;
  if constexpr (FMT == FMT_NVFP4) {
    constexpr int qthreads = ((K / 16) + 31) / 32 * 32;
    quantize_nvfp4_rows_kernel<K><<<a_rows, qthreads, 0, stream>>>(g.a, g.w.a_codes, g.w.a_scales, g.w.a_factors);
    constexpr int SMEM = Nvfp4Cfg<K, TM, TN_>::SMEM;
    static const cudaError_t attr = cudaFuncSetAttribute(wgemm_nvfp4_kernel<MODE, K, TM, TN_, I>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm nvfp4: shared memory attribute failed");
    wgemm_nvfp4_kernel<MODE, K, TM, TN_, I><<<grid, NT, SMEM, stream>>>(g);
  } else {
    constexpr int qthreads = ((K / 32) + 31) / 32 * 32;
    quantize_mxfp8_rows_kernel<K><<<a_rows, qthreads, 0, stream>>>(g.a, g.w.a_codes, g.w.a_scales);
    constexpr int SMEM = Mxfp8Cfg<K, TM, TN_>::SMEM;
    static const cudaError_t attr = cudaFuncSetAttribute(wgemm_mxfp8_kernel<MODE, K, TM, TN_, I>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm mxfp8: shared memory attribute failed");
    wgemm_mxfp8_kernel<MODE, K, TM, TN_, I><<<grid, NT, SMEM, stream>>>(g);
  }
}

template <int FMT, Mode MODE, int K, int I = kInter>
void launch_rows(const GemmArgs& g, int a_rows, int tm, int row_tiles, cudaStream_t stream) {
  const MmaBackend backend = select_backend();
  const bool has_scratch = (g.w.a_codes != nullptr && g.w.a_scales != nullptr &&
                            (FMT == FMT_MXFP8 || g.w.a_factors != nullptr));
  const bool use_native = has_scratch &&
      ((FMT == FMT_NVFP4 && (backend == MmaBackend::Native || backend == MmaBackend::Nvfp4Only)) ||
       (FMT == FMT_MXFP8 && (backend == MmaBackend::Native || backend == MmaBackend::Mxfp8Only)));
  if (use_native) {
    if (tm == 32) launch_native<FMT, MODE, K, 32, 128, I>(g, a_rows, row_tiles, stream);
    else if (tm == 256) launch_native<FMT, MODE, K, 256, 64, I>(g, a_rows, row_tiles, stream);
    else launch_native<FMT, MODE, K, 128, 128, I>(g, a_rows, row_tiles, stream);
    return;
  }
  constexpr int NE = MODE == Mode::Up ? 2 * I : kHidden;
  const int grid = (NE / TN_DEFAULT) * row_tiles;
  if (tm == 32) {
    constexpr int SMEM = Bf16Cfg<FMT, K, 32>::SMEM;
    static const cudaError_t attr =
        cudaFuncSetAttribute(wgemm_bf16_kernel<FMT, MODE, K, 32, I>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm bf16: shared memory attribute failed");
    wgemm_bf16_kernel<FMT, MODE, K, 32, I><<<grid, NT, SMEM, stream>>>(g);
  } else {
    constexpr int SMEM = Bf16Cfg<FMT, K, 128>::SMEM;
    static const cudaError_t attr =
        cudaFuncSetAttribute(wgemm_bf16_kernel<FMT, MODE, K, 128, I>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm bf16: shared memory attribute failed");
    wgemm_bf16_kernel<FMT, MODE, K, 128, I><<<grid, NT, SMEM, stream>>>(g);
  }
}

GemmArgs expert_args(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a,
                     const MoeWork& w, __nv_bfloat16* out) {
  GemmArgs g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

}  // namespace

void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w13, e.w13_scale, e.w13_factor, x, w, act);
  auto dispatch = [&]<int I>() {
    const int rows = moe_tile_rows(T), tiles = moe_max_tiles(T);
    if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Up, kHidden, I>(g, T, rows, tiles, stream);
    else launch_rows<FMT_MXFP8, Mode::Up, kHidden, I>(g, T, rows, tiles, stream);
  };
  if (e.inter == kInter) dispatch.template operator()<kInter>();
  else if (e.inter == kInter / 2) dispatch.template operator()<kInter / 2>();
  else throw std::invalid_argument("prefill MoE: unsupported intermediate width");
}

void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, down);
  const int a_rows = T * kTopK;
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Down, kInter>(g, a_rows, moe_tile_rows(T), moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Down, kInter>(g, a_rows, moe_tile_rows(T), moe_max_tiles(T), stream);
}

void moe_down_partial(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w,
                       float* down, cudaStream_t stream) {
  constexpr int I = kInter / 2;
  if (e.inter != I) throw std::invalid_argument("prefill MoE partial: expected a TP=2 expert shard");
  const GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, reinterpret_cast<__nv_bfloat16*>(down));
  const int a_rows = T * kTopK;
  const int rows = moe_tile_rows(T), tiles = moe_max_tiles(T);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::DownPartial, I>(g, a_rows, rows, tiles, stream);
  else launch_rows<FMT_MXFP8, Mode::DownPartial, I>(g, a_rows, rows, tiles, stream);
}

}  // namespace ingine::kernels::prefill
