#include "kernels/batch/dense.h"

#include "kernels/common/l2.cuh"
#include "kernels/common/nvfp4.cuh"
#include "kernels/common/pico12.cuh"

#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace ingine::kernels::batch {
namespace {

constexpr int H = prefill::kHidden;
constexpr int ROW_T = H / 4;  // 576 threads, float4 each
constexpr int ROW_W = ROW_T / 32;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  return std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// ---- the head: L[M, N] = A[M, K] . B[N, K]^T (FP32), and kb::sample choice of a token per row
constexpr int BK = 64;

struct HeadArgs {
  const __nv_bfloat16* A;
  const void* B;
  float* C;  // nullptr: L not stored
  int lda, ldb, ldc, M, N, K;
  const float* temps;
  std::uint64_t seed;
  int2* partials;  // [M, ntiles]: (key bits, v)
  int ntiles;
  int v_begin;
  const std::uint64_t* d_seed;
  pico12::DeviceView pico;
};

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ std::uint64_t row_key(std::uint64_t seed, int row) {
  return mix64(seed ^ (std::uint64_t(row) * 0xD1B54A32D192ED03ull));
}
__device__ __forceinline__ float pick_key(float l, bool greedy, float inv_t, std::uint64_t rkey, int v) {
  float key = 30.f * tanhf(bf16r(l) / 30.f);
  if (!greedy) {
    const float u = (float(mix64(rkey ^ std::uint64_t(v)) >> 41) + 0.5f) * (1.f / 8388608.f);
    key = key * inv_t - logf(-logf(u));
  }
  return key;
}
__device__ __forceinline__ bool better(float a, int av, float b, int bv) { return a > b || (a == b && av < bv); }
template <int LANES>
__device__ __forceinline__ void lanes_best(float& k, int& v) {
#pragma unroll
  for (int o = 1; o < LANES; o <<= 1) {
    const float ok = __shfl_xor_sync(0xffffffffu, k, o);
    const int ov = __shfl_xor_sync(0xffffffffu, v, o);
    if (better(ok, ov, k, v)) k = ok, v = ov;
  }
}

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async4(std::uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MF_, int NF_, int WM_, int WN_, int STAGES_>
struct MmaCfg {
  static constexpr int MF = MF_, NF = NF_, WM = WM_, WN = WN_, STAGES = STAGES_;
  static constexpr int TM = WM * MF * 16, TN = WN * NF * 8, NT = WM * WN * 32;
  static constexpr int A_BYTES = TM * BK * 2, STAGE = (TM + TN) * BK * 2, SMEM = STAGES * STAGE;
  static_assert(NF % 2 == 0 && SMEM <= 96 * 1024);
};

template <class C, bool Packed, bool Escapes>
__global__ void __launch_bounds__(C::NT) head_kernel(const HeadArgs p) {
  extern __shared__ __align__(128) std::uint8_t smem[];
  constexpr int MF = C::MF, NF = C::NF, TM = C::TM, TN = C::TN, NT = C::NT, STAGES = C::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int mtile = blockIdx.x, ntile = blockIdx.y;
  const __nv_bfloat16* A = p.A + std::size_t(mtile) * TM * p.lda;
  const int m_rows = min(TM, p.M - mtile * TM), n_rows = min(TN, p.N - ntile * TN);
  const int nk = p.K / BK;

  auto load_a = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE;
    for (int i = tid; i < TM * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < m_rows;
      cp_async16(smem_addr(base + swz(r, c)), A + std::size_t(ok ? r : 0) * p.lda + kt * BK + c * 8, ok);
    }
  };
  auto load_b = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE + C::A_BYTES;
    for (int i = tid; i < TN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < n_rows;
      const int col = kt * BK + c * 8;
      if constexpr (Packed) {
        uint4 bf16{};
        if (ok) {
          const std::size_t row = std::size_t(ntile) * TN + r;
          const auto* src = static_cast<const std::uint8_t*>(p.B) + row * pico12::row_bytes(p.K);
          const auto chunk = pico12::load8<l2::Evict::none>(src, p.K, col, 0);
          if constexpr (Escapes) {
            if (p.K > pico12::kCompactCols) bf16 = pico12::restore8<true>(chunk, p.pico, int(row), col);
            else bf16 = pico12::restore8<false>(chunk, p.pico, int(row), col);
          } else {
            bf16 = pico12::decode8(chunk, p.pico);
          }
        }
        *reinterpret_cast<uint4*>(base + swz(r, c)) = bf16;
      } else {
        const auto* src = static_cast<const __nv_bfloat16*>(p.B) + (std::size_t(ntile) * TN + (ok ? r : 0)) * p.ldb;
        cp_async16(smem_addr(base + swz(r, c)), src + col, ok);
      }
    }
  };

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nk) load_b(s, s);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_a(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < nk) load_a(next % STAGES, next), load_b(next % STAGES, next);
    cp_commit();
    const std::uint32_t a_base = smem_addr(smem + (kt % STAGES) * C::STAGE), b_base = a_base + C::A_BYTES;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      std::uint32_t af[MF][4], bf[NF][2];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ldsm_x4(af[i], a_base + swz(warp_m * MF * 16 + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < NF / 2; ++j) {
        std::uint32_t r[4];
        ldsm_x4(r, b_base + swz(warp_n * NF * 8 + j * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
        bf[2 * j][0] = r[0], bf[2 * j][1] = r[1], bf[2 * j + 1][0] = r[2], bf[2 * j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
  }
  cp_wait<0>();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  const std::uint64_t seed = p.d_seed ? *p.d_seed : p.seed;
  float best_k[MF][2];
  int best_v[MF][2];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_m * MF * 16 + i * 16 + (lane >> 2) + h * 8, grow = mtile * TM + row;
      best_k[i][h] = -INFINITY, best_v[i][h] = 0x7fffffff;
      if (row >= m_rows) continue;
      const float t = p.temps[grow];
      const bool greedy = !(t > 0.f);
      const float inv_t = greedy ? 1.f : 1.f / t;
      const std::uint64_t rkey = row_key(seed, grow);
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int col = warp_n * NF * 8 + j * 8 + (lane & 3) * 2;
        if (col >= n_rows) continue;
        if (p.C)
          *reinterpret_cast<float2*>(p.C + std::size_t(grow) * p.ldc + ntile * TN + col) =
              make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          if (col + e >= n_rows) continue;
          const int v = p.v_begin + ntile * TN + col + e;
          const float key = pick_key(acc[i][j][2 * h + e], greedy, inv_t, rkey, v);
          if (better(key, v, best_k[i][h], best_v[i][h])) best_k[i][h] = key, best_v[i][h] = v;
        }
      }
    }
  __syncthreads();
  auto red_k = reinterpret_cast<float(*)[C::WN]>(smem);
  auto red_v = reinterpret_cast<int(*)[C::WN]>(smem + TM * C::WN * sizeof(float));
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      lanes_best<4>(best_k[i][h], best_v[i][h]);
      const int row = warp_m * MF * 16 + i * 16 + (lane >> 2) + h * 8;
      if ((lane & 3) == 0) red_k[row][warp_n] = best_k[i][h], red_v[row][warp_n] = best_v[i][h];
    }
  __syncthreads();
  for (int row = tid; row < m_rows; row += NT) {
    float k = red_k[row][0];
    int v = red_v[row][0];
    for (int w = 1; w < C::WN; ++w)
      if (better(red_k[row][w], red_v[row][w], k, v)) k = red_k[row][w], v = red_v[row][w];
    p.partials[std::size_t(mtile * TM + row) * p.ntiles + ntile] = make_int2(__float_as_int(k), v);
  }
}

// =================================================================================================
// NVFP4 fused head + sample (M <= 8 split-K streaming kernel + M > 8 tiled NVFP4-decode kernel)
// =================================================================================================
constexpr int NVFP4_SMALL_TILE = 16;
constexpr int NVFP4_SMALL_SPLIT = 6;
constexpr int NVFP4_TILED_TN = 128;

struct HeadNvfp4Args {
  const __nv_bfloat16* A;
  kernels::head::Nvfp4Matrix w;
  float* C;
  int lda, ldc, M;
  const float* temps;
  std::uint64_t seed;
  int2* partials;
  int ntiles;
  const std::uint64_t* d_seed;
};

__device__ __forceinline__ std::uint32_t decode_e2m1_byte_scaled(std::uint8_t code_byte, __half2 sc_h2, float factor) {
  std::uint32_t h;
  asm("{\n"
      ".reg .b8 b0;\n"
      "cvt.u8.u32 b0, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "}"
      : "=r"(h)
      : "r"(std::uint32_t(code_byte)));
  const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h), sc_h2);
  const float2 v = __half22float2(cs);
  return nvfp4::pack_bf16(v.x * factor, v.y * factor);
}

__global__ void __launch_bounds__(NVFP4_SMALL_SPLIT * 32)
    head_nvfp4_small_kernel(const HeadNvfp4Args p) {
  __shared__ float partial[NVFP4_SMALL_SPLIT][32][4];
  __shared__ float sk[8][NVFP4_SMALL_TILE];
  __shared__ int sv[8][NVFP4_SMALL_TILE];
  constexpr int Chunks = H / (32 * 4 * NVFP4_SMALL_SPLIT);  // 2304 / 768 = 3
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * NVFP4_SMALL_TILE + (lane >> 2);
  const int token = lane >> 2, t = lane & 3;
  const std::uint64_t policy = l2::policy<l2::Evict::first>();

  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float sum[4] = {};
#pragma unroll
  for (int step = 0; step < Chunks; ++step) {
    const int c = (warp * Chunks + step) * 4 + t;
    uint4 clo{}, chi{};
    std::uint16_t slo = 0, shi = 0;
    if (row < p.w.rows) {
      clo = l2::ld_stream<l2::Evict::first>(p.w.codes + std::size_t(row) * (H / 2) + 16 * c, policy);
      slo = __ldg(reinterpret_cast<const std::uint16_t*>(p.w.scales + std::size_t(row) * (H / 16) + 2 * c));
    }
    if (row + 8 < p.w.rows) {
      chi = l2::ld_stream<l2::Evict::first>(p.w.codes + std::size_t(row + 8) * (H / 2) + 16 * c, policy);
      shi = __ldg(reinterpret_cast<const std::uint16_t*>(p.w.scales + std::size_t(row + 8) * (H / 16) + 2 * c));
    }
    std::uint32_t a[16], b[16];
    nvfp4::decode(clo, slo, p.w.outer_scale, a);
    nvfp4::decode(chi, shi, p.w.outer_scale, b);
    const auto* xp = reinterpret_cast<const uint4*>(p.A + (token < p.M ? std::size_t(token) * p.lda + 32 * c : 0));
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint4 in{};
      if (token < p.M) in = xp[q];
      nvfp4::mma_bf16(sum, a[4 * q], b[4 * q], a[4 * q + 1], b[4 * q + 1], in.x, in.y);
      nvfp4::mma_bf16(sum, a[4 * q + 2], b[4 * q + 2], a[4 * q + 3], b[4 * q + 3], in.z, in.w);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) partial[warp][lane][i] = sum[i];
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  if (warp == 0) {
    const std::uint64_t seed = p.d_seed ? *p.d_seed : p.seed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int m = 2 * t + (i & 1), r = row + 8 * (i >> 1);
      if (m < p.M) {
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < NVFP4_SMALL_SPLIT; ++j) dot += partial[j][lane][i];
        float key = -INFINITY;
        int v = 0x7fffffff;
        if (r < p.w.rows) {
          if (p.C) p.C[std::size_t(m) * p.ldc + r] = dot;
          const float temp = p.temps[m];
          const bool greedy = !(temp > 0.f);
          const float inv_t = greedy ? 1.f : 1.f / temp;
          v = p.w.v_begin + r;
          key = pick_key(dot, greedy, inv_t, row_key(seed, m), v);
        }
        sk[m][r - blockIdx.x * NVFP4_SMALL_TILE] = key;
        sv[m][r - blockIdx.x * NVFP4_SMALL_TILE] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < p.M) {
    const int m = threadIdx.x;
    float bk = sk[m][0];
    int bv = sv[m][0];
#pragma unroll
    for (int i = 1; i < NVFP4_SMALL_TILE; ++i)
      if (better(sk[m][i], sv[m][i], bk, bv)) bk = sk[m][i], bv = sv[m][i];
    p.partials[std::size_t(m) * p.ntiles + blockIdx.x] = make_int2(__float_as_int(bk), bv);
  }
}

template <int MF_, int NF_, int WM_, int WN_, int STAGES_>
struct Nvfp4HeadCfg {
  static constexpr int MF = MF_, NF = NF_, WM = WM_, WN = WN_, STAGES = STAGES_;
  static constexpr int TM = WM * MF * 16, TN = WN * NF * 8, NT = WM * WN * 32;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int W_STRIDE = 48;
  static constexpr int W_BYTES = TN * W_STRIDE;
  static constexpr int WSC_BYTES = TN * 4;
  static constexpr int STAGE = A_BYTES + W_BYTES + WSC_BYTES;
  static constexpr int SMEM = STAGES * STAGE;
  static_assert(TN == NVFP4_TILED_TN && SMEM <= 48 * 1024);
};

template <class C>
__global__ void __launch_bounds__(C::NT, 2) head_nvfp4_tiled_kernel(const HeadNvfp4Args p) {
  extern __shared__ __align__(128) std::uint8_t smem[];
  constexpr int MF = C::MF, NF = C::NF, TM = C::TM, TN = C::TN, NT = C::NT, STAGES = C::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int grp = lane >> 2, t4 = lane & 3;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int mtile = blockIdx.x, ntile = blockIdx.y;
  const __nv_bfloat16* A = p.A + std::size_t(mtile) * TM * p.lda;
  const std::uint8_t* Wc = p.w.codes + std::size_t(ntile) * TN * (H / 2);
  const std::uint8_t* Ws = p.w.scales + std::size_t(ntile) * TN * (H / 16);
  const int m_rows = min(TM, p.M - mtile * TM), n_rows = min(TN, p.w.rows - ntile * TN);
  constexpr int nk = H / BK;

  auto load_a = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE;
    for (int i = tid; i < TM * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < m_rows;
      cp_async16(smem_addr(base + swz(r, c)), A + std::size_t(ok ? r : 0) * p.lda + kt * BK + c * 8, ok);
    }
  };
  auto load_w = [&](int stage, int kt) {
    std::uint8_t* wbase = smem + stage * C::STAGE + C::A_BYTES;
    std::uint8_t* scbase = wbase + C::W_BYTES;
    {
      const int r = tid >> 1, c16 = tid & 1;
      const bool ok = r < n_rows;
      cp_async16(smem_addr(wbase + r * C::W_STRIDE + (c16 << 4)),
                 Wc + std::size_t(ok ? r : 0) * (H / 2) + kt * 32 + (c16 << 4), ok);
    }
    if (tid < TN) {
      const bool ok = tid < n_rows;
      cp_async4(smem_addr(scbase + tid * 4), Ws + std::size_t(ok ? tid : 0) * (H / 16) + kt * 4, ok);
    }
  };

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) load_w(s, s);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    load_a(s, s);
    cp_commit();
  }

  const float factor = p.w.outer_scale;
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < nk) load_a(next % STAGES, next), load_w(next % STAGES, next);
    cp_commit();

    const std::uint8_t* stage_ptr = smem + (kt % STAGES) * C::STAGE;
    const std::uint32_t a_base = smem_addr(stage_ptr);
    const std::uint8_t* w_base = stage_ptr + C::A_BYTES;
    const std::uint8_t* sc_base = w_base + C::W_BYTES;

    __half2 sc_lo[NF], sc_hi[NF];
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const int row_n = warp_n * NF * 8 + j * 8 + grp;
      const std::uint32_t s4 = *reinterpret_cast<const std::uint32_t*>(sc_base + row_n * 4);
      const std::uint32_t hlo = nvfp4::f16x2_from_e4m3x2(std::uint16_t(s4 & 0xffffu));
      const std::uint32_t hhi = nvfp4::f16x2_from_e4m3x2(std::uint16_t(s4 >> 16));
      sc_lo[j] = *reinterpret_cast<const __half2*>(&hlo);
      sc_hi[j] = *reinterpret_cast<const __half2*>(&hhi);
    }

#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      std::uint32_t af[MF][4];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ldsm_x4(af[i], a_base + swz(warp_m * MF * 16 + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int row_n = warp_n * NF * 8 + j * 8 + grp;
        const std::uint8_t* wr = w_base + row_n * C::W_STRIDE + ks * 8;
        const __half2 sc_h2 = (ks == 0) ? __low2half2(sc_lo[j])
                            : (ks == 1) ? __high2half2(sc_lo[j])
                            : (ks == 2) ? __low2half2(sc_hi[j])
                                        : __high2half2(sc_hi[j]);
        const std::uint32_t b0 = decode_e2m1_byte_scaled(wr[t4], sc_h2, factor);
        const std::uint32_t b1 = decode_e2m1_byte_scaled(wr[4 + t4], sc_h2, factor);
#pragma unroll
        for (int i = 0; i < MF; ++i) mma_bf16(acc[i][j], af[i], b0, b1);
      }
    }
  }
  cp_wait<0>();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  const std::uint64_t seed = p.d_seed ? *p.d_seed : p.seed;
  float best_k[MF][2];
  int best_v[MF][2];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_m * MF * 16 + i * 16 + grp + h * 8, grow = mtile * TM + row;
      best_k[i][h] = -INFINITY, best_v[i][h] = 0x7fffffff;
      if (row >= m_rows) continue;
      const float t = p.temps[grow];
      const bool greedy = !(t > 0.f);
      const float inv_t = greedy ? 1.f : 1.f / t;
      const std::uint64_t rkey = row_key(seed, grow);
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int col = warp_n * NF * 8 + j * 8 + t4 * 2;
        if (col >= n_rows) continue;
        if (p.C) {
          if (col + 1 < n_rows)
            *reinterpret_cast<float2*>(p.C + std::size_t(grow) * p.ldc + ntile * TN + col) =
                make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          else
            p.C[std::size_t(grow) * p.ldc + ntile * TN + col] = acc[i][j][2 * h];
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          if (col + e >= n_rows) continue;
          const int v = p.w.v_begin + ntile * TN + col + e;
          const float key = pick_key(acc[i][j][2 * h + e], greedy, inv_t, rkey, v);
          if (better(key, v, best_k[i][h], best_v[i][h])) best_k[i][h] = key, best_v[i][h] = v;
        }
      }
    }
  __syncthreads();
  auto red_k = reinterpret_cast<float(*)[C::WN]>(smem);
  auto red_v = reinterpret_cast<int(*)[C::WN]>(smem + TM * C::WN * sizeof(float));
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      lanes_best<4>(best_k[i][h], best_v[i][h]);
      const int row = warp_m * MF * 16 + i * 16 + grp + h * 8;
      if (t4 == 0) red_k[row][warp_n] = best_k[i][h], red_v[row][warp_n] = best_v[i][h];
    }
  __syncthreads();
  for (int row = tid; row < m_rows; row += NT) {
    float k = red_k[row][0];
    int v = red_v[row][0];
    for (int w = 1; w < C::WN; ++w)
      if (better(red_k[row][w], red_v[row][w], k, v)) k = red_k[row][w], v = red_v[row][w];
    p.partials[std::size_t(mtile * TM + row) * p.ntiles + ntile] = make_int2(__float_as_int(k), v);
  }
}

constexpr int PICK_T = 256;
__global__ void __launch_bounds__(PICK_T) head_pick_kernel(const int2* __restrict__ partials, int ntiles,
                                                           std::int32_t* __restrict__ out,
                                                           int2* __restrict__ cand_out) {
  const int2* const p = partials + std::size_t(blockIdx.x) * ntiles;
  float k = -INFINITY;
  int v = 0x7fffffff;
  for (int i = threadIdx.x; i < ntiles; i += PICK_T) {
    const int2 x = p[i];
    if (better(__int_as_float(x.x), x.y, k, v)) k = __int_as_float(x.x), v = x.y;
  }
  lanes_best<32>(k, v);
  __shared__ float s_k[PICK_T / 32];
  __shared__ int s_v[PICK_T / 32];
  if ((threadIdx.x & 31) == 0) s_k[threadIdx.x >> 5] = k, s_v[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < PICK_T / 32; ++w)
      if (better(s_k[w], s_v[w], s_k[0], s_v[0])) s_k[0] = s_k[w], s_v[0] = s_v[w];
    if (cand_out) cand_out[blockIdx.x] = make_int2(__float_as_int(s_k[0]), s_v[0]);
    if (out) out[blockIdx.x] = s_v[0];
  }
}

constexpr int HEAD_TN = 128;
using Cfg64 = MmaCfg<2, 4, 2, 4, 2>;
using Cfg32 = MmaCfg<1, 4, 2, 4, 2>;
static_assert(Cfg64::TN == HEAD_TN && Cfg32::TN == HEAD_TN);

using Nvfp4Cfg32 = Nvfp4HeadCfg<2, 2, 1, 8, 2>;
using Nvfp4Cfg64 = Nvfp4HeadCfg<2, 4, 2, 4, 2>;

template <class C, bool Packed, bool Escapes>
void launch(const HeadArgs& a, cudaStream_t stream) {
  static const bool attr = [] {
    return cudaFuncSetAttribute(head_kernel<C, Packed, Escapes>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess;
  }();
  if (!attr) throw std::runtime_error("batch head_sample: shared memory attribute");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((a.M + C::TM - 1) / C::TM, a.ntiles);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl;
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &pdl;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, head_kernel<C, Packed, Escapes>, a) != cudaSuccess)
    throw std::runtime_error("batch head_sample: launch failed");
}

struct HeadNvfp4TopkArgs {
  const __nv_bfloat16* A;
  kernels::head::Nvfp4Matrix w;
  int lda, M, ntiles;
  kernels::head::Candidate* partials;  // [M, ntiles, 16]
};

template <class C>
struct Nvfp4TopkSmem {
  static constexpr int TILE_BYTES = C::TM * C::TN * int(sizeof(float));
  static constexpr int BYTES = C::SMEM > TILE_BYTES ? C::SMEM : TILE_BYTES;
};

template <class C>
__global__ void __launch_bounds__(C::NT, 2) head_nvfp4_topk16_kernel(const HeadNvfp4TopkArgs p) {
  extern __shared__ __align__(128) std::uint8_t smem[];
  constexpr int MF = C::MF, NF = C::NF, TM = C::TM, TN = C::TN, NT = C::NT, STAGES = C::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int grp = lane >> 2, t4 = lane & 3;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int mtile = blockIdx.x, ntile = blockIdx.y;
  const __nv_bfloat16* A = p.A + std::size_t(mtile) * TM * p.lda;
  const std::uint8_t* Wc = p.w.codes + std::size_t(ntile) * TN * (H / 2);
  const std::uint8_t* Ws = p.w.scales + std::size_t(ntile) * TN * (H / 16);
  const int m_rows = min(TM, p.M - mtile * TM), n_rows = min(TN, p.w.rows - ntile * TN);
  constexpr int nk = H / BK;

  auto load_a = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE;
    for (int i = tid; i < TM * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < m_rows;
      cp_async16(smem_addr(base + swz(r, c)), A + std::size_t(ok ? r : 0) * p.lda + kt * BK + c * 8, ok);
    }
  };
  auto load_w = [&](int stage, int kt) {
    std::uint8_t* wbase = smem + stage * C::STAGE + C::A_BYTES;
    std::uint8_t* scbase = wbase + C::W_BYTES;
    {
      const int r = tid >> 1, c16 = tid & 1;
      const bool ok = r < n_rows;
      cp_async16(smem_addr(wbase + r * C::W_STRIDE + (c16 << 4)),
                 Wc + std::size_t(ok ? r : 0) * (H / 2) + kt * 32 + (c16 << 4), ok);
    }
    if (tid < TN) {
      const bool ok = tid < n_rows;
      cp_async4(smem_addr(scbase + tid * 4), Ws + std::size_t(ok ? tid : 0) * (H / 16) + kt * 4, ok);
    }
  };

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) load_w(s, s);
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    load_a(s, s);
    cp_commit();
  }

  const float factor = p.w.outer_scale;
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < nk) load_a(next % STAGES, next), load_w(next % STAGES, next);
    cp_commit();

    const std::uint8_t* stage_ptr = smem + (kt % STAGES) * C::STAGE;
    const std::uint32_t a_base = smem_addr(stage_ptr);
    const std::uint8_t* w_base = stage_ptr + C::A_BYTES;
    const std::uint8_t* sc_base = w_base + C::W_BYTES;

    __half2 sc_lo[NF], sc_hi[NF];
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const int row_n = warp_n * NF * 8 + j * 8 + grp;
      const std::uint32_t s4 = *reinterpret_cast<const std::uint32_t*>(sc_base + row_n * 4);
      const std::uint32_t hlo = nvfp4::f16x2_from_e4m3x2(std::uint16_t(s4 & 0xffffu));
      const std::uint32_t hhi = nvfp4::f16x2_from_e4m3x2(std::uint16_t(s4 >> 16));
      sc_lo[j] = *reinterpret_cast<const __half2*>(&hlo);
      sc_hi[j] = *reinterpret_cast<const __half2*>(&hhi);
    }

#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      std::uint32_t af[MF][4];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ldsm_x4(af[i], a_base + swz(warp_m * MF * 16 + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int row_n = warp_n * NF * 8 + j * 8 + grp;
        const std::uint8_t* wr = w_base + row_n * C::W_STRIDE + ks * 8;
        const __half2 sc_h2 = (ks == 0) ? __low2half2(sc_lo[j])
                            : (ks == 1) ? __high2half2(sc_lo[j])
                            : (ks == 2) ? __low2half2(sc_hi[j])
                                        : __high2half2(sc_hi[j]);
        const std::uint32_t b0 = decode_e2m1_byte_scaled(wr[t4], sc_h2, factor);
        const std::uint32_t b1 = decode_e2m1_byte_scaled(wr[4 + t4], sc_h2, factor);
#pragma unroll
        for (int i = 0; i < MF; ++i) mma_bf16(acc[i][j], af[i], b0, b1);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();

  auto tile_logits = reinterpret_cast<float (*)[TN]>(smem);
#pragma unroll
  for (int i = 0; i < MF; ++i) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_m * MF * 16 + i * 16 + grp + h * 8;
      if (row < m_rows) {
#pragma unroll
        for (int j = 0; j < NF; ++j) {
          const int col = warp_n * NF * 8 + j * 8 + t4 * 2;
          tile_logits[row][col] = (col < n_rows) ? acc[i][j][2 * h] : -INFINITY;
          tile_logits[row][col + 1] = (col + 1 < n_rows) ? acc[i][j][2 * h + 1] : -INFINITY;
        }
      }
    }
  }
  __syncthreads();

  constexpr int NW = NT / 32;
  for (int r = warp; r < m_rows; r += NW) {
    const int grow = mtile * TM + r;
    float vals[4];
    int ids[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = c * 32 + lane;
      vals[c] = tile_logits[r][col];
      ids[c] = (col < n_rows) ? (p.w.v_begin + ntile * TN + col) : 0x7fffffff;
    }
    auto* out_row = p.partials + (std::size_t(grow) * p.ntiles + ntile) * 16;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      float bv = vals[0];
      int bi = ids[0];
      int bs = 0;
#pragma unroll
      for (int c = 1; c < 4; ++c) {
        if (better(vals[c], ids[c], bv, bi)) {
          bv = vals[c];
          bi = ids[c];
          bs = c;
        }
      }
      float wv = bv;
      int wi = bi;
      lanes_best<32>(wv, wi);
      if (lane == 0) out_row[k] = kernels::head::Candidate{wv, wi};
      if (bi == wi && bv == wv) vals[bs] = -INFINITY;
    }
  }
}

constexpr int TOPK_RED_THREADS = 128;
constexpr int TOPK_GROUP_TILES = 32;  // 32 * 16 = 512 candidates per stage-1 CTA

template <int MAX_PER_THREAD, bool FINAL_OUT>
__global__ void __launch_bounds__(TOPK_RED_THREADS)
    reduce_topk16_kernel(const kernels::head::Candidate* __restrict__ in, int row_stride_cands, int cands_per_group,
                         kernels::head::Candidate* __restrict__ stage_out, float* __restrict__ out_vals,
                         std::int32_t* __restrict__ out_ids) {
  constexpr int NW = TOPK_RED_THREADS / 32;
  __shared__ float sw_v[NW];
  __shared__ int sw_i[NW];
  const int group = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int start = group * cands_per_group;
  const int count = min(cands_per_group, row_stride_cands - start);
  const auto* src = in + std::size_t(row) * row_stride_cands + start;

  float vals[MAX_PER_THREAD];
  int ids[MAX_PER_THREAD];
#pragma unroll
  for (int c = 0; c < MAX_PER_THREAD; ++c) {
    const int idx = c * TOPK_RED_THREADS + tid;
    if (idx < count) {
      const auto cand = src[idx];
      vals[c] = cand.val;
      ids[c] = cand.global_id;
    } else {
      vals[c] = -INFINITY;
      ids[c] = 0x7fffffff;
    }
  }

  for (int k = 0; k < 16; ++k) {
    float bv = vals[0];
    int bi = ids[0];
    int bs = 0;
#pragma unroll
    for (int c = 1; c < MAX_PER_THREAD; ++c) {
      if (better(vals[c], ids[c], bv, bi)) {
        bv = vals[c];
        bi = ids[c];
        bs = c;
      }
    }
    float wv = bv;
    int wi = bi;
    lanes_best<32>(wv, wi);
    if (lane == 0) {
      sw_v[warp] = wv;
      sw_i[warp] = wi;
    }
    __syncthreads();
    if (tid == 0) {
      float fv = sw_v[0];
      int fi = sw_i[0];
#pragma unroll
      for (int w = 1; w < NW; ++w) {
        if (better(sw_v[w], sw_i[w], fv, fi)) {
          fv = sw_v[w];
          fi = sw_i[w];
        }
      }
      sw_v[0] = fv;
      sw_i[0] = fi;
      if constexpr (FINAL_OUT) {
        out_vals[std::size_t(row) * 16 + k] = fv;
        out_ids[std::size_t(row) * 16 + k] = fi;
      } else {
        stage_out[(std::size_t(row) * gridDim.x + group) * 16 + k] = kernels::head::Candidate{fv, fi};
      }
    }
    __syncthreads();
    const float fv = sw_v[0];
    const int fi = sw_i[0];
    if (bi == fi && bv == fv) vals[bs] = -INFINITY;
  }
}

template <class C>
void launch_nvfp4_topk16(const HeadNvfp4TopkArgs& a, cudaStream_t stream) {
  constexpr int SMEM = Nvfp4TopkSmem<C>::BYTES;
  static const bool attr = [] {
    return cudaFuncSetAttribute(head_nvfp4_topk16_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) ==
           cudaSuccess;
  }();
  if (!attr) throw std::runtime_error("batch head_topk16_nvfp4: shared memory attribute");
  head_nvfp4_topk16_kernel<C><<<dim3((a.M + C::TM - 1) / C::TM, a.ntiles), C::NT, SMEM, stream>>>(a);
  if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("batch head_topk16_nvfp4: launch failed");
}

template <class C>
void launch_nvfp4_tiled(const HeadNvfp4Args& a, cudaStream_t stream) {
  static const bool attr = [] {
    return cudaFuncSetAttribute(head_nvfp4_tiled_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) ==
           cudaSuccess;
  }();
  if (!attr) throw std::runtime_error("batch head_sample_nvfp4: shared memory attribute");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((a.M + C::TM - 1) / C::TM, a.ntiles);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl;
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &pdl;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, head_nvfp4_tiled_kernel<C>, a) != cudaSuccess)
    throw std::runtime_error("batch head_sample_nvfp4: launch failed");
}

__global__ void __launch_bounds__(ROW_T)
    gate_apply_kernel(const __grid_constant__ prefill::GateRows rows, const float* __restrict__ fg, int T,
                      const float* __restrict__ norm_w, std::uint16_t* __restrict__ out, std::size_t out_stride) {
  const int t = blockIdx.x, g = blockIdx.y, c = threadIdx.x * 4;
  const std::size_t off = std::size_t(t) * H + c;
  const float* f = fg + (std::size_t(g) * T + t) * 2 * H + c;
  const float4 h4 = *reinterpret_cast<const float4*>(rows.h[g] + off);
  const float4 m4 = *reinterpret_cast<const float4*>(rows.msg[g] + off);
  const float4 f4 = *reinterpret_cast<const float4*>(f), g4 = *reinterpret_cast<const float4*>(f + H);
  const float hv[4] = {h4.x, h4.y, h4.z, h4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
  const float fv[4] = {f4.x, f4.y, f4.z, f4.w}, gv[4] = {g4.x, g4.y, g4.z, g4.w};
  float y[4];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    y[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[i])), hv[i]), __fmul_rn(tanhf(gv[i]), mv[i]));
    ss += y[i] * y[i];
  }
  __shared__ float red[ROW_W];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  ss = 0.f;
  for (int w = 0; w < ROW_W; ++w) ss += red[w];
  const float r = rsqrtf(ss * (1.f / H) + kEps);
  const float4 w4 = *reinterpret_cast<const float4*>(norm_w + std::size_t(g) * H + c);
  *reinterpret_cast<uint2*>(out + g * out_stride + off) =
      make_uint2(pack_bf16x2(y[0] * r * w4.x, y[1] * r * w4.y), pack_bf16x2(y[2] * r * w4.z, y[3] * r * w4.w));
}

}  // namespace

std::size_t head_partials(int M, int N) { return std::size_t(M) * ((N + HEAD_TN - 1) / HEAD_TN); }

std::size_t head_partials_nvfp4(int M, int N) {
  return std::size_t(M) * ((N + NVFP4_SMALL_TILE - 1) / NVFP4_SMALL_TILE);
}

void head_sample(const __nv_bfloat16* A, int lda, const void* B, int ldb, int M, int N, int K,
                 const float* temps, std::uint64_t seed, float* logits, int ldc, int2* partials, std::int32_t* out,
                 cudaStream_t stream, int v_begin, void* cand_out, const std::uint64_t* d_seed,
                 const pico12::Layout* packed) {
  if (M <= 0 || N <= 0 || K % BK || lda % 8 || ldb % 8 || (logits && (N % 2 || ldc % 2)) ||
      (packed && (packed->rows != N || packed->cols != K)))
    throw std::invalid_argument("batch head_sample: unsupported shape");
  const auto pico = packed ? pico12::device_view(*packed, static_cast<const std::uint8_t*>(B)) : pico12::DeviceView{};
  const HeadArgs a{A, B, logits, lda, ldb, ldc, M, N, K, temps, seed, partials, (N + HEAD_TN - 1) / HEAD_TN, v_begin,
                   d_seed, pico};
  auto dispatch = [&]<class C>() {
    if (!packed) launch<C, false, false>(a, stream);
    else if (packed->escapes) launch<C, true, true>(a, stream);
    else launch<C, true, false>(a, stream);
  };
  if (M <= 32) dispatch.template operator()<Cfg32>();
  else dispatch.template operator()<Cfg64>();
  head_pick_kernel<<<M, PICK_T, 0, stream>>>(partials, a.ntiles, out, static_cast<int2*>(cand_out));
}

void head_sample_nvfp4(const __nv_bfloat16* A, int lda, kernels::head::Nvfp4Matrix w, int M, const float* temps,
                       std::uint64_t seed, float* logits, int ldc, int2* partials, std::int32_t* out,
                       cudaStream_t stream, void* cand_out, const std::uint64_t* d_seed) {
  if (M <= 0 || !A || !w.codes || !w.scales || w.rows <= 0 || lda % 8)
    throw std::invalid_argument("batch head_sample_nvfp4: unsupported shape");
  if (M <= 8) {
    const int ntiles = (w.rows + NVFP4_SMALL_TILE - 1) / NVFP4_SMALL_TILE;
    const HeadNvfp4Args a{A, w, logits, lda, ldc, M, temps, seed, partials, ntiles, d_seed};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(ntiles);
    cfg.blockDim = dim3(NVFP4_SMALL_SPLIT * 32);
    cfg.stream = stream;
    cudaLaunchAttribute pdl;
    pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &pdl;
    cfg.numAttrs = 1;
    if (cudaLaunchKernelEx(&cfg, head_nvfp4_small_kernel, a) != cudaSuccess)
      throw std::runtime_error("batch head_sample_nvfp4 small: launch failed");
    head_pick_kernel<<<M, PICK_T, 0, stream>>>(partials, ntiles, out, static_cast<int2*>(cand_out));
  } else {
    const int ntiles = (w.rows + NVFP4_TILED_TN - 1) / NVFP4_TILED_TN;
    const HeadNvfp4Args a{A, w, logits, lda, ldc, M, temps, seed, partials, ntiles, d_seed};
    if (M <= 32) launch_nvfp4_tiled<Nvfp4Cfg32>(a, stream);
    else launch_nvfp4_tiled<Nvfp4Cfg64>(a, stream);
    head_pick_kernel<<<M, PICK_T, 0, stream>>>(partials, ntiles, out, static_cast<int2*>(cand_out));
  }
}

std::size_t head_topk16_nvfp4_workspace_bytes(int M, int rows) {
  const int ntiles = (rows + NVFP4_TILED_TN - 1) / NVFP4_TILED_TN;
  const int groups = (ntiles + TOPK_GROUP_TILES - 1) / TOPK_GROUP_TILES;
  const std::size_t p_bytes = std::size_t(M) * ntiles * 16 * sizeof(kernels::head::Candidate);
  const std::size_t s_bytes = std::size_t(M) * groups * 16 * sizeof(kernels::head::Candidate);
  return ((p_bytes + 255) / 256 * 256) + s_bytes;
}

void head_topk16_nvfp4(const __nv_bfloat16* A, int lda, kernels::head::Nvfp4Matrix w, int M, void* ws,
                       float* out_vals, std::int32_t* out_ids, cudaStream_t stream) {
  if (M <= 0 || !A || !w.codes || !w.scales || w.rows < 16 || lda % 8 || !ws || !out_vals || !out_ids)
    throw std::invalid_argument("batch head_topk16_nvfp4: unsupported shape");
  const int ntiles = (w.rows + NVFP4_TILED_TN - 1) / NVFP4_TILED_TN;
  const int groups = (ntiles + TOPK_GROUP_TILES - 1) / TOPK_GROUP_TILES;
  if (groups * 16 > TOPK_RED_THREADS * 8)
    throw std::invalid_argument("batch head_topk16_nvfp4: vocabulary exceeds stage-2 capacity");
  auto* partials = static_cast<kernels::head::Candidate*>(ws);
  const std::size_t p_bytes = (std::size_t(M) * ntiles * 16 * sizeof(kernels::head::Candidate) + 255) / 256 * 256;
  auto* stage1 = reinterpret_cast<kernels::head::Candidate*>(static_cast<std::uint8_t*>(ws) + p_bytes);
  const HeadNvfp4TopkArgs a{A, w, lda, M, ntiles, partials};
  if (M <= 32) launch_nvfp4_topk16<Nvfp4Cfg32>(a, stream);
  else launch_nvfp4_topk16<Nvfp4Cfg64>(a, stream);
  reduce_topk16_kernel<4, false><<<dim3(groups, M), TOPK_RED_THREADS, 0, stream>>>(
      partials, ntiles * 16, TOPK_GROUP_TILES * 16, stage1, nullptr, nullptr);
  reduce_topk16_kernel<8, true><<<dim3(1, M), TOPK_RED_THREADS, 0, stream>>>(
      stage1, groups * 16, groups * 16, nullptr, out_vals, out_ids);
  if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("batch head_topk16_nvfp4: reduce launch failed");
}

void gate_apply(const prefill::GateRows& rows, int G, const float* fg, int T, const float* norm_w, __nv_bfloat16* out,
                std::size_t out_stride, cudaStream_t stream) {
  gate_apply_kernel<<<dim3(T, G), ROW_T, 0, stream>>>(rows, fg, T, norm_w, reinterpret_cast<std::uint16_t*>(out),
                                                       out_stride);
}

}  // namespace ingine::kernels::batch
