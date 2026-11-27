// proj, proj_conv, head, fp8_rows: weight-streaming GEMVs for M = 1..8 tokens.
//
// Design. Decode GEMVs are pure weight streams (M <= 8 activations, K = 2304 or 3072): the goal is to keep
// DRAM busy with the fewest instructions per weight byte. Two kernels share the epilogues:
//
// M = 1, `gemv_kernel` (CUDA cores):
// * One row per lane group: a full warp per BF16 row, a half warp per FP8 row, so every lane owns exactly
//   CH = K * bytes / (16 * LANES) 16-byte chunks (BF16 K=2304: 9; FP8 K=2304: 9; K=3072: 12). Chunk i of
//   lane l is bytes [(i * LANES + l) * 16, +16) of the row: each load instruction of a group reads one
//   contiguous 256/512-byte span (fully coalesced).
// * The CH chunk loads of a row are issued at the top of the kernel (read-only streaming loads, no L1
//   allocation; ptxas interleaves them with the FMAs of earlier chunks, leaving ~4 chunks = 2 KB in
//   flight per warp), and with ~32 resident warps per SM (low register count) that is ~64 KB per SM,
//   several times the bandwidth-delay product. (Measured alternatives, no gain: persistent grids, L2 bulk
//   prefetch of the row, 128/512-thread blocks.)
// * The activation is read with 16-byte L1-cached loads at the matching columns; FP32 FMAs, then a
//   warp reduction.
// * PICO12 keeps the BF16 lane mapping. Each warp loads 256 contiguous BF16 low bytes and 128
//   contiguous sign/exponent-code bytes per step, then reconstructs the BF16 operands in registers.
//
// M >= 2, `gemv_mma_kernel` (tensor cores): on CUDA cores every weight costs M FMAs plus the unpacking of
// M activations, which made M = 8 instruction-bound (FP8 proj lost 20-40%; staging the activations in
// shared memory or giving each lane two rows was measured and did not help: both raise register or
// shared-memory pressure and cut occupancy). Instead each warp runs mma.m16n8k16 (BF16 x BF16 -> FP32):
// A = 16 weight rows, B = the activations of up to 8 tokens (N = 8 = max M),
// so one instruction does 16 x 8 x 16 multiply-adds with FP32 accumulation, exact BF16 products as on the
// CUDA-core path. The reduction order over k is free, so the fragments are fed straight from coalesced
// loads: lane (g, t) loads columns [c + CPL t, +CPL) of rows g and g + 8 (CPL = 8 for BF16, 16 for FP8
// and PICO12) and the same columns of token g's activation. PICO12's four lanes fetch 64 BF16 low
// bytes and 32 sign/exponent bytes per row per step, filling three 32-byte sectors. A block of SPLITK warps
// owns 16 rows; warp w covers a 1/SPLITK
// slice of K (enough warps in flight for small row counts), and the partial 16 x 8 tiles are summed in
// shared memory in a fixed order. proj blocks that straddle a segment boundary with different inputs run
// one pass per input with the other rows masked to zero.
//
// Epilogues: proj rounds to BF16 (input vector per row from the device-side segment table), head applies
// the softcap, proj_conv applies the post-attention causal conv for its channel (= row) over the call's
// M tokens, rewrites that channel's conv state and optionally records the conv inputs (conv_hist).
// M is a template parameter. FP8 rows multiply the dot by the row scale. Exact PICO12 restores the
// original BF16 operands in the BF16 lane mapping.
// The experimental packed MMA path uses a different lane mapping and adds CSR outlier products after reduction.
// Eligible sparse scatter epilogues defer that work to a separate PDL kernel over just the outlier rows.
//
// Programmatic dependent launch (pdl=True): the M = 1 kernel issues all its row's weight loads, the
// M >= 2 kernel its first batch, before griddepcontrol.wait, so they overlap the previous kernel's tail
// (pdl_wait_fenced: a plain wait lets ptxas sink the loads below it; measured M = 1 proj 13.7 -> 11.8 us
// behind a 2 us predecessor); activations are read (coherent, non-hoistable loads: ld_act) and
// outputs/state written only after it, and the next kernel is released right after the wait.
// L2: the weight rows are read once per step, so their lines are filled evict-first (kernels/common/l2.cuh):
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
// Ported from engine/csrc/dense/gemv.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/common/l2.cuh"
#include "kernels/common/pico12.cuh"
#include "kernels/common/residual_conv.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include <cuda/atomic>

#include <cstdint>
#include <type_traits>

namespace ingine::kernels::dense {
namespace {

constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

enum class PicoMode { none, direct, sparse, exact_direct, exact_sparse, direct_defer };

template <PicoMode Mode>
constexpr bool packed_mma = Mode == PicoMode::direct || Mode == PicoMode::sparse || Mode == PicoMode::direct_defer;

template <int K, bool FP8, bool Packed, bool Mma = false>
struct WeightStream {
  static constexpr int columns = FP8 || (Packed && Mma) ? 16 : 8;
  static constexpr size_t row_bytes = Packed ? pico12::row_bytes(K) : size_t(K) * (FP8 ? 1 : 2);
  using PicoChunk = std::conditional_t<Mma, pico12::Chunk16, pico12::Chunk>;
  using Chunk = std::conditional_t<Packed, PicoChunk, uint4>;

  __device__ static __forceinline__ Chunk load(const uint8_t* row, int col, uint64_t policy) {
    if constexpr (Packed && Mma) return pico12::load16<WEIGHT_L2>(row, K, col, policy);
    else if constexpr (Packed) return pico12::load8<WEIGHT_L2>(row, K, col, policy);
    else return l2::ld_stream<WEIGHT_L2>(row + size_t(col) * (FP8 ? 1 : 2), policy);
  }

  __device__ static __forceinline__ uint32_t dependency(Chunk chunk) {
    if constexpr (Packed && Mma) return chunk.bf_low.x ^ chunk.codes.x;
    else if constexpr (Packed) return chunk.bf_low.x ^ chunk.codes;
    else return chunk.x ^ chunk.w;
  }

  __device__ static __forceinline__ uint4 bf16(Chunk chunk, const pico12::DeviceView& pico) {
    if constexpr (Packed) return pico12::decode8(chunk, pico);
    else return chunk;
  }
};

// ---- epilogues ----------------------------------------------------------------------------------------
// input_of(row): input index of a row (-1: row not computed); x_of(i): that input, bf16 [M, K].
// kScatter epilogues store one (row, token) value at a time (store1); proj_conv needs every token of a
// row together (store: lane-parallel for the CUDA-core kernel; store_row: one thread).

template <int K, bool L2Only = false, int N>
__device__ __forceinline__ void add_pico12_outliers(const uint32_t* entries, uint32_t begin, uint32_t end,
                                                   const __nv_bfloat16* x, float (&acc)[N]) {
  static_assert(K <= pico12::kCompactCols);
  for (uint32_t i = begin; i < end; ++i) {
    const uint32_t entry = __ldg(entries + i);
    const float weight = bf_hi(entry);
    const unsigned col = entry & 0xffffu;
#pragma unroll
    for (int m = 0; m < N; ++m) {
      unsigned short activation;
      const auto* xm = x + size_t(m) * K + col;
      // Use coherent loads after the PDL wait. Deferred rows bypass L1 across graph replays.
      if constexpr (L2Only)
        asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(activation) : "l"(xm) : "memory");
      else
        asm volatile("ld.global.u16 %0, [%1];" : "=h"(activation) : "l"(xm) : "memory");
      acc[m] = fmaf(weight, bf_lo(activation), acc[m]);
    }
  }
}

template <int K, int N>
__device__ __forceinline__ void add_pico12_outliers(const pico12::DeviceView& pico, int row,
                                                   const __nv_bfloat16* x, float (&acc)[N]) {
  const uint32_t begin = __ldg(pico.row_ptr + row), end = __ldg(pico.row_ptr + size_t(row) + 1);
  add_pico12_outliers<K>(pico.entries, begin, end, x, acc);
}

template <int M>
struct ProjEpi {
  static constexpr int kInputs = 3;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* xin;  // [3, M, H]
  const int* segments;       // [S, 3]
  int nseg, R;
  float* out;                // [M, R]
  __device__ int input_of(int row) const {
    for (int s = 0; s < nseg; ++s) {
      const int b = __ldg(segments + 3 * s), e = __ldg(segments + 3 * s + 1);
      if (row >= b && row < e) return __ldg(segments + 3 * s + 2);
    }
    return -1;
  }
  __device__ const __nv_bfloat16* x_of(int i) const { return xin + size_t(i) * M * H; }
  __device__ float* raw_slot(int row, int m) const { return out + size_t(m) * R + row; }
  __device__ void store1(int row, int m, float v, bool defer = false) const {
    *raw_slot(row, m) = defer ? v : round_bf16(v);
  }
};

__device__ __forceinline__ float head_score(float v) {
  return tanhf(round_bf16(v) / 30.f) * 30.f;
}

template <int M>
struct HeadEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int V;
  float* out;  // [M, V]
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ float* raw_slot(int row, int m) const { return out + size_t(m) * V + row; }
  __device__ void store1(int row, int m, float v, bool defer = false) const {
    *raw_slot(row, m) = defer ? v : head_score(v);
  }
};

// Same weight lanes, K partition, outlier order and epilogue as HeadEpi, including for a
// partial tile. Only tile scheduling and the reduction of finished scores are different.
template <int M>
struct SelectHeadEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  HeadTiles selection;

  __device__ unsigned tokens(int row) const {
    return selection.masks ? selection.masks[row] | *selection.fallback : 0xffu;
  }
  __device__ int input_of(int row) const { return tokens(row) ? 0 : -1; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    if (!(tokens(row) & (1u << m))) return;
    const float score = head_score(v);
    if (isnan(score)) return;
    const unsigned bits = __float_as_uint(score == 0.f ? 0.f : score);
    const unsigned ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    const auto key = (static_cast<unsigned long long>(ordered) << 32) | ~unsigned(row);
    const cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> best(selection.best[m]);
    if (key > best.load(cuda::memory_order_relaxed)) atomicMax(selection.best + m, key);
  }
};

// plain GEMV (drafter): one input x [M, K] (row stride K), out [M, R] = bf16(w . x)
template <int M>
struct GemvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int R;
  float* out;  // [M, R]
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ float* raw_slot(int row, int m) const { return out + size_t(m) * R + row; }
  __device__ void store1(int row, int m, float v, bool defer = false) const {
    *raw_slot(row, m) = defer ? v : round_bf16(v);
  }
};

template <int M>
struct RawEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  float* out;
  __nv_bfloat16* out_bf16;
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ float* raw_slot(int row, int m) const { return out + size_t(m) * H + row; }
  __device__ void store1(int row, int m, float v, bool defer = false) const {
    const size_t i = size_t(m) * H + row;
    if (out) out[i] = v;
    if (out_bf16 && !defer) out_bf16[i] = __float2bfloat16_rn(v);
  }
};

template <PicoMode Mode, class Epi>
__device__ __forceinline__ void store_scatter(const Epi& epi, const pico12::DeviceView& pico,
                                              int row, int m, float v) {
  if constexpr (Mode == PicoMode::direct_defer)
    epi.store1(row, m, v, (__ldg(pico.row_mask + (row >> 5)) >> (row & 31)) & 1u);
  else
    epi.store1(row, m, v);
}

// The DFlash2 projection has a variable output width. Keep it separate from
// the fixed-stride target convolution projection used in verify graphs.
struct DFlash2Epi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  __nv_bfloat16* out;
  int rows;
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    out[size_t(m) * rows + row] = __float2bfloat16_rn(v);
  }
};

struct DFlash2F32Epi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  float* out;
  int rows;
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    out[size_t(m) * rows + row] = v;
  }
};

template <bool Mlp, bool BF16>
__global__ void conv_epilogue_kernel(const float* x, const __nv_bfloat16* x_bf16, int M, const float* conv_w,
                                    float* state, float* out, float* hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  float a = state[c], b = state[H + c], d = state[2 * H + c];
  for (int m = 0; m < M; ++m) {
    const int i = m * H + c;
    const float cur = BF16 ? __bfloat162float(x_bf16[i]) : round_bf16(x[i]);
    const float ext[4] = {a, b, d, cur};
    out[i] = residual_conv<Mlp>(cw, ext);
    if (hist) hist[i] = cur;
    a = b;
    b = d;
    d = cur;
  }
  state[c] = a;
  state[H + c] = b;
  state[2 * H + c] = d;
}

template <bool Mlp, bool BF16>
__global__ void conv_epilogue_batch_kernel(const float* x, const __nv_bfloat16* x_bf16,
                                          const float* conv_w, float* const* states, float* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  float* state = states[blockIdx.y];
  const int i = blockIdx.y * H + c;
  const float cur = BF16 ? __bfloat162float(x_bf16[i]) : round_bf16(x[i]);
  const float ext[4] = {state[c], state[H + c], state[2 * H + c], cur};
  out[i] = residual_conv<Mlp>(cw, ext);
  state[c] = ext[1];
  state[H + c] = ext[2];
  state[2 * H + c] = ext[3];
}

template <int M>
struct ConvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = false;  // the conv of token m needs y of tokens m-3 .. m
  const __nv_bfloat16* x;  // [M, 3072]
  const float* conv_w;     // [H, 4], tap 3 = current token
  float* state;            // [3, H], oldest first
  float* out;              // [M, H]
  float* hist;             // [M, H] BF16-rounded conv inputs y, or null
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void load_ext(int c, const float (&acc)[M], float (&ext)[M + 3]) const {
#pragma unroll
    for (int j = 0; j < 3; ++j) ext[j] = state[j * H + c];
#pragma unroll
    for (int m = 0; m < M; ++m) ext[3 + m] = round_bf16(acc[m]);
  }
  __device__ void token(int c, int m, const float (&ext)[M + 3]) const {
    const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
    out[size_t(m) * H + c] = residual_conv<false>(cw, ext + m);
    if (hist) hist[size_t(m) * H + c] = ext[m + 3];
  }
  // lane-parallel (all lanes of the group hold every token's sum)
  __device__ void store(int c, const float (&acc)[M], int gl, unsigned mask) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m)
      if (gl == m) token(c, m, ext);
    __syncwarp(mask);  // the group has read the old state before it is overwritten
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (gl == 8 + j) state[j * H + c] = ext[M + j];
  }
  // one thread per channel
  __device__ void store_row(int c, const float (&acc)[M]) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m) token(c, m, ext);
#pragma unroll
    for (int j = 0; j < 3; ++j) state[j * H + c] = ext[M + j];
  }
};

// ---- M = 1: CUDA cores --------------------------------------------------------------------------------

template <bool FP8, int K, int LANES, int BLOCK, class Epi, PicoMode PICO = PicoMode::none>
__device__ __forceinline__ void gemv_tile(const uint8_t* w, const float* w_scale, int rows, uint32_t zero,
                                          const Epi& epi, const pico12::DeviceView& pico, int tile) {
  using Weights = WeightStream<K, FP8, PICO != PicoMode::none>;
  constexpr int EPC = Weights::columns;
  constexpr int CH = K / (EPC * LANES);
  static_assert(K % (EPC * LANES) == 0, "row must split evenly over the lane group");

  const int gl = threadIdx.x % LANES;
  const int row = (tile * BLOCK + threadIdx.x) / LANES;
  // A whole group shares `row`, so groups exit together and the shuffles below stay within live groups.
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const __nv_bfloat16* x = epi.x_of(in);  // pointer arithmetic only: the segment table is a constant

  const uint8_t* wr = w + size_t(row) * Weights::row_bytes;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  typename Weights::Chunk wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = Weights::load(wr, (i * LANES + gl) * EPC, wpol);
  pdl_wait_fenced(zero);  // the whole row is in flight; the activation comes from an earlier kernel
  pdl_trigger();

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (FP8) unpack_e4m3x16(wv[i], wf);
    else if constexpr (PICO == PicoMode::exact_sparse) unpack_bf16x8(pico12::restore8<false>(wv[i], pico, row, col), wf);
    else unpack_bf16x8(Weights::bf16(wv[i], pico), wf);
    unpack_bf16x8(ld_act(x + col), xf);
    if constexpr (FP8) unpack_bf16x8(ld_act(x + col + 8), xf + 8);
#pragma unroll
    for (int j = 0; j < EPC; ++j) acc = fmaf(wf[j], xf[j], acc);
  }

  float v[1] = {group_sum<LANES>(acc) * (FP8 ? __ldg(w_scale + row) : 1.f)};
  if constexpr (PICO == PicoMode::sparse) {
    if (gl == 0) add_pico12_outliers<K>(pico, row, x, v);
    if constexpr (!Epi::kScatter) v[0] = __shfl_sync(group_mask<LANES>(), v[0], 0, LANES);
  }
  if constexpr (Epi::kScatter) {
    if (gl == 0) store_scatter<PICO>(epi, pico, row, 0, v[0]);
  } else {
    epi.store(row, v, gl, group_mask<LANES>());
  }
}

template <class Epi, class Work>
__device__ __forceinline__ void visit_tiles(const Epi& epi, int tiles, Work work) {
  if constexpr (requires { epi.selection; }) {
    pdl_wait();
    const auto sel = epi.selection;
    const bool all = sel.fallback && *sel.fallback;
    const int count = all ? tiles : int(*sel.count);
    for (int i = blockIdx.x; i < count; i += gridDim.x) {
      const int tile = all || !sel.indices ? i : sel.indices[i];
      work(tile);
      __syncthreads();
    }
  } else {
    work(blockIdx.x);
  }
}

template <bool FP8, int K, int LANES, int BLOCK, class Epi, PicoMode PICO = PicoMode::none>
__global__ void __launch_bounds__(BLOCK, (PICO == PicoMode::direct_defer && BLOCK == 256) ? 4 : 0)
gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ w_scale, int rows, uint32_t zero,
            Epi epi, const __grid_constant__ pico12::DeviceView pico) {
  constexpr int tile_rows = BLOCK / LANES;
  visit_tiles(epi, (rows - 1) / tile_rows + 1, [&](int tile) {
    gemv_tile<FP8, K, LANES, BLOCK, Epi, PICO>(w, w_scale, rows, zero, epi, pico, tile);
  });
}

// ---- M >= 2: tensor cores -----------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2 E4M3 (low 16 bits of u) -> BF16x2, exact: one instruction (cvt.rn.bf16x2.e4m3x2; through FP16 and
// FP32 it took four, 16 per mma)
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1))
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#else
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
  float2 f = __half22float2(__half2(hr));
  __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&b);
#endif
}

#ifndef MMA_U_BF16
#define MMA_U_BF16 6
#endif
#ifndef MMA_U_FP8
#define MMA_U_FP8 3
#endif
// `zero` is 0 at run time (opaque to the compiler): every mma of a batch takes an operand xor-ed with
// (xor of all the batch's loads) & zero, so ptxas cannot interleave the mma with the batch's loads and
// all U steps' loads of a warp are in flight together (otherwise it issues one step's loads at a time,
// ~1 KB per warp, and scattered 64-byte requests from 16 rows run at ~40% of DRAM bandwidth).
// PICO12 batches the weight loads and reads the reused, L1-cached activations as they are consumed
// to leave more registers available for the packed fragments and high-byte decode.
template <bool FP8, int K, int M, int SPLITK, int U, class Epi, PicoMode PICO = PicoMode::none>
__device__ __forceinline__ void gemv_mma_tile(const uint8_t* w, const float* w_scale, int rows,
                                              uint32_t zero, const Epi& epi, const pico12::DeviceView& pico,
                                              int tile, float (&cs)[SPLITK][16][8]) {
  constexpr bool PACKED = PICO != PicoMode::none;
  constexpr bool PACKED_MMA = packed_mma<PICO>;
  using Weights = WeightStream<K, FP8, PACKED, PACKED_MMA>;
  using Chunk = typename Weights::Chunk;
  constexpr int CPL = Weights::columns;
  constexpr int SPAN = 4 * CPL;      // columns per step (4 lanes t)
  constexpr int KS = K / SPLITK;     // columns per warp
  static_assert(K % SPLITK == 0 && KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int r0 = tile * 16;
  const int ra = r0 + g, rb = r0 + g + 8;
  const int ia = ra < rows ? epi.input_of(ra) : -1, ib = rb < rows ? epi.input_of(rb) : -1;
  const int c0 = warp * KS + t * CPL;

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  bool waited = false;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  auto pass = [&]<bool Uniform = false>(int cur) {
    const bool la = Uniform || ia == cur, lb = Uniform || ib == cur, lx = g < M;
    const int row_a = la ? ra : 0, row_b = lb ? rb : 0;
    const uint8_t* wa = w + size_t(row_a) * Weights::row_bytes;
    const uint8_t* wb = w + size_t(row_b) * Weights::row_bytes;
    const __nv_bfloat16* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
    // Masked loads produce code zero. In direct format it can name a nonzero exponent.
    constexpr bool selected = requires { epi.selection; };
    constexpr bool mask_high = PICO == PicoMode::direct && !Uniform && (Epi::kInputs > 1 || selected);
    [[maybe_unused]] const uint32_t ma = mask_high ? 0u - uint32_t(la) : 0xffffffffu;
    [[maybe_unused]] const uint32_t mb = mask_high ? 0u - uint32_t(lb) : 0xffffffffu;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      Chunk A[U], B[U];
      [[maybe_unused]] uint4 X[U][CPL / 8];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
        if constexpr (Uniform || (PACKED_MMA && Epi::kInputs == 1 && !selected)) {
          // Single-input tails may read row zero. Their accumulators are discarded.
          A[u] = Weights::load(wa, col, wpol);
          B[u] = Weights::load(wb, col, wpol);
        } else {
          A[u] = la ? Weights::load(wa, col, wpol) : Chunk{};
          B[u] = lb ? Weights::load(wb, col, wpol) : Chunk{};
        }
      }
      if (!waited) {  // the first batch of weights is in flight; activations come from earlier kernels
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
      if constexpr (!PACKED_MMA) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int col = c0 + (jb + u) * SPAN;
#pragma unroll
          for (int h = 0; h < CPL / 8; ++h) X[u][h] = lx ? ld_act(xg + col + 8 * h) : make_uint4(0, 0, 0, 0);
        }
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        dep ^= Weights::dependency(A[u]) ^ Weights::dependency(B[u]);
        if constexpr (!PACKED_MMA) {
#pragma unroll
          for (int h = 0; h < CPL / 8; ++h) dep ^= X[u][h].x;
        }
      }
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (FP8) {
          // 16 columns = 4 k-steps of 4 columns (2 k-pairs) each
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_bf16(c, e4m3x2_to_bf16x2(aw[s]), e4m3x2_to_bf16x2(bw[s]), e4m3x2_to_bf16x2(aw[s] >> 16),
                     e4m3x2_to_bf16x2(bw[s] >> 16), xw[2 * s] ^ dep, xw[2 * s + 1]);
        } else if constexpr (PACKED_MMA) {
          const uint4 as = A[u].bf_low, bs = B[u].bf_low;
          const uint32_t aw[4] = {as.x, as.y, as.z, as.w}, bw[4] = {bs.x, bs.y, bs.z, bs.w};
          const uint32_t ae[2] = {A[u].codes.x, A[u].codes.y}, be[2] = {B[u].codes.x, B[u].codes.y};
          const int col = c0 + (jb + u) * SPAN;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            // Lanes for unused tokens can read token zero and discard those outputs.
            const uint4 x = ld_act(xg + col + 8 * h);
            const uint2 al = pico12::decode4(pico, aw[2 * h], ae[h], ma);
            const uint2 bl = pico12::decode4(pico, bw[2 * h], be[h], mb);
            mma_bf16(c, al.x, bl.x, al.y, bl.y, x.x ^ dep, x.y);
            const uint2 ah = pico12::decode4(pico, aw[2 * h + 1], ae[h] >> 16, ma);
            const uint2 bh = pico12::decode4(pico, bw[2 * h + 1], be[h] >> 16, mb);
            mma_bf16(c, ah.x, bh.x, ah.y, bh.y, x.z ^ dep, x.w);
          }
        } else {
          [[maybe_unused]] const int col = c0 + (jb + u) * SPAN;
          auto decode = [&](Chunk chunk, int row, bool live) {
            if constexpr (PACKED) {
              if (!live) return uint4{};
              if constexpr (PICO == PicoMode::exact_sparse) return pico12::restore8<false>(chunk, pico, row, col);
              else return pico12::decode8(chunk, pico);
            } else {
              return chunk;
            }
          };
          const uint4 a = decode(A[u], row_a, la), b = decode(B[u], row_b, lb);
          mma_bf16(c, a.x, b.x, a.y, b.y, X[u][0].x ^ dep, X[u][0].y);
          mma_bf16(c, a.z, b.z, a.w, b.w, X[u][0].z ^ dep, X[u][0].w);
        }
      }
    }
  };
  auto masked_passes = [&] {
    // inputs present among the block's 16 rows (warp-uniform; every warp sees the same rows)
#pragma unroll 1
    for (int i = 0; i < Epi::kInputs; ++i)
      if (__any_sync(0xffffffffu, ia == i || ib == i)) pass(i);
  };
  if constexpr (Epi::kInputs == 1) {
    pass(0);
  } else if constexpr (PACKED_MMA) {
    // Vote on each row's first matching segment, including holes and partial tiles.
    const int cur = __shfl_sync(0xffffffffu, ia, 0);
    if (cur >= 0 && cur < Epi::kInputs && __all_sync(0xffffffffu, ia == cur && ib == cur))
      pass.template operator()<true>(cur);
    else
      masked_passes();
  } else {
    masked_passes();
  }

  // C fragment: c0, c1 = rows g, tokens 2t, 2t+1; c2, c3 = row g + 8
  cs[warp][g][2 * t] = c[0];
  cs[warp][g][2 * t + 1] = c[1];
  cs[warp][g + 8][2 * t] = c[2];
  cs[warp][g + 8][2 * t + 1] = c[3];
  __syncthreads();
  auto total = [&](int r, int m) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < SPLITK; ++k) s += cs[k][r][m];
    return s;
  };
  if constexpr (Epi::kScatter && PICO == PicoMode::sparse) {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const int in = epi.input_of(row);
      if (in >= 0) {
        // One thread owns the CSR range and reuses every entry for all tokens.
        float y[M];
#pragma unroll
        for (int m = 0; m < M; ++m) y[m] = total(r, m);
        add_pico12_outliers<K>(pico, row, epi.x_of(in), y);
#pragma unroll
        for (int m = 0; m < M; ++m) epi.store1(row, m, y[m]);
      }
    }
  } else if constexpr (Epi::kScatter) {
    for (int k = threadIdx.x; k < 16 * M; k += 32 * SPLITK) {
      const int r = k / M, m = k % M, row = r0 + r;
      if (row >= rows) continue;
      const int in = epi.input_of(row);
      if (in >= 0) {
        const float y = total(r, m) * (FP8 ? __ldg(w_scale + row) : 1.f);
        store_scatter<PICO>(epi, pico, row, m, y);
      }
    }
  } else {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const float scale = FP8 ? __ldg(w_scale + row) : 1.f;
      float y[M];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m] = total(r, m) * scale;
      if constexpr (PICO == PicoMode::sparse) add_pico12_outliers<K>(pico, row, epi.x_of(0), y);
      epi.store_row(row, y);
    }
  }
}

template <bool FP8, int K, int M, int SPLITK, int U, class Epi, PicoMode PICO = PicoMode::none>
__global__ void __launch_bounds__(32 * SPLITK) gemv_mma_kernel(const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scale, int rows,
                                                              uint32_t zero, Epi epi,
                                                              const __grid_constant__ pico12::DeviceView pico) {
  __shared__ float cs[SPLITK][16][8];
  visit_tiles(epi, (rows - 1) / 16 + 1, [&](int tile) {
    gemv_mma_tile<FP8, K, M, SPLITK, U, Epi, PICO>(w, w_scale, rows, zero, epi, pico, tile, cs);
  });
}

constexpr int kOutlierBlock = 128;

template <int K, int M, class Epi>
__global__ void __launch_bounds__(kOutlierBlock) pico12_apply_outliers_kernel(
    Epi epi, const __grid_constant__ pico12::DeviceView pico) {
  const unsigned idx = blockIdx.x * kOutlierBlock + threadIdx.x;
  if (idx >= unsigned(pico.num_outlier_rows)) return;
  const int row = __ldg(pico.outlier_rows + idx);
  const int in = epi.input_of(row);
  if (in < 0) return;
  const uint32_t begin = __ldg(pico.row_ptr + row), end = __ldg(pico.row_ptr + size_t(row) + 1);
  // The row list contains only nonempty rows. Start fetching the immutable CSR data during GEMV.
  asm volatile("prefetch.global.L2 [%0];" :: "l"(pico.entries + begin) : "memory");
  pdl_wait();

  float y[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float* raw = epi.raw_slot(row, m);
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(y[m]) : "l"(raw) : "memory");
  }
  add_pico12_outliers<K, true>(pico.entries, begin, end, epi.x_of(in), y);
#pragma unroll
  for (int m = 0; m < M; ++m) epi.store1(row, m, y[m]);
  // Release dependent kernels at completion, after the final epilogue stores.
}

// ---- launch helpers -----------------------------------------------------------------------------------

// BLOCK: threads per block of the M = 1 kernel (0: M = 1 runs the tensor-core kernel too, for K whose
// whole row per lane would not fit in registers); SPLITK: warps per 16-row block of the M >= 2 kernel;
// UFP8: steps per FP8 load batch. Packed PICO12 can override both the warp count and the batch size.
template <int K, int BLOCK, int SPLITK, int UFP8, template <int> class Epi,
          int PicoSplitK = SPLITK, int PicoU = UFP8, class Make>
void launch_gemv(const Matrix& w, int M, bool pdl, cudaStream_t stream, Make make, int selected_blocks = 0) {
  const bool fp8 = w.scale != nullptr;
  require(w.w != nullptr && w.rows > 0, "weight matrix is empty");
  const int rows = w.rows;
  const float* sc = w.scale;
  const uint8_t* wp = static_cast<const uint8_t*>(w.w);
  pico12::DeviceView pico{};
  if (w.pico12) {
    static_assert(K % pico12::kTile == 0);
    const auto& layout = *w.pico12;
    require(!fp8 && layout.rows == rows && layout.cols == K && (w.cols == 0 || w.cols == K) &&
                reinterpret_cast<uintptr_t>(wp) % 16 == 0,
            "PICO12: packed matrix shape, alignment, or scale is invalid");
    pico = pico12::device_view(layout, wp);
  }
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    auto epi = make(std::integral_constant<int, MM>{});
    using E = decltype(epi);
    auto launch = [&](auto fp8_tag, auto pico_tag) {
      constexpr bool FP8 = decltype(fp8_tag)::value;
      constexpr PicoMode PICO = decltype(pico_tag)::value;
      if constexpr (MM == 1 && BLOCK > 0) {
        constexpr int B = BLOCK > 0 ? BLOCK : 32;  // (the branch is discarded for BLOCK = 0)
        constexpr int LANES = FP8 ? 16 : 32;
        constexpr int ROWS_PER_BLOCK = B / LANES;
        const dim3 blocks(selected_blocks ? selected_blocks : (rows - 1) / ROWS_PER_BLOCK + 1);
        constexpr PicoMode mode = PICO == PicoMode::exact_direct ? PicoMode::direct : PICO;
        launch_ex(gemv_kernel<FP8, K, LANES, B, E, mode>, blocks, B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, pico);
      } else {
        constexpr int split_k = packed_mma<PICO> ? PicoSplitK : SPLITK;
        // The long-K, single-token MMA scatter needs fewer live fragments to stay spill-free.
        constexpr bool single_step = PICO == PicoMode::direct_defer && MM == 1 && BLOCK == 0;
        constexpr int U = single_step ? 1 : (packed_mma<PICO> ? PicoU : (FP8 ? UFP8 : MMA_U_BF16));
        const dim3 blocks(selected_blocks ? selected_blocks : (rows - 1) / 16 + 1);
        launch_ex(gemv_mma_kernel<FP8, K, MM, split_k, U, E, PICO>, blocks, 32 * split_k, 0, 1,
                  pdl, stream, wp, sc, rows, 0u, epi, pico);
      }
    };
    if (fp8) launch(std::true_type{}, std::integral_constant<PicoMode, PicoMode::none>{});
    else if (!w.pico12) launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::none>{});
    else if (w.pico12_exact) {
      if (w.pico12->escapes) launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::exact_sparse>{});
      else launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::exact_direct>{});
    } else if (w.pico12->escapes) {
      static const bool defer_csr = [] {
        const char* e = std::getenv("INGINE_PICO12_DEFER_CSR");
        return e && std::atoi(e) != 0;
      }();
      if constexpr (E::kScatter && !requires { epi.selection; } && requires { epi.raw_slot(0, 0); }) {
        if (defer_csr && epi.out && pico.num_outlier_rows > 0) {
          launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::direct_defer>{});
          const dim3 blocks((pico.num_outlier_rows - 1) / kOutlierBlock + 1);
          launch_ex(pico12_apply_outliers_kernel<K, MM, E>, blocks, kOutlierBlock, 0, 1, pdl, stream, epi, pico);
          return;
        }
      }
      launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::sparse>{});
    } else {
      launch(std::false_type{}, std::integral_constant<PicoMode, PicoMode::direct>{});
    }
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "M must be in 1..8");
  }
}

// Warps per 16-row block of the M >= 2 kernel (K slice per warp must hold a whole number of U-step
// batches). Measured at M = 2 and 8 (shared GPU): proj 6 > 4 > 2 (FP8 +4-8%), head 6 ~ 4 >> 2 >> 1
// (+20-30%: 12.5K blocks need short K slices to keep enough rows in flight), proj_conv 4 ~ 8 ~ 16.
#ifndef PROJ_SPLITK
#define PROJ_SPLITK 6
#endif
// Keep the packed projection's default K partition and batch its whole slice before the PDL wait.
// A different split is for benchmarking and changes the packed MMA reduction order.
#ifndef PROJ_PICO_SPLITK
#define PROJ_PICO_SPLITK 6
#endif
#ifndef PROJ_PICO_U
#define PROJ_PICO_U (H / (64 * PROJ_PICO_SPLITK))
#endif
#ifndef CONV_SPLITK
#define CONV_SPLITK 8
#endif
// FP8 steps per load batch of proj_conv's M >= 2 kernel: 6 = a warp's whole K slice. Its 144 blocks are all
// resident at once (2 per SM at 120 registers), so every weight byte is requested before the PDL wait, as in
// the M = 1 kernel (measured against 3 steps: 5.9 -> 4.9 us per call at M = 2, 6.6 -> 6.0 at M = 8). proj
// (538 blocks) loses with 6: half its blocks would wait for a slot (14.0 -> 14.5 us at M = 2).
#ifndef CONV_U_FP8
#define CONV_U_FP8 6
#endif
#ifndef HEAD_SPLITK
#define HEAD_SPLITK 6
#endif

__global__ void fp8_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, __nv_fp8_storage_t* q,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = amax / 448.f;
  if (lane == 0) scale[row] = s;
  for (int k = lane; k < K; k += 32)
    q[size_t(row) * K + k] =
        s > 0.f ? __nv_cvt_float_to_fp8(__bfloat162float(wr[k]) / s, __NV_SATFINITE, __NV_E4M3) : 0;
}

// weights.dequant: bf16(float(q) * scale[row]); one thread per 16 elements (K % 16 == 0).
__global__ void dequant_rows_kernel(const __nv_fp8_storage_t* __restrict__ q, const float* __restrict__ scale,
                                    int K, long long total16, __nv_bfloat16* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const float s = scale[e / K];
  const uint4 u = *reinterpret_cast<const uint4*>(q + e);
  float f[16];
  unpack_e4m3x16(u, f);
  __align__(16) __nv_bfloat16 v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = __float2bfloat16_rn(f[j] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

template <bool Escapes, bool Gather, bool Wide>
__global__ void unpack_pico12_rows_kernel(const uint8_t* blob, int cols, size_t chunks, int first,
                                         const std::int32_t* indices, pico12::DeviceView view, uint4* out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= chunks) return;
  const int chunks_per_row = cols / pico12::kChunk;
  const int out_row = int(i / chunks_per_row);
  const int row = Gather ? indices[out_row] : first + out_row;
  const int col = int(i % chunks_per_row) * pico12::kChunk;
  const auto* src = blob + size_t(row) * pico12::row_bytes(cols);
  const auto chunk = pico12::load8<l2::Evict::none>(src, cols, col, 0);
  if constexpr (Escapes) out[i] = pico12::restore8<Wide>(chunk, view, row, col);
  else out[i] = pico12::decode8(chunk, view);
}

}  // namespace

void proj(const __nv_bfloat16* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream) {
  require(xin && segments && out && nseg > 0, "proj: null argument");
  const int R = w.rows;
  launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, ProjEpi, PROJ_PICO_SPLITK, PROJ_PICO_U>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ProjEpi<MM>{xin, segments, nseg, R, out};
  });
}

void proj_conv(const __nv_bfloat16* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream) {
  require(x && conv_w && conv_state && out, "proj_conv: null argument");
  require(w.rows == H && (w.cols == 0 || w.cols == 3072), "proj_conv: w must be [2304, 3072]");
  // M = 1: 2304 rows = 1152-2304 warps; 32-thread blocks spread them evenly over the SMs (measured: equal
  // or better than 64-256)
  launch_gemv<3072, 32, CONV_SPLITK, CONV_U_FP8, ConvEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ConvEpi<MM>{x, conv_w, conv_state, out, conv_hist};
  });
}

void proj_raw(const __nv_bfloat16* x, int M, Matrix w, float* out, __nv_bfloat16* out_bf16,
              bool pdl, cudaStream_t stream) {
  require(x && (out || out_bf16) && w.rows == H, "proj_raw: invalid arguments");
  auto epi = [&](auto mc) { return RawEpi<decltype(mc)::value>{x, out, out_bf16}; };
  const int K = w.cols ? w.cols : 3072;
  if (K == 3072) launch_gemv<3072, 32, CONV_SPLITK, CONV_U_FP8, RawEpi>(w, M, pdl, stream, epi);
  else if (K == 1536) launch_gemv<1536, 32, 4, CONV_U_FP8, RawEpi>(w, M, pdl, stream, epi);
  else require(false, "proj_raw: K must be 3072 or 1536");
}

void conv_residual_epilogue(const float* x, const __nv_bfloat16* x_bf16, int M, ConvKind kind,
                            const float* conv_w, float* conv_state, float* out, float* conv_hist,
                            bool pdl, cudaStream_t stream) {
  require(bool(x) != bool(x_bf16) && conv_w && conv_state && out && M >= 1 && M <= 8,
          "conv_residual_epilogue: invalid arguments");
  const auto kernel = kind == ConvKind::Mlp
      ? (x ? conv_epilogue_kernel<true, false> : conv_epilogue_kernel<true, true>)
      : (x ? conv_epilogue_kernel<false, false> : conv_epilogue_kernel<false, true>);
  launch_ex(kernel, dim3(H / 128), 128, 0, 1, pdl, stream, x, x_bf16, M, conv_w, conv_state, out, conv_hist);
}

void conv_residual_epilogue_batch(const float* x, const __nv_bfloat16* x_bf16, int batch, ConvKind kind,
                                  const float* conv_w, float* const* states, float* out,
                                  bool pdl, cudaStream_t stream) {
  require(bool(x) != bool(x_bf16) && conv_w && states && out && batch >= 1 && batch <= 8,
          "conv_residual_epilogue_batch: invalid arguments");
  const auto kernel = kind == ConvKind::Mlp
      ? (x ? conv_epilogue_batch_kernel<true, false> : conv_epilogue_batch_kernel<true, true>)
      : (x ? conv_epilogue_batch_kernel<false, false> : conv_epilogue_batch_kernel<false, true>);
  launch_ex(kernel, dim3(H / 128, batch), 128, 0, 1, pdl, stream, x, x_bf16, conv_w, states, out);
}

void head(const __nv_bfloat16* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "head: null argument");
  const int V = w.rows;
  launch_gemv<H, 256, HEAD_SPLITK, MMA_U_FP8, HeadEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return HeadEpi<MM>{x, V, out};
  });
}

void head_tiles(const __nv_bfloat16* x, int M, Matrix w, HeadTiles work, int blocks, cudaStream_t stream) {
  require(x && !w.scale && work.count && work.best && blocks > 0, "head_tiles: invalid arguments");
  launch_gemv<H, 256, HEAD_SPLITK, MMA_U_FP8, SelectHeadEpi>(w, M, false, stream, [&](auto mc) {
    return SelectHeadEpi<decltype(mc)::value>{x, work};
  }, blocks);
}

void gemv(const __nv_bfloat16* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "gemv: null argument");
  const int R = w.rows;
  auto make = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return GemvEpi<MM>{x, R, out};
  };
  // 2304 rows = 144 blocks of 16 rows: 16 warps per block (K slices of 192 / 576 columns) keep enough
  // bytes in flight (measured M = 8, BF16: K = 3072 10.4 -> 8.4 us, K = 9216 28.3 -> 26.0 us vs 8 warps).
  // K = 3072, M = 1: as proj_conv; K = 9216: 36 chunks per lane would spill, so M = 1 uses tensor cores.
  if (K == H) launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 3072) launch_gemv<3072, 32, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 9216) launch_gemv<9216, 0, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else require(false, "gemv: K must be 2304, 3072, or 9216");
}

void dflash2_fp8(const __nv_bfloat16* x, Matrix w, __nv_bfloat16* out, cudaStream_t stream) {
  require(x && w.w && w.scale && !w.pico12 && w.rows > 0 && out, "dflash2_fp8: invalid arguments");
  const dim3 grid((w.rows + 15) / 16);
  const auto* codes = static_cast<const uint8_t*>(w.w);
  const DFlash2Epi epi{x, out, w.rows};
  switch (w.cols) {
    case 2048:
      launch_ex(gemv_mma_kernel<true, 2048, 8, 8, 4, DFlash2Epi>, grid, 256, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 2304:
      launch_ex(gemv_mma_kernel<true, 2304, 8, 12, 3, DFlash2Epi>, grid, 384, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 3456:
      launch_ex(gemv_mma_kernel<true, 3456, 8, 9, 3, DFlash2Epi>, grid, 288, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 4096:
      launch_ex(gemv_mma_kernel<true, 4096, 8, 8, 4, DFlash2Epi>, grid, 256, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 6912:
      launch_ex(gemv_mma_kernel<true, 6912, 8, 12, 3, DFlash2Epi>, grid, 384, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 9216:
      launch_ex(gemv_mma_kernel<true, 9216, 8, 16, 3, DFlash2Epi>, grid, 512, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    default: require(false, "dflash2_fp8: unsupported input width");
  }
}

void dflash2_fp8_f32(const __nv_bfloat16* x, Matrix w, float* out, cudaStream_t stream) {
  require(x && w.w && w.scale && !w.pico12 && w.rows > 0 && out, "dflash2_fp8_f32: invalid arguments");
  const dim3 grid((w.rows + 15) / 16);
  const auto* codes = static_cast<const uint8_t*>(w.w);
  const DFlash2F32Epi epi{x, out, w.rows};
  switch (w.cols) {
    case 2048:
      launch_ex(gemv_mma_kernel<true, 2048, 8, 8, 4, DFlash2F32Epi>, grid, 256, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    case 3456:
      launch_ex(gemv_mma_kernel<true, 3456, 8, 9, 3, DFlash2F32Epi>, grid, 288, 0, 1,
                false, stream, codes, w.scale, w.rows, 0u, epi, pico12::DeviceView{});
      break;
    default: require(false, "dflash2_fp8_f32: unsupported input width");
  }
}

void fp8_rows(const __nv_bfloat16* w, int rows, int K, void* q, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0, "fp8_rows: bad shape");
  if (rows > 0)
    fp8_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<__nv_fp8_storage_t*>(q), scale);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_rows(const void* q, const float* scale, int rows, int K, __nv_bfloat16* out, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 16 == 0, "dequant_rows: K must be a multiple of 16");
  const long long total16 = (long long)rows * K / 16;
  if (total16 > 0)
    dequant_rows_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_fp8_storage_t*>(q), scale, K, total16, out);
  CUDA_CHECK(cudaGetLastError());
}

void unpack_pico12_rows(Matrix w, int rows, __nv_bfloat16* out, cudaStream_t stream, int first,
                        const std::int32_t* indices) {
  require(w.w && !w.scale && w.pico12 && w.rows == w.pico12->rows && w.cols == w.pico12->cols &&
              w.cols > 0 && w.cols % pico12::kTile == 0 && out && rows >= 0 &&
              first >= 0 && first <= w.rows && rows <= w.rows - first,
          "unpack_pico12_rows: invalid matrix or row range");
  if (rows == 0) return;
  const auto* blob = static_cast<const uint8_t*>(w.w);
  const auto view = pico12::device_view(*w.pico12, blob);
  const size_t chunks = size_t(rows) * (w.cols / pico12::kChunk);
  auto launch = [&]<bool Escapes, bool Gather, bool Wide = false>() {
    unpack_pico12_rows_kernel<Escapes, Gather, Wide><<<unsigned((chunks + 255) / 256), 256, 0, stream>>>(
        blob, w.cols, chunks, first, indices, view, reinterpret_cast<uint4*>(out));
  };
  if (w.pico12->escapes && w.cols > pico12::kCompactCols) {
    if (indices) launch.template operator()<true, true, true>();
    else launch.template operator()<true, false, true>();
  } else if (w.pico12->escapes) {
    if (indices) launch.template operator()<true, true>();
    else launch.template operator()<true, false>();
  } else {
    if (indices) launch.template operator()<false, true>();
    else launch.template operator()<false, false>();
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense
