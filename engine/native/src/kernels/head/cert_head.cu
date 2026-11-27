#include "kernels/head/cert_head.h"

#include "core/device.h"
#include "kernels/common/nvfp4.cuh"
#include "kernels/dense/dense.cuh"
#include "model/tp_layout.h"

#include <math_constants.h>

#include <algorithm>
#include <stdexcept>

namespace ingine::kernels::head {
namespace {

constexpr int K = model::HIDDEN;
constexpr int Threads = 256;
constexpr int Warps = Threads / 32;
constexpr int ExactBlocks = 512;

// The CUDA-core reference uses FP32 FMA and a warp sum. The MMA paths specify at least
// FP32 accumulation. Allow a full ulp per product/accumulation, even in a sequential
// K-term sum, plus split-K and sparse-outlier additions. 2^-10 exceeds gamma_(2K+32)
// with u = 2^-23. Bounding both dot errors separately gives the sum of the two norms.
constexpr float DotRelativeError = 0x1p-10f;
// Keep all partial sums far from overflow. BF16 subnormal input flushing and FP32
// underflow then contribute less than K * 2^(-126+30) * 2 to both projections combined.
constexpr float MaxOperand = 0x1p30f;
constexpr float DotAbsoluteError = 0x1p-70f;

__device__ __forceinline__ float warp_sum_up(float v) {
  for (int d = 16; d; d >>= 1)
    v = __fadd_ru(v, __shfl_xor_sync(0xffffffffu, v, d));
  return v;
}

__global__ void row_bounds(const __nv_bfloat16* exact, Nvfp4Matrix proxy, float* out) {
  const int row = blockIdx.x * Warps + threadIdx.x / 32;
  if (row >= proxy.rows) return;
  const int lane = threadIdx.x & 31;
  float diff = 0.f, w2 = 0.f, p2 = 0.f;
  bool valid = true;
  for (int c = lane; c < K / 32; c += 32) {
    const size_t elem = size_t(row) * K + c * 32;
    const auto codes = reinterpret_cast<const uint4*>(proxy.codes)[elem / 32];
    const auto scales = reinterpret_cast<const uint16_t*>(proxy.scales)[elem / 32];
    uint32_t pairs[16];
    nvfp4::decode(codes, scales, proxy.outer_scale, pairs);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 chunk = reinterpret_cast<const uint4*>(exact)[elem / 8 + j];
      const uint32_t original[] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 w = {dense::bf_lo(original[i]), dense::bf_hi(original[i])};
        const float2 p = {dense::bf_lo(pairs[4 * j + i]), dense::bf_hi(pairs[4 * j + i])};
        valid &= fabsf(w.x) <= MaxOperand && fabsf(w.y) <= MaxOperand &&
                 fabsf(p.x) <= MaxOperand && fabsf(p.y) <= MaxOperand;
        const float dx = __fsub_ru(fmaxf(w.x, p.x), fminf(w.x, p.x));
        const float dy = __fsub_ru(fmaxf(w.y, p.y), fminf(w.y, p.y));
        diff = __fmaf_ru(dy, dy, __fmaf_ru(dx, dx, diff));
        w2 = __fmaf_ru(w.y, w.y, __fmaf_ru(w.x, w.x, w2));
        p2 = __fmaf_ru(p.y, p.y, __fmaf_ru(p.x, p.x, p2));
      }
    }
  }
  const unsigned valid_lanes = __ballot_sync(0xffffffffu, valid);
  diff = warp_sum_up(diff);
  w2 = warp_sum_up(w2);
  p2 = warp_sum_up(p2);
  if (lane == 0) {
    const float norms = __fadd_ru(__fsqrt_ru(w2), __fsqrt_ru(p2));
    out[row] = valid_lanes == 0xffffffffu ?
        __fmaf_ru(DotRelativeError, norms, __fsqrt_ru(diff)) : CUDART_INF_F;
  }
}

struct State {
  unsigned long long best[8];
  float norms[8], cutoff[8];
  unsigned suspects[8];
  unsigned seed_count, tile_count, fallback;
  int seed_tiles[8];
};

struct Workspace {
  State* state;
  unsigned* masks;
  int* tiles;
  Candidate* seeds;
  Candidate* proxy_partials;
};

Workspace carve_workspace(void* buffer, int rows, int width) {
  auto* state = static_cast<State*>(buffer);
  auto* masks = reinterpret_cast<unsigned*>(state + 1);
  const int max_tiles = (rows - 1) / 8 + 1;
  auto* tiles = reinterpret_cast<int*>(masks + rows);
  auto* seeds = reinterpret_cast<Candidate*>(tiles + max_tiles);
  return {state, masks, tiles, seeds, seeds + width};
}

__global__ void prepare(const __nv_bfloat16* x, int m, int rows, int tile_rows,
                        const Candidate* seeds, State* state) {
  __shared__ float parts[Warps];
  const int token = blockIdx.x;
  float sum = 0.f;
  bool valid = true;
  for (int i = threadIdx.x; i < K; i += Threads) {
    const float v = __bfloat162float(x[size_t(token) * K + i]);
    valid &= fabsf(v) <= MaxOperand;
    sum = __fmaf_ru(v, v, sum);
  }
  const unsigned valid_lanes = __ballot_sync(0xffffffffu, valid);
  sum = warp_sum_up(sum);
  if (!(threadIdx.x & 31)) parts[threadIdx.x / 32] = valid_lanes == 0xffffffffu ? sum : CUDART_INF_F;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = 0.f;
    for (float p : parts) sum = __fadd_ru(sum, p);
    state->norms[token] = __fsqrt_ru(sum);
    state->best[token] = 0;
    state->suspects[token] = 0;
    const int id = seeds[token].global_id;
    state->seed_tiles[token] = (id >= 0 && id < rows ? id : 0) / tile_rows;
    if (token == 0) {
      state->seed_count = m;
      state->tile_count = 0;
      state->fallback = 0;
    }
  }
}

__device__ __forceinline__ float unordered_float(unsigned key) {
  return __uint_as_float(key & 0x80000000u ? key ^ 0x80000000u : ~key);
}

__global__ void make_cutoffs(State* state) {
  const int token = threadIdx.x;
  const auto best = state->best[token];
  // Search over BF16 encodings for a point whose softcap is strictly below the incumbent.
  // RN division is monotone. CUDA tanhf has 2 ULP maximum error, and multiplication by
  // 30 rounds once. Both endpoint errors combined are below 2^-14 over the entire range.
  // Subtracting this allowance preserves possible ties even where the FP32 softcap saturates.
  const float incumbent = best ? unordered_float(unsigned(best >> 32)) : -CUDART_INF_F;
  unsigned lo = 0x007fu;  // ordered encoding of -infinity, treated as a sentinel
  unsigned hi = 0xff80u;  // +infinity
  while (hi - lo > 1) {
    const unsigned mid = (lo + hi) / 2;
    const unsigned bits = mid & 0x8000u ? mid ^ 0x8000u : ~mid & 0xffffu;
    const float score = tanhf(__uint_as_float(bits << 16) / 30.f) * 30.f;
    if (__fadd_ru(score, 0x1p-14f) < incumbent) lo = mid;
    else hi = mid;
  }
  state->cutoff[token] = __uint_as_float((lo & 0x8000u ? lo ^ 0x8000u : ~lo & 0xffffu) << 16);
}

template <int TileRows>
__global__ void collect(const float* dots, const float* bounds, int rows, int m, Workspace work) {
  constexpr int Tiles = Threads / TileRows;
  __shared__ unsigned masks[Tiles];
  __shared__ unsigned partial[Warps];
  const int lane = threadIdx.x & 31;
  const int idx = blockIdx.x * Threads + threadIdx.x;
  const float factor = idx < rows ? bounds[idx] : 0.f;
  unsigned tile_mask = 0;
  unsigned row_mask = 0;
  for (int token = 0; token < m; ++token) {
    bool keep = false;
    if (idx < rows) {
      const float norm = work.state->norms[token];
      const float dot = dots[size_t(token) * rows + idx];
      const float upper = __fadd_ru(__fmaf_ru(norm, factor, dot), DotAbsoluteError);
      keep = !isfinite(factor) || !isfinite(norm) || !isfinite(upper) ||
             dense::round_bf16(upper) > work.state->cutoff[token];
    }
    const unsigned hits = __ballot_sync(0xffffffffu, keep);
    const unsigned tile_hits = ((1u << TileRows) - 1) << (lane / TileRows * TileRows);
    if (hits & tile_hits) tile_mask |= 1u << token;
    if (keep) row_mask |= 1u << token;
    if (lane == 0) partial[threadIdx.x / 32] = __popc(hits);
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned total = 0;
      for (unsigned count : partial) total += count;
      if (total) atomicAdd(&work.state->suspects[token], total);
    }
    __syncthreads();
  }
  if (idx < rows) work.masks[idx] = row_mask;
  if (threadIdx.x % TileRows == 0) masks[threadIdx.x / TileRows] = tile_mask;
  __syncthreads();
  if (threadIdx.x < 32) {
    const unsigned live = __ballot_sync(0xffffffffu, lane < Tiles && masks[lane]);
    if (live) {
      unsigned first = lane == 0 ? atomicAdd(&work.state->tile_count, __popc(live)) : 0;
      first = __shfl_sync(0xffffffffu, first, 0);
      if (live & (1u << lane))
        work.tiles[first + __popc(live & ((1u << lane) - 1))] = blockIdx.x * Tiles + lane;
    }
  }
}

__global__ void mark_fallback(State* state, int m, unsigned limit, unsigned* counts) {
  unsigned fallback = 0;
  for (int i = 0; i < m; ++i) {
    const unsigned n = state->suspects[i];
    if (counts) counts[i] = n;
    if (n > limit) fallback |= 1u << i;
  }
  state->fallback = fallback;
}

__global__ void publish(const State* state, int32_t* ids) {
  const auto best = state->best[threadIdx.x];
  ids[threadIdx.x] = best ? int32_t(~unsigned(best)) : 0;
}

}  // namespace

void cert_head_bounds(const __nv_bfloat16* exact, Nvfp4Matrix proxy, float* bounds, cudaStream_t stream) {
  if (!exact || !proxy.codes || !proxy.scales || proxy.rows < 1 || !bounds)
    throw std::invalid_argument("invalid certified head calibration arguments");
  row_bounds<<<(proxy.rows - 1) / Warps + 1, Threads, 0, stream>>>(exact, proxy, bounds);
  CUDA_CHECK(cudaGetLastError());
}

std::size_t cert_head_workspace(int rows, int max_width) {
  const int tiles = (rows - 1) / 8 + 1;
  return sizeof(State) + size_t(tiles) * sizeof(int) + size_t(rows) * sizeof(unsigned) +
         size_t(max_width) * sizeof(Candidate) + nvfp4_candidate_workspace(max_width, rows);
}

void cert_head(const __nv_bfloat16* x, int M, dense::Matrix exact, CertifiedHead cert, float* dots,
               void* workspace, int32_t* ids, cudaStream_t stream, unsigned* counts, unsigned suspect_limit) {
  const int rows = exact.rows;
  const auto proxy = cert.proxy;
  if (!x || M < 1 || M > 8 || !exact.w || exact.scale || rows < 1 || proxy.rows != rows ||
      proxy.v_begin != 0 || !proxy.codes || !proxy.scales || !cert.bounds || !dots || !workspace || !ids)
    throw std::invalid_argument("invalid certified head arguments");
  const Workspace w = carve_workspace(workspace, rows, M);
  const int tile_rows = M == 1 ? 8 : 16;
  head_nvfp4_raw(x, M, proxy, dots, w.seeds, w.proxy_partials, stream);
  prepare<<<M, Threads, 0, stream>>>(x, M, rows, tile_rows, w.seeds, w.state);
  dense::head_tiles(x, M, exact,
                    {w.state->seed_tiles, &w.state->seed_count, nullptr, nullptr, w.state->best}, M, stream);
  make_cutoffs<<<1, M, 0, stream>>>(w.state);
  const int scan_blocks = (rows - 1) / Threads + 1;
  if (M == 1) collect<8><<<scan_blocks, Threads, 0, stream>>>(dots, cert.bounds, rows, M, w);
  else collect<16><<<scan_blocks, Threads, 0, stream>>>(dots, cert.bounds, rows, M, w);
  mark_fallback<<<1, 1, 0, stream>>>(w.state, M, suspect_limit, counts);
  dense::head_tiles(x, M, exact, {w.tiles, &w.state->tile_count, w.masks, &w.state->fallback, w.state->best},
                    std::min(ExactBlocks, (rows - 1) / tile_rows + 1), stream);
  publish<<<1, M, 0, stream>>>(w.state, ids);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::head
