#include "kernels/draft/dflash2_selector.h"

#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int kSlots = GAMMA;
constexpr int kCandidates = DFLASH2_SUPPORT;
constexpr int kRank = 256;

__device__ __forceinline__ float score_edge(float unary, const __nv_bfloat16* projected,
                                            const __nv_bfloat16* predecessor,
                                            const __nv_bfloat16* successor, int lane, float edge_scale) {
  float dot = 0.0f;
  for (int r = lane; r < kRank; r += 32) {
    const auto weighted = __float2bfloat16_rn(__bfloat162float(predecessor[r]) * __bfloat162float(projected[r]));
    dot = fmaf(__bfloat162float(weighted), __bfloat162float(successor[r]), dot);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    dot += __shfl_down_sync(0xffffffffu, dot, offset);
  }
  const float corr = __bfloat162float(__float2bfloat16_rn(dot));
  const float combined = edge_scale == 1.0f ? unary + corr : unary + edge_scale * corr;
  return lane == 0 ? 30.0f * tanhf(combined / 30.0f) : 0.f;
}

__global__ void score_edges(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice, float edge_scale) {
  const int edge = blockIdx.x;
  const int previous = blockIdx.y;
  const int next = blockIdx.z;
  const int lane = threadIdx.x;
  const int predecessor = edge == 0 ? *anchor : ids[(edge - 1) * kCandidates + previous];
  const int candidate = ids[edge * kCandidates + next];
  const float score = score_edge(unary[edge * kCandidates + next], projected + edge * kRank,
                                 predecessors + predecessor * kRank, successors + candidate * kRank, lane, edge_scale);
  if (lane == 0) lattice[(edge * kCandidates + previous) * kCandidates + next] = score;
}

__device__ double uniform_for_edge(std::uint64_t seed, int edge) {
  std::uint64_t x = seed + (edge + 1) * 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return double((x ^ (x >> 31)) >> 11) * 0x1p-53;
}

__device__ int pick(const std::int32_t* ids, const float* scores, int edge, float temperature,
                    std::uint64_t seed, DFlash2Support support, float* confidence) {
  int best = 0;
  for (int next = 1; next < kCandidates; ++next) {
    if (scores[next] > scores[best]) best = next;
  }
  if (!(temperature > 0.f)) {
    if (confidence) {
      float mass = 0;
      for (int next = 0; next < kCandidates; ++next) mass += expf(scores[next] - scores[best]);
      *confidence = 1.f / mass;
    }
    return best;
  }

  auto* probs = support.probs + edge * kCandidates;
  auto* out_ids = support.ids + edge * kCandidates;
  // Fold any duplicate entries into the first occurrence, so each token has
  // exactly one probability for the accept ratio and the sparse residual.
  double weights[kCandidates]{};
  double total = 0.0;
  for (int j = 0; j < kCandidates; ++j) {
    const float weight = expf(__fdiv_rn(scores[j] - scores[best], temperature));
    int first = 0;
    while (ids[first] != ids[j]) ++first;
    weights[first] += weight;
    total += weight;
    out_ids[j] = ids[j];
  }
  for (int j = 0; j < kCandidates; ++j) probs[j] = float(weights[j] / total);

  const double draw = uniform_for_edge(seed, edge) * total;
  double prefix = 0.0;
  int picked = 0;
  for (int j = 0; j < kCandidates; ++j) {
    if (weights[j] == 0.0) continue;
    picked = j;
    prefix += weights[j];
    if (draw < prefix) break;
  }
  support.picked[edge] = picked;
  if (confidence) *confidence = probs[picked];
  return picked;
}

__global__ void walk(const std::int32_t* ids, const float* lattice, std::int32_t* tokens, float* confidence,
                     bool cumulative, const float* temperature, const std::uint64_t* seed,
                     DFlash2Support support) {
  const float temp = temperature ? *temperature : 0.f;
  const std::uint64_t rng = temp > 0.f ? *seed : 0;
  float product = 1.f;
  int previous = 0;
  for (int edge = 0; edge < kSlots; ++edge) {
    const auto* row = ids + edge * kCandidates;
    const float* scores = lattice + (edge * kCandidates + previous) * kCandidates;
    float conf;
    previous = pick(row, scores, edge, temp, rng, support, confidence ? &conf : nullptr);
    tokens[edge] = row[previous];
    if (confidence) {
      product *= conf;
      confidence[edge] = cumulative ? product : conf;
    }
  }
}

__global__ void fused_walk(const std::int32_t* ids, const float* unary,
                           const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                           const __nv_bfloat16* successors, const std::int32_t* anchor,
                           std::int32_t* tokens, float* confidence, bool cumulative,
                           const float* temperature, const std::uint64_t* seed, DFlash2Support support, float edge_scale) {
  __shared__ float scores[kCandidates];
  __shared__ int predecessor;
  const int next = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const float temp = temperature ? *temperature : 0.f;
  const std::uint64_t rng = temp > 0.f ? *seed : 0;
  float product = 1.f;
  if (threadIdx.x == 0) predecessor = *anchor;
  __syncthreads();
  for (int edge = 0; edge < kSlots; ++edge) {
    const auto* row = ids + edge * kCandidates;
    const float score = score_edge(unary[edge * kCandidates + next], projected + edge * kRank,
                                   predecessors + predecessor * kRank, successors + row[next] * kRank, lane, edge_scale);
    if (lane == 0) scores[next] = score;
    __syncthreads();
    if (threadIdx.x == 0) {
      float conf;
      const int chosen = pick(row, scores, edge, temp, rng, support, confidence ? &conf : nullptr);
      tokens[edge] = predecessor = row[chosen];
      if (confidence) {
        product *= conf;
        confidence[edge] = cumulative ? product : conf;
      }
    }
    __syncthreads();
  }
}

// One warp per depth. Each distinct token contributes p and min(p, total q) once.
__global__ void gather_audit(const float* confidence, const int* ids, int eligible, int vocab,
                             const int* target, const float* p, const float* q, const int* picked,
                             DFlash2Audit* out) {
  const int row = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool valid = row < eligible && lane < kCandidates;
  const int slot = row * kCandidates + lane;
  const int id = valid ? ids[slot] : -1;
  const bool inside = id >= 0 && id < vocab;
  const unsigned hits = __ballot_sync(0xffffffffu, inside && target && target[row] == id);
  const unsigned peers = __match_any_sync(0xffffffffu, id);
  const float prob = inside && p ? p[std::size_t(row) * vocab + id] : 0.f;
  float mass = 0.f, overlap = 0.f;
  if (inside && p && lane == __ffs(peers) - 1) {
    float total_q = 0.f;
    for (unsigned mask = peers; mask; mask &= mask - 1)
      total_q += q[row * kCandidates + __ffs(mask) - 1];
    mass = prob;
    overlap = fminf(prob, total_q);
  }
  for (int offset = 16; offset; offset /= 2) {
    mass += __shfl_down_sync(0xffffffffu, mass, offset);
    overlap += __shfl_down_sync(0xffffffffu, overlap, offset);
  }
  const bool exact = p && row < eligible;
  const int pick = exact ? picked[row] : -1;
  const bool has_pick = pick >= 0 && pick < kCandidates;
  const float selected_p = __shfl_sync(0xffffffffu, prob, has_pick ? pick : 0);
  if (lane == 0) {
    out->conf[row] = confidence[row];
    out->in_support[row] = row < eligible && target ? int(hits != 0) : -1;
    out->target_mass[row] = exact ? mass : -1.f;
    out->overlap[row] = exact ? overlap : -1.f;
    out->picked_p[row] = has_pick ? selected_p : -1.f;
    out->picked_q[row] = has_pick ? q[row * kCandidates + pick] : -1.f;
  }
}

constexpr int kResidualThreads = 256;
constexpr int kResidualSpan = 4 * kResidualThreads;

__host__ __device__ int residual_blocks(int vocab) { return (vocab - 1) / kResidualSpan + 1; }

__global__ void sparse_residual(const float* p, int vocab, const std::int32_t* drafts, int eligible,
                                const std::int32_t* ids, const float* probs, const std::int32_t* picked,
                                const float* uniforms, float* ws, float* residual, AcceptResult* out) {
  __shared__ int accepted;
  __shared__ float tile[kResidualSpan];
  __shared__ float sums[33];
  const int lane = threadIdx.x;
  if (lane == 0) {
    int a = 0;
    for (; a < eligible; ++a) {
      const int token = drafts[a];
      const int j = picked[a];
      if (token < 0 || token >= vocab || j < 0 || j >= kCandidates || ids[a * kCandidates + j] != token) break;
      const float q = probs[a * kCandidates + j];
      const float target = p[std::size_t(a) * vocab + token];
      if (!(q > 0.f && (target >= q || __fmul_rn(uniforms[a], q) < target))) break;
    }
    accepted = a;
    if (blockIdx.x == 0) out->accepted = a;
  }
  __syncthreads();
  const auto* target = p + std::size_t(accepted) * vocab;
  const int start = blockIdx.x * kResidualSpan;
  const int count = min(vocab - start, kResidualSpan);
  if (accepted == eligible) {
    for (int i = lane; i < count; i += kResidualThreads) residual[start + i] = target[start + i];
    return;
  }

  for (int i = lane; i < count; i += kResidualThreads) tile[i] = target[start + i];
  __syncthreads();
  if (lane < kCandidates) {
    const auto* row_ids = ids + accepted * kCandidates;
    const auto* row_probs = probs + accepted * kCandidates;
    const int id = row_ids[lane];
    if (id >= start && id < start + count && row_probs[lane] > 0.f) {
      const int offset = id - start;
      tile[offset] = fmaxf(0.f, tile[offset] - row_probs[lane]);
    }
  }
  __syncthreads();
  float sum = 0.f;
  for (int i = lane; i < count; i += kResidualThreads) {
    const float mass = tile[i];
    residual[start + i] = mass;
    sum += mass;
  }
  const float total = block_sum(sum, sums);
  if (lane == 0) ws[blockIdx.x] = total;
}

__global__ void normalize_residual(const float* p, int vocab, int eligible, const float* ws,
                                    float* residual, const AcceptResult* out) {
  const int a = out->accepted;
  if (a == eligible) return;
  __shared__ float sums[33];
  float sum = 0.f;
  for (int i = threadIdx.x; i < residual_blocks(vocab); i += kResidualThreads) sum += ws[i];
  const float mass = block_sum(sum, sums);
  const int start = blockIdx.x * kResidualSpan;
  const int end = start + min(vocab - start, kResidualSpan);
  const auto* fallback = p + std::size_t(a) * vocab;
  for (int i = start + threadIdx.x; i < end; i += kResidualThreads)
    residual[i] = mass > 0.f ? residual[i] / mass : fallback[i];
}

}  // namespace

void dflash2_greedy_selector(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice, std::int32_t* tokens, cudaStream_t stream, float* confidence) {
  dflash2_lattice_selector(ids, unary, projected, predecessors, successors, anchor, lattice, tokens,
                           confidence, false, nullptr, nullptr, {}, stream);
}

void dflash2_fused_selector(const std::int32_t* ids, const float* unary,
                           const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                           const __nv_bfloat16* successors, const std::int32_t* anchor,
                           std::int32_t* tokens, float* confidence, bool cumulative,
                           const float* temperature, const std::uint64_t* seed,
                           DFlash2Support support, cudaStream_t stream, float edge_scale) {
  fused_walk<<<1, kCandidates * 32, 0, stream>>>(ids, unary, projected, predecessors, successors, anchor,
                                                tokens, confidence, cumulative, temperature, seed, support, edge_scale);
  CUDA_CHECK(cudaGetLastError());
}

void dflash2_lattice_selector(const std::int32_t* ids, const float* unary,
                             const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                             const __nv_bfloat16* successors, const std::int32_t* anchor,
                             float* lattice, std::int32_t* tokens, float* confidence, bool cumulative,
                             const float* temperature, const std::uint64_t* seed,
                             DFlash2Support support, cudaStream_t stream, float edge_scale) {
  score_edges<<<dim3(kSlots, kCandidates, kCandidates), 32, 0, stream>>>(
      ids, unary, projected, predecessors, successors, anchor, lattice, edge_scale);
  walk<<<1, 1, 0, stream>>>(ids, lattice, tokens, confidence, cumulative, temperature, seed, support);
  CUDA_CHECK(cudaGetLastError());
}

void dflash2_audit(const float* confidence, const std::int32_t* ids, int eligible, int vocab,
                   const std::int32_t* target, const float* p, const float* q,
                   const std::int32_t* picked, DFlash2Audit* out, cudaStream_t stream) {
  if (eligible < 0 || eligible > kSlots || vocab < 1)
    throw std::invalid_argument("dflash2_audit: invalid verify shape");
  gather_audit<<<1, kSlots * 32, 0, stream>>>(confidence, ids, eligible, vocab, target, p, q, picked, out);
  CUDA_CHECK(cudaGetLastError());
}

std::size_t dflash2_accept_workspace_floats(int vocab) { return std::size_t(residual_blocks(vocab)); }

void dflash2_exact_accept(const float* p, int vocab, const std::int32_t* drafts, int eligible,
                         const std::int32_t* support_ids, const float* support_probs,
                         const std::int32_t* picked, const float* uniforms, float* ws,
                         float* residual, AcceptResult* out, cudaStream_t stream) {
  if (eligible < 0 || eligible > kSlots || vocab < 1)
    throw std::invalid_argument("dflash2_exact_accept: invalid verify shape");
  const int blocks = residual_blocks(vocab);
  sparse_residual<<<blocks, kResidualThreads, 0, stream>>>(
      p, vocab, drafts, eligible, support_ids, support_probs, picked, uniforms, ws, residual, out);
  normalize_residual<<<blocks, kResidualThreads, 0, stream>>>(p, vocab, eligible, ws, residual, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::draft
