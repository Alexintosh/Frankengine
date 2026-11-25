#include "model/dflash2.h"
#include "model/tp_runtime.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/dflash2_selector.h"
#include "kernels/draft/gemm.h"
#include "kernels/draft/draft.h"
#include "kernels/head/nvfp4_head.h"

#include <math_constants.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ingine::model {
namespace {
using Bf = __nv_bfloat16;
namespace head = kernels::head;
constexpr int H = 2304, I = 6912, Q = 4096, KV = 1024, W = 2048;
constexpr int D = 128, QHeads = Q / D, KVHeads = KV / D;
constexpr int Group = 16, Groups = H / Group, Coefficients = 4 * Groups;
constexpr int CommitBatch = 128, Threads = 128;
constexpr int Block = DFlash2::block, Proposals = Block - 1;
constexpr int Rank = 256, Vocabulary = 201024;

__global__ void normalize(const Bf* x, const Bf* weight, int width, int heads,
                          int position, Bf* out, int ring, const int* device_position = nullptr) {
  if (device_position) position = *device_position;
  __shared__ float sums[Threads];
  const int row = blockIdx.x / heads;
  const int head_idx = blockIdx.x % heads;
  const int lane = threadIdx.x;
  x += blockIdx.x * width;
  const int slot = ring ? (position + row) % ring : row;
  out += (slot * heads + head_idx) * width;
  float sum = 0;
  for (int i = lane; i < width; i += Threads) {
    const float value = __bfloat162float(x[i]);
    sum += value * value;
  }
  sums[lane] = sum;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sums[0] / width + 1e-6f);
  if (position < 0) {
    for (int i = lane; i < width; i += Threads)
      out[i] = __float2bfloat16(__bfloat162float(x[i]) * scale * __bfloat162float(weight[i]));
  } else {
    const int other = (lane + D / 2) % D;
    const float value = __bfloat162float(__float2bfloat16(__bfloat162float(x[lane]) * scale *
                                                        __bfloat162float(weight[lane])));
    const float pair = __bfloat162float(__float2bfloat16(__bfloat162float(x[other]) * scale *
                                                       __bfloat162float(weight[other])));
    float sine, cosine;
    sincosf((position + row) * powf(10000.f, -2.f * (lane % (D / 2)) / D), &sine, &cosine);
    out[lane] = __float2bfloat16(value * cosine + (lane < D / 2 ? -pair : pair) * sine);
  }
}

__global__ void convolution(const Bf* x, const Bf* coeff, const Bf* base, int part,
                            Bf* out, bool residual) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= H) return;
  coeff += row * Coefficients + part * 2 * Groups + col / Group;
  base += part * 2 * H + col;
  const int index = row * H + col;
  const Bf a = __hadd(base[0], coeff[0]);
  const Bf b = __hadd(base[H], coeff[Groups]);
  const Bf shifted = row ? x[index - H] : __float2bfloat16(0.f);
  const Bf value = __hadd_rn(__hmul_rn(a, x[index]), __hmul_rn(b, shifted));
  out[index] = residual ? __hadd(out[index], value) : value;
}

__global__ void swiglu(const Bf* gate, const Bf* up, Bf* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __bfloat162float(gate[i]);
  out[i] = __hmul(__float2bfloat16(x / (1.f + expf(-x))), up[i]);
}

__global__ void convert(const float* x, Bf* out, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = __float2bfloat16(x[i]);
}

// One CTA per query. Warp-cooperative dot products followed by a stable
// softmax and a channel-parallel value reduction.
__global__ void attend(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                       const Bf* block_k, const Bf* block_v, int position, Bf* out,
                       const int* device_position = nullptr) {
  if (device_position) position = *device_position;
  __shared__ float scores[W + Block];
  __shared__ float reductions[Threads];
  const int tid = threadIdx.x;
  const int qrow = blockIdx.x / QHeads;
  const int kvhead = (blockIdx.x % QHeads) / (QHeads / KVHeads);
  const int begin = max(0, position + qrow - W + 1);
  const int count = position + Block - begin;
  query += blockIdx.x * D;
  for (int i = tid / 32; i < count; i += Threads / 32) {
    const int pos = begin + i;
    const Bf* key = (pos < position ? ring_k + (pos % W) * KV : block_k + (pos - position) * KV)
                   + kvhead * D;
    float dot = 0;
    for (int j = tid % 32; j < D; j += 32)
      dot += __bfloat162float(query[j]) * __bfloat162float(key[j]);
    for (int offset = 16; offset; offset /= 2)
      dot += __shfl_down_sync(0xffffffff, dot, offset);
    if (tid % 32 == 0) scores[i] = dot * rsqrtf(float(D));
  }
  __syncthreads();
  float local_max = -CUDART_INF_F;
  for (int i = tid; i < count; i += Threads) local_max = fmaxf(local_max, scores[i]);
  reductions[tid] = local_max;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (tid < stride) reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
    __syncthreads();
  }
  const float peak = reductions[0];
  __syncthreads();
  float sum = 0;
  for (int i = tid; i < count; i += Threads) {
    scores[i] = expf(scores[i] - peak);
    sum += scores[i];
  }
  reductions[tid] = sum;
  __syncthreads();
  for (int stride = Threads / 2; stride; stride /= 2) {
    if (tid < stride) reductions[tid] += reductions[tid + stride];
    __syncthreads();
  }
  float value = 0;
  for (int i = 0; i < count; ++i) {
    const int pos = begin + i;
    const Bf* row = pos < position ? ring_v + (pos % W) * KV : block_v + (pos - position) * KV;
    value += scores[i] * __bfloat162float(row[kvhead * D + tid]);
  }
  out[blockIdx.x * D + tid] = __float2bfloat16(value / reductions[0]);
}

// Split long windows across CTAs. Each warp streams one subset of rows
// and accumulates a local online softmax in registers.
constexpr int AttentionTile = 128;
constexpr int MaxSplits = (W + Block + AttentionTile - 1) / AttentionTile;

__global__ void attend_split(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                             const Bf* block_k, const Bf* block_v, int position,
                             float* partial, const int* device_position = nullptr) {
  if (device_position) position = *device_position;
  constexpr int Warps = Threads / 32;
  __shared__ float numerators[Warps][D];
  __shared__ float peaks[Warps], sums[Warps];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int query_idx = blockIdx.x;
  const int kvhead = (query_idx % QHeads) / (QHeads / KVHeads);
  const int begin = max(0, position + query_idx / QHeads - W + 1);
  const int first = begin + blockIdx.y * AttentionTile;
  const int end = min(first + AttentionTile, position + Block);
  float peak = -CUDART_INF_F, sum = 0.f;
  float acc[4] = {};
  const auto* q = reinterpret_cast<const __nv_bfloat162*>(query + query_idx * D);
  const float2 q0 = __bfloat1622float2(q[2 * lane]);
  const float2 q1 = __bfloat1622float2(q[2 * lane + 1]);
  for (int pos = first + warp; pos < end; pos += Warps) {
    const int offset = (pos < position ? pos % W : pos - position) * KV + kvhead * D;
    const auto* k = reinterpret_cast<const __nv_bfloat162*>(
        (pos < position ? ring_k : block_k) + offset);
    const auto* v = reinterpret_cast<const __nv_bfloat162*>(
        (pos < position ? ring_v : block_v) + offset);
    const float2 k0 = __bfloat1622float2(k[2 * lane]);
    const float2 k1 = __bfloat1622float2(k[2 * lane + 1]);
    float dot = q0.x * k0.x + q0.y * k0.y + q1.x * k1.x + q1.y * k1.y;
    for (int delta = 16; delta; delta /= 2) dot += __shfl_down_sync(0xffffffff, dot, delta);
    const float score = __shfl_sync(0xffffffff, dot, 0) * rsqrtf(float(D));
    const float next = fmaxf(peak, score);
    const float carry = __expf(peak - next);
    const float weight = __expf(score - next);
    sum = sum * carry + weight;
    peak = next;
    const float2 v0 = __bfloat1622float2(v[2 * lane]);
    const float2 v1 = __bfloat1622float2(v[2 * lane + 1]);
    acc[0] = acc[0] * carry + weight * v0.x;
    acc[1] = acc[1] * carry + weight * v0.y;
    acc[2] = acc[2] * carry + weight * v1.x;
    acc[3] = acc[3] * carry + weight * v1.y;
  }
  if (lane == 0) {
    peaks[warp] = peak;
    sums[warp] = sum;
  }
  for (int i = 0; i < 4; ++i) numerators[warp][lane * 4 + i] = acc[i];
  __syncthreads();
  peak = -CUDART_INF_F;
  for (int i = 0; i < Warps; ++i) peak = fmaxf(peak, peaks[i]);
  sum = 0.f;
  float numerator = 0.f;
  for (int i = 0; i < Warps; ++i) {
    const float factor = sums[i] > 0.f ? __expf(peaks[i] - peak) : 0.f;
    sum += factor * sums[i];
    numerator += factor * numerators[i][threadIdx.x];
  }
  partial += (query_idx * MaxSplits + blockIdx.y) * (D + 2);
  partial[threadIdx.x] = numerator;
  if (threadIdx.x == 0) {
    partial[D] = peak;
    partial[D + 1] = sum;
  }
}

// Reuse each 128-bit K/V load across all 32 queries in one KV group.
constexpr int GqaTile = 32, QPerKV = QHeads / KVHeads, GqaRows = Block * QPerKV;
constexpr int GqaSplits = (W + Block + GqaTile - 1) / GqaTile;

__global__ void gqa_attention(const Bf* query, const Bf* ring_k, const Bf* ring_v,
                              const Bf* block_k, const Bf* block_v, int position,
                              float* partial, const int* device_position) {
  if (device_position) position = *device_position;
  using namespace nvcuda::wmma;
  __shared__ __align__(32) Bf qs[GqaRows][D], ks[GqaTile][D], vs[GqaTile][D];
  __shared__ __align__(32) float scores[GqaRows][GqaTile];
  __shared__ __align__(32) Bf probs[GqaRows][GqaTile];
  __shared__ __align__(32) float scratch[4][16][16];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int head = blockIdx.x;
  const int begin = max(0, position - W + 1);
  const int first = begin + blockIdx.y * GqaTile;
  const int end = min(first + GqaTile, position + Block);
  if (first >= end) return;
  for (int i = tid; i < GqaTile * D / 8; i += Threads) {
    const int row = i / (D / 8), col = i % (D / 8);
    const int pos = first + row;
    const int qidx = (row / QPerKV) * QHeads + head * QPerKV + row % QPerKV;
    reinterpret_cast<uint4*>(qs[row])[col] = reinterpret_cast<const uint4*>(query + qidx * D)[col];
    uint4 k = make_uint4(0, 0, 0, 0), v = k;
    if (pos < end) {
      const int off = (pos < position ? pos % W : pos - position) * KV + head * D;
      k = reinterpret_cast<const uint4*>((pos < position ? ring_k : block_k) + off)[col];
      v = reinterpret_cast<const uint4*>((pos < position ? ring_v : block_v) + off)[col];
    }
    reinterpret_cast<uint4*>(ks[row])[col] = k;
    reinterpret_cast<uint4*>(vs[row])[col] = v;
  }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, Bf, row_major> a;
  fragment<matrix_b, 16, 16, 16, Bf, col_major> b;
  fragment<accumulator, 16, 16, 16, float> c;
  fill_fragment(c, 0.f);
  for (int d = 0; d < D; d += 16) {
    load_matrix_sync(a, &qs[16 * (warp / 2)][d], D);
    load_matrix_sync(b, &ks[16 * (warp % 2)][d], D);
    mma_sync(c, a, b, c);
  }
  store_matrix_sync(&scores[16 * (warp / 2)][16 * (warp % 2)], c, GqaTile, mem_row_major);
  __syncthreads();
  // Each warp reduces one query row at a time. The left edge advances
  // for each proposal token, while all proposal K/V remain visible.
  for (int qr = warp; qr < GqaRows; qr += 4) {
    const int qrow = qr / QPerKV, qidx = qrow * QHeads + head * QPerKV + qr % QPerKV;
    const int pos = first + lane;
    const bool valid = pos < end && pos >= max(0, position + qrow - W + 1);
    const float score = valid ? scores[qr][lane] * rsqrtf(float(D)) : -CUDART_INF_F;
    float peak = score;
    for (int i = 16; i; i /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, i));
    const float value = valid ? __expf(score - peak) : 0.f;
    probs[qr][lane] = __float2bfloat16(value);
    float sum = value;
    for (int i = 16; i; i /= 2) sum += __shfl_xor_sync(0xffffffff, sum, i);
    if (lane == 0) {
      float* dest = partial + (qidx * GqaSplits + blockIdx.y) * (D + 2);
      dest[D] = peak;
      dest[D + 1] = sum;
    }
  }
  __syncthreads();
  // Multiply the BF16 exponentials relative to the tile maximum by the shared
  // values. The cross-tile merge uses the FP32 peaks and denominators.
  fragment<matrix_b, 16, 16, 16, Bf, row_major> v;
  for (int d = (warp % 2) * 64; d < (warp % 2 + 1) * 64; d += 16) {
    fill_fragment(c, 0.f);
    for (int t = 0; t < GqaTile; t += 16) {
      load_matrix_sync(a, &probs[(warp / 2) * 16][t], GqaTile);
      load_matrix_sync(v, &vs[t][d], D);
      mma_sync(c, a, v, c);
    }
    store_matrix_sync(&scratch[warp][0][0], c, 16, mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
      const int qr = (warp / 2) * 16 + i / 16;
      const int qidx = (qr / QPerKV) * QHeads + head * QPerKV + qr % QPerKV;
      partial[(qidx * GqaSplits + blockIdx.y) * (D + 2) + d + i % 16] = scratch[warp][i / 16][i % 16];
    }
    __syncwarp();
  }
}

template <int Slots>
__global__ void merge_attention(const float* partial, int splits, Bf* out,
                                const int* device_position) {
  if (device_position)
    splits = (min(*device_position, W - 1) + Block + GqaTile - 1) / GqaTile;
  partial += blockIdx.x * Slots * (D + 2);
  __shared__ float factors[Slots], denominator;
  if (threadIdx.x == 0) {
    float peak = -CUDART_INF_F;
    for (int i = 0; i < splits; ++i) peak = fmaxf(peak, partial[i * (D + 2) + D]);
    float sum = 0.f;
    for (int i = 0; i < splits; ++i) {
      const float* tile = partial + i * (D + 2);
      factors[i] = __expf(tile[D] - peak);
      sum += factors[i] * tile[D + 1];
    }
    denominator = sum;
  }
  __syncthreads();
  float numerator = 0.f;
  for (int i = 0; i < splits; ++i)
    numerator += factors[i] * partial[i * (D + 2) + threadIdx.x];
  out[blockIdx.x * D + threadIdx.x] = __float2bfloat16(numerator / denominator);
}

__global__ void input_ids(int64_t* ids, int* anchor_out, int anchor,
                          int* position_out, int position) {
  ids[threadIdx.x] = threadIdx.x ? DFlash2::mask_token : anchor;
  if (threadIdx.x == 0) {
    *anchor_out = anchor;
    if (position_out) *position_out = position;
  }
}

__global__ void best16(const float* values, const int* ids, float* out_values, int* out_ids) {
  constexpr int Pool = kernels::draft::CANDIDATES;
  __shared__ float scores[Pool];
  __shared__ head::Candidate best[Threads];
  const int tid = threadIdx.x;
  values += blockIdx.x * Pool;
  ids += blockIdx.x * Pool;
  for (int i = tid; i < Pool; i += Threads) scores[i] = values[i];
  __syncthreads();
  for (int rank = 0; rank < 16; ++rank) {
    head::Candidate local{-CUDART_INF_F, INT_MAX};
    for (int i = tid; i < Pool; i += Threads)
      if (scores[i] > local.val || (scores[i] == local.val && ids[i] < local.global_id))
        local = {scores[i], ids[i]};
    best[tid] = local;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride /= 2) {
      if (tid < stride) {
        const auto other = best[tid + stride];
        if (other.val > best[tid].val ||
            (other.val == best[tid].val && other.global_id < best[tid].global_id))
          best[tid] = other;
      }
      __syncthreads();
    }
    const auto winner = best[0];
    __syncthreads();
    if (tid == 0) {
      out_values[blockIdx.x * 16 + rank] = winner.val;
      out_ids[blockIdx.x * 16 + rank] = winner.global_id;
    }
    for (int i = tid; i < Pool; i += Threads)
      if (ids[i] == winner.global_id) scores[i] = -CUDART_INF_F;
    __syncthreads();
  }
}

struct Linear {
  const Bf* bf16 = nullptr;
  head::Nvfp4Matrix packed;
  kernels::dense::Matrix fp8;
  int rows = 0, cols = 0;
};

struct Conv {
  const Bf* base;
  Linear projection;
};

struct DraftLayer {
  const Bf *pre, *post, *qnorm, *knorm;
  Conv attention, mlp;
  Linear q, k, v, o, gate, up, down;
  Bf *keys, *values;
};
}  // namespace

namespace {
struct DraftProfile {
  enum Stage { Commit, Embedding, Linear, Attention, Conv, Head, Selector, Count };
  struct Span {
    cudaEvent_t begin = nullptr, end = nullptr;
    Stage stage;
    ~Span() {
      if (begin) cudaEventDestroy(begin);
      if (end) cudaEventDestroy(end);
    }
  };
  cudaStream_t stream;
  bool enabled;
  bool capturing = false;
  std::array<Span, 45> spans;
  int used = 0, rounds = 0, commits = 0;
  int position = 0;
  std::array<double, Count> totals{};

  void begin(Stage stage) {
    if (!enabled) return;
    auto& span = spans.at(used);
    if (!span.begin) CUDA_CHECK(cudaEventCreate(&span.begin));
    if (!span.end) CUDA_CHECK(cudaEventCreate(&span.end));
    span.stage = stage;
    CUDA_CHECK(cudaEventRecordWithFlags(span.begin, stream, capturing ? cudaEventRecordExternal : 0));
  }

  void end() {
    if (enabled)
      CUDA_CHECK(cudaEventRecordWithFlags(spans.at(used++).end, stream, capturing ? cudaEventRecordExternal : 0));
  }

  void finish(int current_position) {
    if (!enabled || used == 0) return;
    // Profiling is opt-in. This wait is excluded from the GPU stage totals.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < used; ++i) {
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, spans[i].begin, spans[i].end));
      totals[spans[i].stage] += ms;
    }
    used = 0;
    position = current_position;
    ++rounds;
  }

  ~DraftProfile() { print(); }

  void print() const {
    if (!rounds) return;
    const char* names[Count] = {"commit", "embed/final_norm", "5L linear", "5L attn",
                                "5L conv/norm", "head/topk", "selector"};
    std::fprintf(stderr, "DFlash2 profile (%d rounds, final position %d)", rounds, position);
    double total = 0;
    for (int i = 0; i < Count; ++i) {
      const double mean = totals[i] / rounds;
      total += mean;
      std::fprintf(stderr, " | %s %.4f ms", names[i], mean);
    }
    std::fprintf(stderr, " | stages %.4f ms", total);
    if (commits)
      std::fprintf(stderr, " | commit actual %.4f ms (%d commits)", totals[Commit] / commits, commits);
    std::fprintf(stderr, "\n");
  }

  void reset() {
    print();
    used = rounds = commits = 0;
    totals.fill(0);
  }
};
}  // namespace

namespace {
__global__ void cumulative_confidence(float* confidence) {
  float product = 1.f;
  for (int i = 0; i < Proposals; ++i) {
    product *= confidence[i];
    confidence[i] = product;
  }
}
}  // namespace

struct DFlash2::Impl {
  cudaStream_t s;
  DraftProfile profile;
  int length = 0;
  int* device_position = nullptr;
  bool warmed = false;
  bool gqa = false;
  std::array<cudaGraphExec_t, MaxSplits + 1> graphs{};
  std::array<int, MaxSplits + 1> graph_profile_spans{};
  ~Impl() {
    for (auto graph : graphs)
      if (graph) cudaGraphExecDestroy(graph);
  }

  int attention_splits() const {
    if (length + Block <= AttentionTile) return 0;
    return (std::min(length, W - 1) + Block + AttentionTile - 1) / AttentionTile;
  }
  std::vector<DeviceBuffer> buffers;
  kernels::draft::Bf16Gemm gemm;
  std::array<DraftLayer, 5> layers;
  Linear fc, selector;
  Bf* padded_taps = nullptr;
  const Bf *hidden_norm, *final_norm, *pred, *succ;
  Bf *ctx, *commit_k, *commit_v, *x, *normed, *convolved, *coeff;
  Bf *q, *key, *value, *block_key, *attention, *projection, *gate, *up, *projected;
  float* attention_partial;
  float *fp4_scratch, *lattice, *confidence;
  int* tokens;
  head::Nvfp4Matrix target_head;
  DFlash2Head* sharded_head = nullptr;
  const Bf* embed = nullptr;
  const float* embed_norm = nullptr;
  Bf* embeddings = nullptr;
  int64_t* ids;
  float* embed_output;
  std::filesystem::path dump_dir;
  const bool cumulative = std::getenv("INKLING_DFLASH2_CUMULATIVE_CONFIDENCE") != nullptr;
  bool dump_written = false;
  int dump_round = 0, proposal_round = 0;
  float *unary, *pool_values, *top_values;
  int *anchor_dev, *pool_ids, *top_ids;
  void* top_workspace;

  template <typename T = Bf>
  T* allocate(std::size_t n) {
    buffers.emplace_back(n * sizeof(T));
    return static_cast<T*>(buffers.back().p);
  }

  Impl(const std::filesystem::path& path, cudaStream_t stream, bool nvfp4) : s(stream),
        profile{stream, std::getenv("INKLING_DFLASH2_PROFILE") != nullptr} {
    if (const auto* dir = std::getenv("INKLING_DFLASH2_DUMP")) {
      dump_dir = dir;
      if (const auto* round = std::getenv("INKLING_DFLASH2_DUMP_ROUND")) {
        std::size_t parsed = 0;
        dump_round = std::stoi(round, &parsed);
        if (dump_round < 0 || round[parsed] != '\0')
          throw std::invalid_argument("INKLING_DFLASH2_DUMP_ROUND must be a nonnegative integer");
      }
      std::filesystem::create_directories(dump_dir);
    }
    const auto* precision_env = std::getenv("INKLING_DFLASH2_PRECISION");
    const std::string_view precision = precision_env ? precision_env : nvfp4 ? "" : "fp8";
    if (!precision.empty() && precision != "fp8" && precision != "hybrid")
      throw std::invalid_argument("INKLING_DFLASH2_PRECISION must be fp8 or hybrid");
    if (!precision.empty() && nvfp4)
      throw std::invalid_argument("Choose either INKLING_DFLASH2_PRECISION or --draft-nvfp4");
    const bool hybrid = precision == "hybrid";
    if (dump_dir.empty())
      device_position = allocate<int>(1);
    gqa = std::getenv("INKLING_DFLASH2_GQA") != nullptr;
    SafetensorsModel model(path);
    auto upload = [&](const std::string& name, std::vector<std::int64_t> shape, DeviceBuffer& buffer) {
      const auto tensor = model.at(name);
      if (tensor->dtype != "BF16" || tensor->shape != shape)
        throw std::runtime_error("Unexpected DFlash2 tensor shape or dtype: " + name);
      buffer = DeviceBuffer(tensor->bytes);
      const auto bytes = model.read(name);
      buffer.copy_from_host(bytes.data(), bytes.size());
      return static_cast<Bf*>(buffer.p);
    };
    auto load = [&](const std::string& name, std::vector<std::int64_t> shape) {
      buffers.emplace_back();
      return upload(name, std::move(shape), buffers.back());
    };
    auto linear = [&](const std::string& name, int rows, int cols, bool pack = false) {
      Linear layer;
      layer.rows = rows;
      layer.cols = cols;
      const bool mlp = name.find(".mlp.") != std::string::npos;
      if (pack && !precision.empty() && !(hybrid && mlp)) {
        DeviceBuffer source;
        const auto* weights = upload(name, {rows, cols}, source);
        auto* codes = allocate<uint8_t>(std::size_t(rows) * cols);
        auto* scale = allocate<float>(rows);
        kernels::dense::fp8_rows(weights, rows, cols, codes, scale, s);
        layer.fp8 = {codes, scale, rows, cols};
        // Bulk context commit uses the same quantized K/V weights through the
        // BF16 GEMM path. Keep mirrors only for the two commit projections.
        if (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight")) {
          auto* mirror = allocate(std::size_t(rows) * cols);
          kernels::dense::dequant_rows(codes, scale, rows, cols, mirror, s);
          layer.bf16 = mirror;
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
      } else if (pack && (nvfp4 || hybrid)) {
        DeviceBuffer source;
        const auto* weights = upload(name, {rows, cols}, source);
        const auto elements = std::size_t(rows) * cols;
        auto* codes = allocate<uint8_t>(elements / 2);
        auto* scales = allocate<uint8_t>(elements / 16);
        const float scale = head::nvfp4_outer_scale(weights, elements, s);
        head::nvfp4_pack_rows(weights, rows, cols, scale, codes, scales, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        layer.packed = {codes, scales, scale, rows, 0};
      } else {
        layer.bf16 = load(name, {rows, cols});
      }
      return layer;
    };
    fc = linear("fc.weight", H, 4 * H);
    if (!precision.empty()) {
      auto* codes = allocate<uint8_t>(std::size_t(H) * 4 * H);
      auto* scale = allocate<float>(H);
      kernels::dense::fp8_rows(fc.bf16, H, 4 * H, codes, scale, s);
      fc.fp8 = {codes, scale, H, 4 * H};
      padded_taps = allocate(Block * 4 * H);
    }
    hidden_norm = load("hidden_norm.weight", {H});
    final_norm = load("norm.weight", {H});
    selector = linear("candidate_selector.hidden_projection.weight", Rank, H);
    pred = load("candidate_selector.predecessor_codebook", {Vocabulary, Rank});
    succ = load("candidate_selector.successor_codebook", {Vocabulary, Rank});
    for (std::size_t i = 0; i < layers.size(); ++i) {
      auto& l = layers[i];
      const std::string prefix = "layers." + std::to_string(i) + ".";
      auto conv = [&](const std::string& name) {
        return Conv{load(prefix + name + ".base_kernel", {2, 2, H}),
                    linear(prefix + name + ".kernel_projection.weight", Coefficients, H)};
      };
      l.pre = load(prefix + "input_layernorm.weight", {H});
      l.post = load(prefix + "post_attention_layernorm.weight", {H});
      l.qnorm = load(prefix + "self_attn.q_norm.weight", {D});
      l.knorm = load(prefix + "self_attn.k_norm.weight", {D});
      l.attention = conv("attention_conv");
      l.mlp = conv("mlp_conv");
      l.q = linear(prefix + "self_attn.q_proj.weight", Q, H, true);
      l.k = linear(prefix + "self_attn.k_proj.weight", KV, H, true);
      l.v = linear(prefix + "self_attn.v_proj.weight", KV, H, true);
      l.o = linear(prefix + "self_attn.o_proj.weight", H, Q, true);
      l.gate = linear(prefix + "mlp.gate_proj.weight", I, H, true);
      l.up = linear(prefix + "mlp.up_proj.weight", I, H, true);
      l.down = linear(prefix + "mlp.down_proj.weight", H, I, true);
      l.keys = allocate(W * KV);
      l.values = allocate(W * KV);
    }
    ctx = allocate(CommitBatch * H);
    commit_k = allocate(CommitBatch * KV);
    commit_v = allocate(CommitBatch * KV);
    x = allocate(Block * H);
    normed = allocate(Block * H);
    convolved = allocate(Block * H);
    coeff = allocate(Block * Coefficients);
    q = allocate(Block * Q);
    key = allocate(Block * KV);
    value = allocate(Block * KV);
    block_key = allocate(Block * KV);
    attention = allocate(Block * Q);
    attention_partial = allocate<float>(Block * QHeads * (gqa ? GqaSplits : MaxSplits) * (D + 2));
    projection = allocate(Block * Q);
    gate = allocate(Block * I);
    up = allocate(Block * I);
    projected = allocate(Proposals * Rank);
    fp4_scratch = (nvfp4 || hybrid) ? allocate<float>(Block * I) : nullptr;
    lattice = allocate<float>(Proposals * 16 * 16);
    tokens = allocate<int>(Proposals);
    confidence = allocate<float>(Proposals);
  }

  template <typename T>
  void dump(const char* name, const T* values, std::size_t count, bool append = false) {
    std::vector<T> data(count);
    CUDA_CHECK(cudaMemcpyAsync(data.data(), values, count * sizeof(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::ofstream file(dump_dir / name, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!file.write(reinterpret_cast<const char*>(data.data()), count * sizeof(T)))
      throw std::runtime_error("Could not write DFlash2 fixture");
  }

  void linear(const Bf* input, int n, const Linear& w, Bf* out) {
    if (w.fp8.w && n == Block) {
      kernels::dense::dflash2_fp8(input, w.fp8, out, s);
    } else if (w.bf16) {
      gemm.run(input, n, w.bf16, w.rows, w.cols, out, s);
    } else {
      for (int first = 0; first < n; first += Block) {
        const int rows = std::min(Block, n - first);
        head::project_nvfp4(input + first * w.cols, rows, w.cols, w.packed, fp4_scratch, s);
        const int elements = rows * w.rows;
        convert<<<(elements + Threads - 1) / Threads, Threads, 0, s>>>(
            fp4_scratch, out + first * w.rows, elements);
      }
    }
  }

  void commit(const Bf* taps, int n, int start, bool decode) {
    if (start != length || n < 0)
      throw std::invalid_argument("DFlash2 commit must append to the committed context");
    if (!dump_dir.empty() && !dump_written && n) dump("taps.bf16", taps, std::size_t(n) * 4 * H, start != 0);
    profile.used = 0;
    if (decode && profile.enabled) {
      profile.begin(DraftProfile::Commit);
      ++profile.commits;
    }
    const bool decode_fp8 = padded_taps && decode && n > 0 && n <= Block;
    if (decode_fp8 && n < Block) {
      CUDA_CHECK(cudaMemcpyAsync(padded_taps, taps, n * 4 * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
      CUDA_CHECK(cudaMemsetAsync(padded_taps + n * 4 * H, 0, (Block - n) * 4 * H * sizeof(Bf), s));
    }
    // Any earlier rows in an oversized commit are outside the resulting ring.
    for (int first = std::max(0, n - W); first < n; first += CommitBatch) {
      const int count = std::min(CommitBatch, n - first);
      const int pos = start + first;
      const int projected_rows = decode_fp8 ? Block : count;
      if (decode_fp8) {
        kernels::dense::dflash2_fp8(n == Block ? taps : padded_taps, fc.fp8, ctx, s);
      } else {
        gemm.run(taps + std::size_t(first) * 4 * H, count, fc.bf16, H, 4 * H, ctx, s);
      }
      normalize<<<projected_rows, Threads, 0, s>>>(ctx, hidden_norm, H, 1, -1, ctx, 0);
      for (const auto& l : layers) {
        linear(ctx, projected_rows, l.k, commit_k);
        linear(ctx, projected_rows, l.v, commit_v);
        normalize<<<count * KVHeads, Threads, 0, s>>>(
            commit_k, l.knorm, D, KVHeads, pos, l.keys, W);
        const int slot = pos % W;
        const int contiguous = std::min(count, W - slot);
        CUDA_CHECK(cudaMemcpyAsync(l.values + slot * KV, commit_v, contiguous * KV * sizeof(Bf),
                                   cudaMemcpyDeviceToDevice, s));
        if (contiguous < count)
          CUDA_CHECK(cudaMemcpyAsync(l.values, commit_v + contiguous * KV,
                                     (count - contiguous) * KV * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
      }
    }
    CUDA_CHECK(cudaGetLastError());
    length += n;
    if (decode) profile.end();
  }

  void bind_embeddings(const Bf* weight, const float* norm) {
    if (embed || !weight || !norm)
      throw std::invalid_argument("DFlash2 requires target embeddings");
    embed = weight;
    embed_norm = norm;
    embeddings = allocate(Block * H);
    ids = allocate<int64_t>(Block);
    embed_output = allocate<float>(Block * H);
    anchor_dev = allocate<int>(1);
  }

  void prepare(const Bf* weight, const Conv& conv) {
    normalize<<<Block, Threads, 0, s>>>(x, weight, H, 1, -1, normed, 0);
    linear(normed, Block, conv.projection, coeff);
    convolution<<<dim3((H + Threads - 1) / Threads, Block), Threads, 0, s>>>(
        normed, coeff, conv.base, 0, convolved, false);
  }

  void residual(const Conv& conv) {
    convolution<<<dim3((H + Threads - 1) / Threads, Block), Threads, 0, s>>>(
        projection, coeff, conv.base, 1, x, true);
  }

  const Bf* forward(const Bf* embeds, const int* position = nullptr) {
    const int splits = attention_splits();
    CUDA_CHECK(cudaMemcpyAsync(x, embeds, Block * H * sizeof(Bf), cudaMemcpyDeviceToDevice, s));
    for (const auto& l : layers) {
      profile.begin(DraftProfile::Conv);
      prepare(l.pre, l.attention);
      profile.end();
      profile.begin(DraftProfile::Linear);
      linear(convolved, Block, l.q, projection);
      normalize<<<Block * QHeads, Threads, 0, s>>>(projection, l.qnorm, D, QHeads, length, q, 0, position);
      linear(convolved, Block, l.k, key);
      normalize<<<Block * KVHeads, Threads, 0, s>>>(key, l.knorm, D, KVHeads, length, block_key, 0, position);
      linear(convolved, Block, l.v, value);
      profile.end();
      profile.begin(DraftProfile::Attention);
      if (!splits) {
        attend<<<Block * QHeads, Threads, 0, s>>>(
            q, l.keys, l.values, block_key, value, length, attention, position);
      } else if (gqa) {
        const int tiles = (std::min(length, W - 1) + Block + GqaTile - 1) / GqaTile;
        const int grid_tiles = position ? GqaSplits : tiles;
        gqa_attention<<<dim3(KVHeads, grid_tiles), Threads, 0, s>>>(
            q, l.keys, l.values, block_key, value, length, attention_partial, position);
        merge_attention<GqaSplits><<<Block * QHeads, Threads, 0, s>>>(
            attention_partial, tiles, attention, position);
      } else {
        attend_split<<<dim3(Block * QHeads, splits), Threads, 0, s>>>(
            q, l.keys, l.values, block_key, value, length, attention_partial, position);
        merge_attention<MaxSplits><<<Block * QHeads, Threads, 0, s>>>(
            attention_partial, splits, attention, nullptr);
      }
      profile.end();
      profile.begin(DraftProfile::Linear);
      linear(attention, Block, l.o, projection);
      profile.end();
      profile.begin(DraftProfile::Conv);
      residual(l.attention);
      prepare(l.post, l.mlp);
      profile.end();
      profile.begin(DraftProfile::Linear);
      linear(convolved, Block, l.gate, gate);
      linear(convolved, Block, l.up, up);
      swiglu<<<Block * I / Threads, Threads, 0, s>>>(gate, up, gate);
      linear(gate, Block, l.down, projection);
      profile.end();
      profile.begin(DraftProfile::Conv);
      residual(l.mlp);
      profile.end();
    }
    profile.begin(DraftProfile::Embedding);
    normalize<<<Block, Threads, 0, s>>>(x, final_norm, H, 1, -1, normed, 0);
    CUDA_CHECK(cudaGetLastError());
    profile.end();
    return normed + H;
  }
};

namespace {
constexpr int HeadCandidates = Proposals * 16;

__global__ void pack_head_candidates(const float* values, const int* ids, int offset, head::Candidate* out) {
  const int i = threadIdx.x;
  out[i] = {values[i], ids[i] + offset};
}

__global__ void merge_head_candidates(const head::Candidate* gathered, int* ids, float* values) {
  const int row = threadIdx.x;
  const auto* first = gathered + row * 16;
  const auto* second = first + HeadCandidates;
  int a = 0, b = 0;
  for (int i = 0; i < 16; ++i) {
    const auto x = first[a], y = second[b];
    const bool pick_first = x.val > y.val || (x.val == y.val && x.global_id < y.global_id);
    const auto candidate = pick_first ? first[a++] : second[b++];
    ids[row * 16 + i] = candidate.global_id;
    values[row * 16 + i] = candidate.val;
  }
}
}  // namespace

struct DFlash2Head::Impl {
  TpRuntime& tp;
  head::Nvfp4Matrix matrix;
  cudaStream_t stream;
  DeviceBuffer hidden{Proposals * H * sizeof(Bf)};
  DeviceBuffer logits{std::size_t(Proposals) * matrix.rows * sizeof(float)};
  DeviceBuffer pool_values{Proposals * kernels::draft::CANDIDATES * sizeof(float)};
  DeviceBuffer pool_ids{Proposals * kernels::draft::CANDIDATES * sizeof(int)};
  DeviceBuffer top_values{HeadCandidates * sizeof(float)}, top_ids{HeadCandidates * sizeof(int)};
  DeviceBuffer local{HeadCandidates * sizeof(head::Candidate)}, gathered{2 * local.bytes};
  DeviceBuffer workspace{kernels::draft::topk_workspace_bytes(Proposals, matrix.rows)};
};

DFlash2Head::DFlash2Head(TpRuntime& tp, head::Nvfp4Matrix matrix, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(tp, matrix, stream)) {
  if (!matrix.codes || !matrix.scales)
    throw std::invalid_argument("DFlash2 sharded head requires --head-nvfp4");
  CUDA_CHECK(cudaMemsetAsync(impl_->workspace.p, 0, impl_->workspace.bytes, stream));
}
DFlash2Head::~DFlash2Head() = default;

void DFlash2Head::run(const Bf* hidden) {
  auto& d = *impl_;
  auto* input = const_cast<Bf*>(hidden);
  if (!input) input = static_cast<Bf*>(d.hidden.p);
  d.tp.exchange_draft_head(input, d.hidden.bytes, nullptr, d.stream);
  auto* logits = static_cast<float*>(d.logits.p);
  auto* pool_values = static_cast<float*>(d.pool_values.p);
  auto* pool_ids = static_cast<int*>(d.pool_ids.p);
  auto* values = static_cast<float*>(d.top_values.p);
  auto* ids = static_cast<int*>(d.top_ids.p);
  head::project_nvfp4(input, Proposals, H, d.matrix, logits, d.stream);
  kernels::draft::topk_candidates(logits, Proposals, d.matrix.rows, nullptr,
                                   pool_values, pool_ids, d.workspace.p, d.stream);
  best16<<<Proposals, Threads, 0, d.stream>>>(pool_values, pool_ids, values, ids);
  pack_head_candidates<<<1, HeadCandidates, 0, d.stream>>>(
      values, ids, d.matrix.v_begin, static_cast<head::Candidate*>(d.local.p));
  d.tp.exchange_draft_head(d.local.p, d.local.bytes, d.gathered.p, d.stream);
  if (hidden)
    merge_head_candidates<<<1, Proposals, 0, d.stream>>>(
        static_cast<const head::Candidate*>(d.gathered.p), ids, values);
  CUDA_CHECK(cudaGetLastError());
}

const int* DFlash2Head::ids() const { return static_cast<const int*>(impl_->top_ids.p); }
const float* DFlash2Head::values() const { return static_cast<const float*>(impl_->top_values.p); }

DFlash2::DFlash2(const std::filesystem::path& path, cudaStream_t stream, bool nvfp4)
    : impl_(std::make_unique<Impl>(path, stream, nvfp4)) {}
DFlash2::~DFlash2() = default;

void DFlash2::bind_head(head::Nvfp4Matrix matrix, const Bf* embed, const float* embed_norm) {
  auto& d = *impl_;
  if (matrix.v_begin || matrix.rows < kernels::draft::CANDIDATES ||
      matrix.rows > Vocabulary || !matrix.codes || !matrix.scales)
    throw std::invalid_argument("DFlash2 requires the complete NVFP4 target head");
  d.target_head = matrix;
  d.bind_embeddings(embed, embed_norm);
  d.unary = d.allocate<float>(Proposals * matrix.rows);
  d.pool_values = d.allocate<float>(Proposals * kernels::draft::CANDIDATES);
  d.pool_ids = d.allocate<int>(Proposals * kernels::draft::CANDIDATES);
  d.top_values = d.allocate<float>(Proposals * 16);
  d.top_ids = d.allocate<int>(Proposals * 16);
  const auto bytes = kernels::draft::topk_workspace_bytes(Proposals, matrix.rows);
  d.top_workspace = d.allocate<std::byte>(bytes);
  CUDA_CHECK(cudaMemsetAsync(d.top_workspace, 0, bytes, d.s));
}

void DFlash2::bind_sharded_head(DFlash2Head& head, const Bf* embed, const float* embed_norm) {
  auto& d = *impl_;
  if (!d.dump_dir.empty())
    throw std::invalid_argument("DFlash2 oracle dump requires the full head");
  d.sharded_head = &head;
  d.bind_embeddings(embed, embed_norm);
}

void DFlash2::warmup() {
  auto& d = *impl_;
  if (!d.device_position) return;  // Proposal dumps use the uncaptured path.

  DeviceBuffer taps(std::size_t(CommitBatch) * 4 * H * sizeof(Bf));
  CUDA_CHECK(cudaMemsetAsync(taps.p, 0, taps.bytes, d.s));
  d.commit(static_cast<const Bf*>(taps.p), 1, 0, false);
  propose(0, true);  // Initialize the kernels before capturing.
  d.profile.used = 0;
  propose(0, true);

  // Capture one graph for each split count. GQA has a single split graph.
  for (int length = AttentionTile - Block + 1; length < W; length += AttentionTile) {
    d.commit(static_cast<const Bf*>(taps.p), length - d.length, d.length, false);
    propose(0, true);
    if (d.gqa) break;
  }
  if (d.sharded_head) propose_sharded_head();
  CUDA_CHECK(cudaStreamSynchronize(d.s));
  reset();
}

const int* DFlash2::propose(int anchor) {
  return propose(anchor, false);
}

const int* DFlash2::propose(int anchor, bool warming_up) {
  auto& d = *impl_;
  if (!d.embed || anchor < 0 || anchor >= (d.sharded_head ? Vocabulary : d.target_head.rows))
    throw std::invalid_argument("Invalid DFlash2 anchor or missing target head");
  // Reserve a consistent commit span for every graph, including the initial proposal.
  if (d.profile.enabled && d.profile.used == 0) {
    d.profile.begin(DraftProfile::Commit);
    d.profile.end();
  }
  input_ids<<<1, Block, 0, d.s>>>(d.ids, d.anchor_dev, anchor, d.device_position, d.length);
  auto launch = [&](const int* position) {
    d.profile.begin(DraftProfile::Embedding);
    kernels::dense::embed(d.ids, Block, d.embed, d.embed_norm, d.embed_output, false, d.s);
    convert<<<Block * H / Threads, Threads, 0, d.s>>>(d.embed_output, d.embeddings, Block * H);
    d.profile.end();
    const auto* hidden = d.forward(d.embeddings, position);
    if (d.sharded_head) return;
    d.profile.begin(DraftProfile::Head);
    head::project_nvfp4(hidden, Proposals, H, d.target_head, d.unary, d.s);
    kernels::draft::topk_candidates(d.unary, Proposals, d.target_head.rows, nullptr,
                                    d.pool_values, d.pool_ids, d.top_workspace, d.s);
    best16<<<Proposals, Threads, 0, d.s>>>(d.pool_values, d.pool_ids, d.top_values, d.top_ids);
    d.profile.end();
    d.profile.begin(DraftProfile::Selector);
    select(d.top_ids, d.top_values, d.anchor_dev);
    d.profile.end();
  };
  if (d.device_position && d.warmed) {
    const int splits = d.attention_splits();
    const int graph_index = d.gqa && splits ? 1 : splits;
    auto& graph = d.graphs[graph_index];
    if (!graph) {
      d.profile.capturing = true;
      CUDA_CHECK(cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal));
      try {
        launch(d.device_position);
      } catch (...) {
        cudaGraph_t failed = nullptr;
        cudaStreamEndCapture(d.s, &failed);
        if (failed) cudaGraphDestroy(failed);
        throw;
      }
      d.profile.capturing = false;
      d.graph_profile_spans[graph_index] = d.profile.used;
      cudaGraph_t captured = nullptr;
      CUDA_CHECK(cudaStreamEndCapture(d.s, &captured));
      const auto status = cudaGraphInstantiate(&graph, captured, 0);
      cudaGraphDestroy(captured);
      CUDA_CHECK(status);
    }
    d.profile.used = d.graph_profile_spans[graph_index];
    CUDA_CHECK(cudaGraphLaunch(graph, d.s));
  } else {
    launch(nullptr);
    d.warmed = true;
  }
  const auto* hidden = d.normed + H;
  if (d.sharded_head && !warming_up) propose_sharded_head();
  if (!warming_up) d.profile.finish(d.length);
  const auto* tokens = d.tokens;
  if (!d.dump_dir.empty() && !d.dump_written && d.proposal_round++ == d.dump_round) {
    const auto elements = std::size_t(d.target_head.rows) * H;
    d.dump("head_codes.u8", d.target_head.codes, elements / 2);
    d.dump("head_scales.u8", d.target_head.scales, elements / 16);
    const float scale = d.target_head.outer_scale;
    std::ofstream scale_file(d.dump_dir / "head_scale.f32", std::ios::binary);
    if (!scale_file.write(reinterpret_cast<const char*>(&scale), sizeof(scale)))
      throw std::runtime_error("Could not write DFlash2 head scale");
    d.dump("embeddings.bf16", d.embeddings, Block * H);
    d.dump("hidden.bf16", hidden, Proposals * H);
    d.dump("unary.f32", d.unary, Proposals * d.target_head.rows);
    d.dump("top.i32", d.top_ids, Proposals * 16);
    d.dump("anchor.i32", d.anchor_dev, 1);
    d.dump("tokens.i32", tokens, Proposals);
    const int rows = std::min(d.length, W);
    const int first_slot = (d.length - rows) % W;
    const int contiguous = std::min(rows, W - first_slot);
    if (rows) {
      for (std::size_t i = 0; i < d.layers.size(); ++i) {
        const auto& layer = d.layers[i];
        for (const auto& [name, data] : {std::pair{"keys", layer.keys}, {"values", layer.values}}) {
          const auto file = std::string(name) + "-" + std::to_string(i) + ".bf16";
          d.dump(file.c_str(), data + first_slot * KV, contiguous * KV);
          if (contiguous < rows)
            d.dump(file.c_str(), data, (rows - contiguous) * KV, true);
        }
      }
    }
    d.dump_written = true;
  }
  return tokens;
}

void DFlash2::propose_sharded_head() {
  auto& d = *impl_;
  d.profile.begin(DraftProfile::Head);
  d.sharded_head->run(d.normed + H);
  d.profile.end();
  d.profile.begin(DraftProfile::Selector);
  select(d.sharded_head->ids(), d.sharded_head->values(), d.anchor_dev);
  d.profile.end();
}

const float* DFlash2::confidence() const { return impl_->confidence; }

bool dflash2_full_head() {
  const char* value = std::getenv("INKLING_DFLASH2_FULL_HEAD");
  return value && std::string_view(value) == "1";
}

void DFlash2::reset() {
  impl_->length = 0;
  impl_->proposal_round = 0;
  impl_->profile.reset();
}

void DFlash2::commit(const Bf* taps, int n, int start, bool decode) {
  impl_->commit(taps, n, start, decode);
}

const Bf* DFlash2::forward(const Bf* embeddings) { return impl_->forward(embeddings); }

const int* DFlash2::select(const int* candidates, const float* unary, const int* anchor) {
  auto& d = *impl_;
  d.linear(d.normed + H, Proposals, d.selector, d.projected);
  kernels::draft::dflash2_greedy_selector(candidates, unary, d.projected, d.pred, d.succ,
                                          anchor, d.lattice, d.tokens, d.s, d.confidence);
  if (d.cumulative) cumulative_confidence<<<1, 1, 0, d.s>>>(d.confidence);
  return d.tokens;
}

}  // namespace ingine::model
