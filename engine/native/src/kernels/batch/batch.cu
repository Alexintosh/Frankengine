#include "kernels/batch/batch.h"

#include "core/device.h"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::batch {
namespace {

__device__ __forceinline__ float bf2f(std::uint16_t v) { return __uint_as_float(std::uint32_t(v) << 16); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// one thread per (channel, sequence); the sequence's rows in order, as prefill::static_conv + conv_state_update
__global__ void conv_kernel(const std::uint16_t* __restrict__ x, int ld, const std::int32_t* __restrict__ rows, int C,
                            const float* __restrict__ w, float* __restrict__ state, long long slot_stride,
                            const std::int32_t* __restrict__ slots, int write_state, float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= C) return;
  float* const st = state + (long long)slots[b] * slot_stride;
  const float4 w4 = *reinterpret_cast<const float4*>(w + std::size_t(c) * 4);
  float e0 = st[c], e1 = st[C + c], e2 = st[2 * C + c];
  const float inv = 1.f / 2.2360679774997896f;
  for (int r0 = rows[b], r1 = rows[b + 1], q = r0; q < r1; ++q) {
    const std::size_t r = std::size_t(q);
    const float e3 = bf2f(x[r * ld + c]);
    float acc = __fmul_rn(w4.x, e0);
    acc = __fadd_rn(acc, __fmul_rn(w4.y, e1));
    acc = __fadd_rn(acc, __fmul_rn(w4.z, e2));
    acc = __fadd_rn(acc, __fmul_rn(w4.w, e3));
    out[r * C + c] = __fmul_rn(__fadd_rn(acc, e3), inv);
    e0 = e1, e1 = e2, e2 = e3;
  }
  if (write_state) st[c] = e0, st[C + c] = e1, st[2 * C + c] = e2;
}

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ bool better(float a, int av, float b, int bv) { return a > b || (a == b && av < bv); }

constexpr int SAMPLE_T = 1024;

template <bool DeviceSeed = false>
__global__ void __launch_bounds__(SAMPLE_T) sample_kernel(const float* __restrict__ logits, int ld, int V, int raw,
                                                          const float* __restrict__ temps, float temperature,
                                                          std::uint64_t seed, std::int32_t* __restrict__ out, const std::uint64_t* d_seed) {
  if constexpr (DeviceSeed) seed = *d_seed;
  const int row = blockIdx.x;
  const float* const l = logits + std::size_t(row) * ld;
  const std::uint64_t row_key = mix64(seed ^ (std::uint64_t(row) * 0xD1B54A32D192ED03ull));
  const float t = temps ? temps[row] : temperature;
  const bool greedy = !(t > 0.f);
  const float inv_t = greedy ? 1.f : 1.f / t;
  float best = -INFINITY;
  int best_v = 0x7fffffff;
  for (int v = threadIdx.x; v < V; v += SAMPLE_T) {
    float key = raw ? 30.f * tanhf(bf16r(l[v]) / 30.f) : l[v];  // model.logits
    if (!greedy) {
      // Gumbel-max: argmax_v s_v / T + G_v is a sample of softmax(s / T); u in (0, 1) from 24 random bits
      // 23-bit midpoint stays strictly in (0, 1) in FP32; 24-bit (16777215 + 0.5f) rounds to 1.0f -> +inf Gumbel.
    const float u = (float(mix64(row_key ^ std::uint64_t(v)) >> 41) + 0.5f) * (1.f / 8388608.f);
      key = key * inv_t - logf(-logf(u));
    }
    if (better(key, v, best, best_v)) best = key, best_v = v;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    if (better(ob, ov, best, best_v)) best = ob, best_v = ov;
  }
  __shared__ float s_best[SAMPLE_T / 32];
  __shared__ int s_v[SAMPLE_T / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) s_best[warp] = best, s_v[warp] = best_v;
  __syncthreads();
  if (warp == 0) {
    best = s_best[lane], best_v = s_v[lane];  // SAMPLE_T / 32 == 32 warps
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int ov = __shfl_xor_sync(0xffffffffu, best_v, o);
      if (better(ob, ov, best, best_v)) best = ob, best_v = ov;
    }
    if (lane == 0) out[row] = best_v;
  }
}

// one block per row: copies K and V segments from proj (uint4 for 16-byte aligned TP=1, uint2 for 8-byte aligned TP=2)
template <typename Vec>
__global__ void keep_kv_kernel(const Vec* __restrict__ proj, int ldv, int kv, int vv, int nvec, Vec* __restrict__ hk,
                               Vec* __restrict__ hv) {
  const int r = blockIdx.x, t = threadIdx.x;
  const Vec* row = proj + std::size_t(r) * ldv;
  if (t < nvec) hk[std::size_t(r) * nvec + t] = row[kv + t];
  else if (t < 2 * nvec) hv[std::size_t(r) * nvec + t - nvec] = row[vv + t - nvec];
}

// The four conv states of a slot in float4 units: per kind (k, v: 256 per row; attn, mlp: 576 per row) 48 visits x
// R rows. Item q -> kind, visit, row, column.
constexpr int KV4 = 1024 / 4, H4 = 2304 / 4;
struct ConvVec {
  int kind, v, row, c4, C4;
};
template <int Visits>
__device__ __forceinline__ ConvVec conv_vec(int q, int R, int kv4) {
  ConvVec a;
  const int kv = Visits * R * kv4, h = Visits * R * H4;
  if (q < kv) a.kind = 0, a.C4 = kv4;
  else if ((q -= kv) < kv) a.kind = 1, a.C4 = kv4;
  else if ((q -= kv) < h) a.kind = 2, a.C4 = H4;
  else q -= h, a.kind = 3, a.C4 = H4;
  a.c4 = q % a.C4;
  const int vr = q / a.C4;
  a.v = vr / R, a.row = vr % R;
  return a;
}
template <int Visits>
__host__ __device__ constexpr int conv_vecs(int R, int kv4 = KV4) { return Visits * R * (2 * kv4 + 2 * H4); }
__device__ __forceinline__ float* state_of(const ConvStates& st, int kind, long long slot) {
  return kind == 0 ? st.k + slot * st.kv_stride
       : kind == 1 ? st.v + slot * st.kv_stride
       : kind == 2 ? st.attn + slot * st.h_stride
                   : st.mlp + slot * st.h_stride;
}

// one thread per (sequence, conv, visit, 4 columns): reads the old rows it keeps, then writes the new ones
template <int Visits>
__global__ void conv_commit_kernel(ConvStates st, const std::int32_t* __restrict__ slots, ConvHistory hist, int T,
                                   const std::int32_t* __restrict__ rows, const std::int32_t* __restrict__ commit, int kv4) {
  const int b = blockIdx.y, q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= conv_vecs<Visits>(1, kv4)) return;
  const ConvVec a = conv_vec<Visits>(q, 1, kv4);
  const int c = commit[b], r0 = rows[b], C = 4 * a.C4;
  float* const s = state_of(st, a.kind, slots[b]) + std::size_t(a.v) * 3 * C + 4 * a.c4;
  const __nv_bfloat16* h = a.kind == 0 ? hist.k : a.kind == 1 ? hist.v : a.kind == 2 ? hist.attn : hist.mlp;
  float4 row[3];
  for (int j = 0; j < 3; ++j) {  // row c + j of [old rows 0..2, the sequence's inputs]
    const int src = c + j;
    if (src < 3) {
      row[j] = *reinterpret_cast<const float4*>(s + src * C);
    } else {
      const uint2 u =
          *reinterpret_cast<const uint2*>(h + (std::size_t(a.v) * T + std::size_t(r0) + src - 3) * C + 4 * a.c4);
      row[j] = make_float4(bf2f(std::uint16_t(u.x)), bf2f(std::uint16_t(u.x >> 16)), bf2f(std::uint16_t(u.y)),
                           bf2f(std::uint16_t(u.y >> 16)));
    }
  }
  for (int j = 0; j < 3; ++j) *reinterpret_cast<float4*>(s + j * C) = row[j];
}

template <bool Restore>
__global__ void verify_rings_kernel(RingRows cache, RingRows backup, int rings, int kv_dim, int max_m,
                                    const int* slots, const std::int64_t* pos, const int* rows,
                                    const int* commit) {
  // Row 0 is always accepted. The old cache row under row 1 is already out of window
  // when verification accepts only the anchor and the next forward processes row 1.
  const int b = blockIdx.x / (max_m - 2);
  const int r = 2 + blockIdx.x % (max_m - 2);
  const int ring = blockIdx.y;
  const std::int64_t absolute = pos[b] + r;
  if (r >= rows[b + 1] - rows[b] || absolute < cache.rows_per_ring) return;
  if constexpr (Restore) {
    if (r <= commit[b]) return;
  }
  const std::size_t from = (std::size_t(slots[b]) * rings + ring) * cache.rows_per_ring +
                           absolute % cache.rows_per_ring;
  const std::size_t saved = std::size_t(ring) * backup.rows_per_ring + std::size_t(b) * (max_m - 2) + r - 2;
  const int vectors = kv_dim / 8;
  const int c = threadIdx.x;
  if (c < vectors) {
    auto* k = reinterpret_cast<uint4*>(cache.k) + from * vectors + c;
    auto* v = reinterpret_cast<uint4*>(cache.v) + from * vectors + c;
    auto* k_saved = reinterpret_cast<uint4*>(backup.k) + saved * vectors + c;
    auto* v_saved = reinterpret_cast<uint4*>(backup.v) + saved * vectors + c;
    if constexpr (Restore) {
      *k = *k_saved;
      *v = *v_saved;
    } else {
      *k_saved = *k;
      *v_saved = *v;
    }
  }
  if (c == 0) {
    if constexpr (Restore) cache.pos[from] = backup.pos[saved];
    else backup.pos[saved] = cache.pos[from];
  }
}

__global__ void gather_rows_kernel(const uint4* __restrict__ src, int row8, const std::int32_t* __restrict__ rows,
                                   uint4* __restrict__ dst) {
  const uint4* s = src + std::size_t(rows[blockIdx.x]) * row8;
  uint4* d = dst + std::size_t(blockIdx.x) * row8;
  for (int i = threadIdx.x; i < row8; i += blockDim.x) d[i] = s[i];
}

}  // namespace

void conv(const __nv_bfloat16* x, int ld, int n, const std::int32_t* rows, int C, const float* w, float* state,
          std::int64_t slot_stride, const std::int32_t* slots, bool write_state, float* out, cudaStream_t stream) {
  if (n <= 0) return;
  const dim3 grid((C + 127) / 128, n);
  conv_kernel<<<grid, 128, 0, stream>>>(reinterpret_cast<const std::uint16_t*>(x), ld, rows, C, w, state, slot_stride,
                                        slots, write_state ? 1 : 0, out);
  CUDA_CHECK(cudaGetLastError());
}

void conv_commit(const ConvStates& st, int n, const std::int32_t* slots, const ConvHistory& hist, int T,
                 const std::int32_t* rows, const std::int32_t* commit, cudaStream_t stream, int kv_dim, int visits) {
  if (n <= 0) return;
  const int kv4 = kv_dim / 4;
  auto launch = [&]<int Visits>() {
    const dim3 grid((conv_vecs<Visits>(1, kv4) + 255) / 256, n);
    conv_commit_kernel<Visits><<<grid, 256, 0, stream>>>(st, slots, hist, T, rows, commit, kv4);
  };
  if (visits == 48) launch.template operator()<48>();
  else if (visits == 54) launch.template operator()<54>();
  else throw std::invalid_argument("batch::conv_commit: unsupported visit count");
  CUDA_CHECK(cudaGetLastError());
}

void verify_rings(const RingRows& cache, const RingRows& backup, int rings, int kv_dim, int sequences, int max_m,
                  const std::int32_t* slots, const std::int64_t* pos, const std::int32_t* rows,
                  const std::int32_t* commit, cudaStream_t stream) {
  const dim3 grid(sequences * (max_m - 2), rings);
  const int threads = kv_dim / 8;
  if (commit)
    verify_rings_kernel<true><<<grid, threads, 0, stream>>>(cache, backup, rings, kv_dim, max_m, slots, pos, rows, commit);
  else
    verify_rings_kernel<false><<<grid, threads, 0, stream>>>(cache, backup, rings, kv_dim, max_m, slots, pos, rows, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void sample(const float* logits, int ld, int n, int V, bool raw, const float* temps, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream, const std::uint64_t* d_seed) {
  if (n <= 0) return;
  if (V <= 0 || V > ld) throw std::invalid_argument("batch::sample: V outside 1..ld");
  if (d_seed)
    sample_kernel<true><<<n, SAMPLE_T, 0, stream>>>(logits, ld, V, raw ? 1 : 0, temps, 0.f, seed, out, d_seed);
  else
    sample_kernel<false><<<n, SAMPLE_T, 0, stream>>>(logits, ld, V, raw ? 1 : 0, temps, 0.f, seed, out, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void sample(const float* logits, int ld, int n, int V, bool raw, float temperature, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (V <= 0 || V > ld) throw std::invalid_argument("batch::sample: V outside 1..ld");
  sample_kernel<false><<<n, SAMPLE_T, 0, stream>>>(logits, ld, V, raw ? 1 : 0, nullptr, temperature, seed, out, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void keep_kv(const __nv_bfloat16* proj, int ld, int rows, int k_off, int v_off, __nv_bfloat16* hist_k,
             __nv_bfloat16* hist_v, cudaStream_t stream, int kv_dim) {
  if (rows <= 0) return;
  if (ld % 8 == 0 && k_off % 8 == 0 && v_off % 8 == 0 && kv_dim % 8 == 0) {
    const int nvec = kv_dim / 8;
    keep_kv_kernel<uint4><<<rows, 2 * nvec, 0, stream>>>(reinterpret_cast<const uint4*>(proj), ld / 8, k_off / 8,
                                                         v_off / 8, nvec, reinterpret_cast<uint4*>(hist_k),
                                                         reinterpret_cast<uint4*>(hist_v));
  } else if (ld % 4 == 0 && k_off % 4 == 0 && v_off % 4 == 0 && kv_dim % 4 == 0) {
    const int nvec = kv_dim / 4;
    keep_kv_kernel<uint2><<<rows, 2 * nvec, 0, stream>>>(reinterpret_cast<const uint2*>(proj), ld / 4, k_off / 4,
                                                         v_off / 4, nvec, reinterpret_cast<uint2*>(hist_k),
                                                         reinterpret_cast<uint2*>(hist_v));
  } else {
    throw std::invalid_argument("batch::keep_kv: unaligned columns");
  }
  CUDA_CHECK(cudaGetLastError());
}

void gather_rows(const __nv_bfloat16* src, int row_elems, const std::int32_t* rows, int count, __nv_bfloat16* dst,
                 cudaStream_t stream) {
  if (count <= 0) return;
  if (row_elems % 8) throw std::invalid_argument("batch::gather_rows: rows of a multiple of 8 elements");
  gather_rows_kernel<<<count, 256, 0, stream>>>(reinterpret_cast<const uint4*>(src), row_elems / 8, rows,
                                                reinterpret_cast<uint4*>(dst));
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::batch
