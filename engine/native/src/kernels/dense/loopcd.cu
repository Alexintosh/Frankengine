#include "kernels/dense/dense.h"

#include "core/device.h"
#include "ingine/generator.h"

#include <cmath>

namespace ingine::kernels::dense {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ float extrapolate(float main, float ref, float omega) {
  return fmaf(omega, main - ref, main);
}

__global__ void loopcd_hidden_kernel(const float* base, const float* write, const float* ref_stream, float* out,
                                     float* depth, const float* ref_depth, std::size_t elements, float omega) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  const float hidden = write ? base[i] + write[i] : base[i];
  out[i] = extrapolate(hidden, ref_stream[i], omega);
  depth[i] = extrapolate(depth[i], ref_depth[i], omega);
}

__global__ void set_loopcd_omega_kernel(float* dst, float omega) {
  *dst = omega;
}

template <bool PerRow>
__global__ void loopcd_hidden_device_kernel(const float* base, const float* write, const float* ref_stream,
                                            float* out, float* depth, const float* ref_depth,
                                            std::size_t width, const float* omegas) {
  const std::size_t col = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= width) return;
  const std::size_t row = PerRow ? blockIdx.y : 0;
  const std::size_t i = row * width + col;
  const float omega = omegas[row];
  const float value = write ? base[i] + write[i] : base[i];
  if (omega == 0.f) {
    out[i] = value;
    return;
  }
  out[i] = extrapolate(value, ref_stream[i], omega);
  depth[i] = extrapolate(depth[i], ref_depth[i], omega);
}

struct TopTwo {
  float first, second;

  __device__ void add(float value) {
    if (value > first) {
      second = first;
      first = value;
    } else {
      second = fmaxf(second, value);
    }
  }
};

__device__ TopTwo warp_top_two(TopTwo top) {
  for (int offset = 16; offset > 0; offset /= 2) {
    const float first = __shfl_xor_sync(0xffffffffu, top.first, offset);
    const float second = __shfl_xor_sync(0xffffffffu, top.second, offset);
    top.add(first);
    top.add(second);
  }
  return top;
}

template <bool Adaptive>
__global__ void loopcd_combine_kernel(float* main, const float* ref, int ld, int vocab, float omega) {
  main += std::size_t(blockIdx.x) * ld;
  ref += std::size_t(blockIdx.x) * ld;
  const int t = threadIdx.x;
  float weight = omega;
  if constexpr (Adaptive) {
    __shared__ TopTwo top_by_warp[kWarps];
    __shared__ float sum_by_warp[kWarps];
    __shared__ TopTwo top_two;
    __shared__ float weight_shared;
    const int lane = t % 32, warp = t / 32;
    TopTwo top{-INFINITY, -INFINITY};
    for (int i = t; i < vocab; i += kThreads) top.add(main[i]);
    top = warp_top_two(top);
    if (lane == 0) top_by_warp[warp] = top;
    __syncthreads();
    if (warp == 0) {
      top = lane < kWarps ? top_by_warp[lane] : TopTwo{-INFINITY, -INFINITY};
      top = warp_top_two(top);
      if (lane == 0) top_two = top;
    }
    __syncthreads();
    float sum = 0.f;
    for (int i = t; i < vocab; i += kThreads) sum += expf(main[i] - top_two.first);
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if (lane == 0) sum_by_warp[warp] = sum;
    __syncthreads();
    if (warp == 0) {
      sum = lane < kWarps ? sum_by_warp[lane] : 0.f;
      for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      if (lane == 0) {
        const float p1 = 1.f / sum;
        const float p2 = expf(top_two.second - top_two.first) / sum;
        weight_shared = omega * fmaxf(0.f, 1.f - (p1 - p2));
      }
    }
    __syncthreads();
    weight = weight_shared;
  }
  for (int i = t; i < vocab; i += kThreads) main[i] = extrapolate(main[i], ref[i], weight);
}

__global__ void loopcd_softcap_kernel(float* logits, int ld, int vocab) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= vocab) return;
  auto& logit = logits[std::size_t(blockIdx.y) * ld + v];
  logit = 30.f * tanhf(__bfloat162float(__float2bfloat16_rn(logit)) / 30.f);
}

}  // namespace

void loopcd_hidden(const float* base, const float* write, const float* ref_stream, float* out,
                   float* depth, const float* ref_depth, std::size_t elements, float omega, cudaStream_t stream) {
  loopcd_hidden_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      base, write, ref_stream, out, depth, ref_depth, elements, omega);
  CUDA_CHECK(cudaGetLastError());
}

void set_loopcd_omega(float* omega_dev, float omega, cudaStream_t stream) {
  set_loopcd_omega_kernel<<<1, 1, 0, stream>>>(omega_dev, omega);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_hidden_device(const float* base, const float* write, const float* ref_stream, float* out,
                          float* depth, const float* ref_depth, std::size_t elements,
                          const float* omega_dev, cudaStream_t stream) {
  loopcd_hidden_device_kernel<false><<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      base, write, ref_stream, out, depth, ref_depth, elements, omega_dev);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_hidden_rows(const float* base, const float* write, const float* ref_stream, float* out,
                        float* depth, const float* ref_depth, int rows, int hidden,
                        const float* row_omegas, cudaStream_t stream) {
  const dim3 grid((hidden + kThreads - 1) / kThreads, rows);
  loopcd_hidden_device_kernel<true><<<grid, kThreads, 0, stream>>>(
      base, write, ref_stream, out, depth, ref_depth, hidden, row_omegas);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_combine_logits(float* main, const float* ref, int rows, int ld, int vocab, model::LoopCdMode mode,
                           float omega, cudaStream_t stream) {
  if (mode == model::LoopCdMode::Adaptive)
    loopcd_combine_kernel<true><<<rows, kThreads, 0, stream>>>(main, ref, ld, vocab, omega);
  else
    loopcd_combine_kernel<false><<<rows, kThreads, 0, stream>>>(main, ref, ld, vocab, omega);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_softcap_logits(float* logits, int rows, int ld, int vocab, cudaStream_t stream) {
  const dim3 grid((vocab + kThreads - 1) / kThreads, rows);
  loopcd_softcap_kernel<<<grid, kThreads, 0, stream>>>(logits, ld, vocab);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense
