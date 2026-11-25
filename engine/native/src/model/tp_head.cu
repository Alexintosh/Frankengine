#include "model/tp_runtime.h"

#include "core/device.h"

namespace ingine::model {
namespace {
using kernels::head::Candidate;

__global__ void store_candidate(const float* logits, const int32_t* index, int begin, int stride, Candidate* local) {
  const int row = threadIdx.x;
  local[row] = {logits[std::size_t(row) * stride + index[row]], begin + index[row]};
}

__global__ void merge_candidates(const Candidate* ranks, int32_t* token, int m) {
  const int row = threadIdx.x;
  const Candidate a = ranks[row], b = ranks[m + row];
  token[row] = b.val > a.val || (b.val == a.val && b.global_id < a.global_id) ? b.global_id : a.global_id;
}

__global__ void narrow_hidden(const float* in, __nv_bfloat16* out, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}
}  // namespace

void tp_store_candidate(const float* logits, const int32_t* index, int begin, int m, int stride, Candidate* local, cudaStream_t s) {
  store_candidate<<<1, m, 0, s>>>(logits, index, begin, stride, local);
  CUDA_CHECK(cudaGetLastError());
}

void tp_merge_candidates(const Candidate* ranks, int32_t* token, int m, cudaStream_t s) {
  merge_candidates<<<1, m, 0, s>>>(ranks, token, m);
  CUDA_CHECK(cudaGetLastError());
}

void tp_narrow_hidden(const float* in, __nv_bfloat16* out, std::size_t n, cudaStream_t s) {
  narrow_hidden<<<(n + 255) / 256, 256, 0, s>>>(in, out, n);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::model
