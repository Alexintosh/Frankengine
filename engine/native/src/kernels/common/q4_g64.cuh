// Q4_G64 cache storage: 128 channels, two independent 64-channel groups.
// Each lane owns four adjacent BF16-rounded channels when packing.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::q4_g64 {

inline constexpr int head_bytes = 64;
inline constexpr int groups = 2;

__device__ __forceinline__ float unpack(std::uint32_t bits, int dim, float scale) {
  return float(int((bits >> (4 * dim)) & 15u) - 8) * scale;
}

__device__ __forceinline__ std::uint16_t pack4(const float (&x)[4], __half& scale) {
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  scale = __float2half_rn(__fdiv_rn(amax, 7.f));
  const float s = __half2float(scale);
  std::uint16_t bits = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = s > 0.f ? min(7, max(-7, __float2int_rn(__fdiv_rn(x[i], s)))) : 0;
    bits |= std::uint16_t((q + 8) << (4 * i));
  }
  return bits;
}

template <bool Values>
__device__ __forceinline__ std::uint32_t expand_pair(std::uint32_t bits, int dim, float scale) {
  const float a = unpack(bits, dim, scale), b = unpack(bits, dim + 1, scale);
  if constexpr (Values) {
    const __half2_raw pair = __floats2half2_rn(a, b);
    return pair.x | (std::uint32_t(pair.y) << 16);
  } else {
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(a, b);
    return pair.x | (std::uint32_t(pair.y) << 16);
  }
}

template <bool Values>
__device__ __forceinline__ uint4 expand8(std::uint32_t bits, float scale) {
  return make_uint4(expand_pair<Values>(bits, 0, scale), expand_pair<Values>(bits, 2, scale),
                    expand_pair<Values>(bits, 4, scale), expand_pair<Values>(bits, 6, scale));
}

}  // namespace ingine::kernels::q4_g64
