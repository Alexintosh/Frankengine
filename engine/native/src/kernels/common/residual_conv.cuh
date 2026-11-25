#pragma once

#include <cuda_runtime.h>

namespace ingine::kernels {

// These are the arithmetic contracts of the original fused attention and MLP writes.
template <bool Mlp>
__device__ __forceinline__ float residual_conv(const float4 w, const float* x) {
  constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
  if constexpr (Mlp) {
    float s = __fmul_rn(w.x, x[0]);
    s = __fadd_rn(s, __fmul_rn(w.y, x[1]));
    s = __fadd_rn(s, __fmul_rn(w.z, x[2]));
    s = __fadd_rn(s, __fmul_rn(w.w, x[3]));
    return __fmul_rn(__fadd_rn(s, x[3]), inv_sqrt5);
  } else {
    const float s = w.x * x[0] + w.y * x[1] + w.z * x[2] + w.w * x[3];
    return (s + x[3]) * inv_sqrt5;
  }
}

}  // namespace ingine::kernels
