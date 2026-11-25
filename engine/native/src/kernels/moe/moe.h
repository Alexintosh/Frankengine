// Routed-expert kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `moe`) and the expert decode to BF16. Launchers run on `stream`, never
// allocate or synchronise, read routing from device memory and are CUDA-graph capturable. Shapes are
// row-major (Python order); H = 2304, I = 1152, 128 experts, top-8. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::moe {

enum class Format : int { NVFP4 = 0, MXFP8 = 1 };

// moe_up: W13 GEMV of the selected experts + softcapped SwiGLU. x bf16 [M, H], idx int32 [M, 8],
// out bf16 [M, 8, I]. NVFP4: w13 u8 [128, 2I, H/2], w13_scale e4m3 [128, 2I, H/16], w13_factor fp32 [128];
// MXFP8: w13 e4m3 [128, 2I, H], w13_scale u8 E8M0 [128, 2I, H/32], factor unused (may be null).
// pdl: the kernel waits for its predecessor before reading anything.
void moe_up(const __nv_bfloat16* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, __nv_bfloat16* out, bool pdl, cudaStream_t stream,
            int intermediate = 1152);

// moe_down: W2 GEMV + routing-weighted sum + post-MLP conv with state update. act bf16 [M, 8, I],
// wts fp32 [M, 8], w2 [128, H, I(/2)] + scales [128, H, I/16 | I/32] (+ factor), conv_w fp32 [H, 4],
// conv_state fp32 [3, H], out fp32 [M, H], conv_hist fp32 [M, H] (BF16-rounded conv inputs) or null.
// pdl: valid only right after moe_up (idx is read before the PDL wait).
void moe_down(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream);

// Local W2 weighted sum before rank reduction (intermediate = 576 or 1152). Write FP32/BF16 [M, H] and/or diagnostic unrounded
// per-expert dots [M, 8, H]. The local weighted sum keeps router slot order and uses FP32 expert dots.
// An optional FP32 workspace [8, 8, H] enables the independent expert/tile kernel at K=576 and M>=2.
void moe_down_partial(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
                      const void* w2_scale, const float* w2_factor, Format fmt, int intermediate,
                      __nv_bfloat16* out_bf16, float* out_f32, float* expert_dots,
                      bool pdl, cudaStream_t stream, float* workspace = nullptr);

// dequant_experts: decode experts[0..n) (int32 device) of one weight ([128, rows, cols] in checkpoint
// encoding: NVFP4 codes u8 [128, rows, cols/2] + e4m3 scales [128, rows, cols/16] + factor fp32 [128];
// MXFP8 codes e4m3 [128, rows, cols] + E8M0 scales [128, rows, cols/32]) into out bf16 [n, rows, cols],
// exactly as model.decode_experts (FP32 product, one BF16 rounding).
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ingine::kernels::moe
