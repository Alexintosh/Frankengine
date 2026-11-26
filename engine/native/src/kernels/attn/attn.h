// Attention core of one layer visit for M = 1..8 consecutive decode tokens (the PyTorch extension's kernel ported
// to a raw-pointer launcher; contract in native/KERNELS.md, area `attn`): q/rel post-processing, key/value convs +
// state update, KV cache write, split-K attention with relative bias, sink and output gate, in one kernel.
// Always launched with programmatic dependent launch (as the Python binding does); graph capturable.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::attn {

struct DecodeArgs {
  const float* proj = nullptr;
  const __nv_bfloat16* proj16 = nullptr;  // fp32 [M, R]: R = (full ? 8664 : 8600) * hkv / 8, SEG order
  int M = 1;                    // 1..8
  const float* q_scale = nullptr;   // fp32 [3*hkv, 128]
  const float* r_scale = nullptr;   // fp32 [3*hkv, 16]
  const float* rel_proj = nullptr;  // fp32 [16, 1024]
  const float* k_conv = nullptr;    // sliding: fp32 [hkv*128, 4] (conv1d order); full: k_static [4, hkv*128] by lag
  const float* v_conv = nullptr;
  float* conv_k_state = nullptr;    // fp32 [3, hkv*128]
  float* conv_v_state = nullptr;
  bool conv_write = true;
  void* k_cache = nullptr;          // [rows, hkv, 128] BF16/FP8, or [rows, hkv, 64] packed Q4
  void* v_cache = nullptr;
  bool cache_fp8 = false;
  bool cache_q4 = false;
  void* k_scale = nullptr;          // Q4: FP16 [rows, hkv, 2]
  void* v_scale = nullptr;
  std::int64_t scale_stride = 0;    // per slot, in FP16 elements
  std::int64_t cache_rows = 0;      // 1024 (sliding) or max_ctx (full), at most 2^30
  std::int64_t* kpos = nullptr;     // sliding: int64 [1024] slot positions (-1 = empty); full: unused
  const std::int64_t* pos = nullptr;  // int64 device, pos[0] = position of the first token
  int window = 0;                   // 1024 (sliding ring) or 0 (full layer, linear cache)
  bool log_scaling = false;         // full layers: q, rel *= tau(p)
  void* workspace = nullptr;        // workspace_bytes(M) bytes, zeroed once at allocation
  std::size_t workspace_size = 0;
  __nv_bfloat16* out = nullptr;     // bf16 [M, 3 * hkv * 128]
  int hkv = 8;                      // local KV heads (8 or 4)
  int batch = 1;
  const std::int32_t* slots = nullptr;
  std::int64_t cache_stride = 0, kpos_stride = 0, conv_stride = 0;
  const std::int32_t* rows = nullptr;
};

// Bytes of workspace attn_decode needs for up to max_m tokens: a counter (zero the workspace once; the
// kernel resets the counter itself) and the split partials. One workspace serves every layer of a
// stream-ordered decode step.
std::size_t workspace_bytes(int max_m, int hkv = 8);
std::size_t workspace_bytes_batched(int max_m, int max_batch, int hkv = 8);

// How many clusters of the selected size fit on the GPU at once for the full-head kernel.
int max_active_clusters(int M, int window);

// Per-sequence state for one layer visit. The table lives on the device.
struct BatchAttnSlot {
  void* k_cache;
  void* v_cache;
  float* conv_k_state;
  float* conv_v_state;
  std::int64_t* kpos;
  std::int64_t position;
  void* k_scale = nullptr;
  void* v_scale = nullptr;
};

void attn_decode_batch(const DecodeArgs& common, const BatchAttnSlot* slots, int batch, cudaStream_t stream);

void attn_decode(const DecodeArgs& a, cudaStream_t stream);

}  // namespace ingine::kernels::attn
