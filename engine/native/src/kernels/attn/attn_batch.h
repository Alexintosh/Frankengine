#pragma once
#include "kernels/attn/attn.h"
#include <cstddef>

namespace ingine::kernels::attn {
std::size_t batched_workspace_bytes(int max_m, int max_batch, int hkv = 8);
void attn_decode_batched(const DecodeArgs& a, cudaStream_t stream);
}  // namespace ingine::kernels::attn
