#pragma once

#include "kernels/dense/dense.h"
#include "kernels/head/nvfp4_head.h"

namespace ingine::kernels::head {

struct CertifiedHead {
  Nvfp4Matrix proxy;
  const float* bounds = nullptr;  // [rows], upper bounds on residual norm plus accumulation allowance
};

// Calibrate from the original BF16 bits (also the input to PICO12 packing).
void cert_head_bounds(const __nv_bfloat16* exact, Nvfp4Matrix proxy, float* bounds, cudaStream_t stream);

std::size_t cert_head_workspace(int rows, int max_width = 8);

// Writes only token IDs. The caller may reuse any [M, rows] float scratch for the raw proxy scores.
// Optional counters [M] report the number of rows that passed the bound test before tile grouping.
// Calls on different streams require separate workspaces and scratch.
void cert_head(const __nv_bfloat16* x, int M, dense::Matrix exact, CertifiedHead cert, float* dots,
               void* workspace, int32_t* ids, cudaStream_t stream, unsigned* counts = nullptr,
               unsigned suspect_limit = 8192);

}  // namespace ingine::kernels::head
