#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::draft {

struct AcceptResult;

inline constexpr int DFLASH2_SUPPORT = 16;

struct DFlash2Support {
  std::int32_t* ids;
  float* probs;
  std::int32_t* picked;
};

// Seven rows of 16 candidates, with unary logits before softcap. projected is
// [7,256], and the codebooks are [vocab,256]. anchor is a device token ID.
// Writes [7,16,16] edge scores and the seven tokens of the greedy local walk.
void dflash2_greedy_selector(const std::int32_t* ids, const float* unary,
                            const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                            const __nv_bfloat16* successors, const std::int32_t* anchor,
                            float* lattice, std::int32_t* tokens, cudaStream_t stream,
                            float* confidence = nullptr);

// Greedy when temperature is null or nonpositive. A sampled walk writes its
// conditional distribution over the 16 candidates at each of the seven steps.
void dflash2_fused_selector(const std::int32_t* ids, const float* unary,
                           const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                           const __nv_bfloat16* successors, const std::int32_t* anchor,
                           std::int32_t* tokens, float* confidence, bool cumulative,
                           const float* temperature, const std::uint64_t* seed,
                           DFlash2Support support, cudaStream_t stream, float edge_scale = 1.0f);

// Same walk with a materialized edge lattice for the legacy and oracle paths.
void dflash2_lattice_selector(const std::int32_t* ids, const float* unary,
                             const __nv_bfloat16* projected, const __nv_bfloat16* predecessors,
                             const __nv_bfloat16* successors, const std::int32_t* anchor,
                             float* lattice, std::int32_t* tokens, float* confidence, bool cumulative,
                             const float* temperature, const std::uint64_t* seed,
                             DFlash2Support support, cudaStream_t stream, float edge_scale = 1.0f);

// A single mapped host record, published before the existing round-completion event.
struct DFlash2Audit {
  float conf[7];
  int in_support[7];
  float target_mass[7], overlap[7], picked_p[7], picked_q[7];
};
static_assert(sizeof(DFlash2Audit) < 1024);

// target is the greedy argmax, or null. p is the filtered target distribution for exact
// sampling, or null. When p is null neither q nor picked is read.
void dflash2_audit(const float* confidence, const std::int32_t* ids, int eligible, int vocab,
                   const std::int32_t* target, const float* p, const float* q,
                   const std::int32_t* picked, DFlash2Audit* out, cudaStream_t stream);

// p is [eligible+1, vocab], support_ids/probs are [7,16], and picked and uniforms
// are [7]. ws holds dflash2_accept_workspace_floats(vocab) floats. Writes the
// accepted prefix length and the distribution for its correction or bonus token.
std::size_t dflash2_accept_workspace_floats(int vocab);
void dflash2_exact_accept(const float* p, int vocab, const std::int32_t* drafts, int eligible,
                         const std::int32_t* support_ids, const float* support_probs,
                         const std::int32_t* picked, const float* uniforms, float* ws,
                         float* residual, AcceptResult* out, cudaStream_t stream);

}  // namespace ingine::kernels::draft
