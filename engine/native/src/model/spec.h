// DSpark speculative decoding loop: C++ port of engine.Engine._generate_spec / _accept / _rollback.
//
// Per round: the drafter proposes 7 tokens after the anchor (the last sampled, not yet processed token at
// position P); only the leading drafts whose confidence (max softmax probability) exceeds
// options.spec_threshold are verified, in the smallest configured width holding the anchor and those drafts. The
// verify call runs the target's decode step of that width (history + taps); acceptance is the longest
// prefix equal to the target argmax (greedy) or speculative rejection sampling against the drafts' own
// distributions (sampled, exact for the target's distribution); rejected rows are rolled back (conv states
// from the call's history, ring slots from the backup), the accepted rows' taps become draft context, and
// the accepted drafts plus the correction/bonus token are emitted.
//
// TP=1 uses device switches for verification and acceptance. TP=2 waits for a rank-0 proposal broadcast,
// then both ranks launch the matching fixed-width graph and apply the same acceptance and local rollback.
// Rank 0 commits the accepted taps and drafts the next block with Drafter::advance.
#pragma once

#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/model_kernels.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ingine::model {

class DFlash2;
class DFlash2Head;

// Read the greedy width policy once, when constructing the speculative loop.
DpWidthPolicy dflash2_dp_policy(const RuntimeOptions& options, bool cumulative);

class SpecLoop {
 public:
  // The runtime and drafter must outlive the loop and share a CUDA stream. Pass no drafter on the TP worker.
  SpecLoop(Runtime& rt, Drafter* drafter, DFlash2* dflash2 = nullptr,
           DFlash2Head* dflash2_head = nullptr);
  ~SpecLoop();
  SpecLoop(const SpecLoop&) = delete;
  SpecLoop& operator=(const SpecLoop&) = delete;

  // Generates after a prefill: `token` is the first sampled token (not yet processed), at position
  // P = tokens_in_state.size(); the drafter must hold every position before P (drafter.length() == P).
  // Emits tokens through on_tokens (several per round; stops after a token in stop_ids, after
  // max_new_tokens tokens, or when on_tokens returns false). Appends every processed token to
  // tokens_in_state and sets stats.generated_tokens, spec_rounds and spec_accepted. Sampled acceptance draws
  // from `rng` (and seeds the drafter's Gumbel noise from it). The caller reserves BLOCK positions of
  // max_ctx for the verify calls (engine.generate: prompt + max_new_tokens + 8 <= max_ctx).
  void generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kernels::head::SamplerRng& rng,
                std::span<const std::int32_t> stop_ids,
                const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                std::vector<std::int32_t>& tokens_in_state);

 private:
  // The accept graph (width_select -> switch { accept + rollback + publish }) for this sampling setup.
  void prepare_accept(const Sampling& sampling);
  template <bool Audit>
  void generate_host_verified(std::int32_t token, int max_new_tokens, const Sampling& sampling, kernels::head::SamplerRng& rng,
                   std::span<const std::int32_t> stop_ids,
                   const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                   GenerateStats& stats, std::vector<std::int32_t>& tokens_in_state);

  void generate_dflash2_device(std::int32_t token, int max_new_tokens, const Sampling& sampling,
                   kernels::head::SamplerRng& rng, std::span<const std::int32_t> stop_ids,
                   const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                   GenerateStats& stats, std::vector<std::int32_t>& tokens_in_state);
  void prepare_dflash2_exact();
  void accept_dflash2_exact(const float* logits, int width, const std::int32_t* drafts, const Sampling& sampling,
                           kernels::head::SamplerRng& rng, kernels::draft::AcceptResult* result);

  Runtime& rt_;
  Drafter* drafter_;
  DFlash2* dflash2_;
  DFlash2Head* dflash2_head_;
  DpWidthPolicy dflash2_dp_{};
  bool host_verify_, dflash2_device_;
  DeviceBuffer proposal_, host_accept_;
  DeviceBuffer target_tokens_;
  DeviceBuffer dflash2_accept_;  // lazily allocated target probabilities, residual, uniforms, and reduction workspace
  MappedHostBuffer audit_record_;  // allocated only on the auditing DFlash2 rank
  PinnedHostBuffer proposal_host_{sizeof(VerifyProposal)};
  Drafter::Draft draft_{};             // the drafter's fixed output buffers
  DeviceBuffer width_;                 // int32: the verify width chosen on the device
  MappedHostBuffer record_;            // VerifyRecord of the last round
  MappedHostBuffer forced_window_;     // forced replay: int32 [8], the forced tokens after the round's anchor
  GraphExec verify_, certified_verify_, accept_;
  struct AcceptKey {
    bool greedy, forced, certified;
    float temperature, top_p;
    int top_k;
    bool operator==(const AcceptKey&) const = default;
  };
  AcceptKey accept_key_{};
  cudaEvent_t done_ = nullptr;
};

}  // namespace ingine::model
