#pragma once

#include "ingine/generator.h"
#include "model/model_data.h"

#include <memory>

namespace ingine::model {

enum class VerifyStage : unsigned {
  Mix, MessageGates, Wproj, Attention, Wo, Router, MoeUp, MoeDown, Conv, Head, Count
};

enum class TpSchedule : unsigned { Decode = 1, Prefill = 2, Request = 3, Proposal = 4, DraftHead = 5 };

class TpRuntime {
 public:
  explicit TpRuntime(const RuntimeOptions& options);
  ~TpRuntime();

  const TpLayout layout;
  const TpWire wire;
  bool full_logits;
  const bool profile, gb10_spec, serving;

  bool bf16_wire(unsigned op) const {
    return wire == TpWire::Bf16 || (wire == TpWire::Bf16Moe && op % 2 != 0);
  }

  void reduce(float* fp32, __nv_bfloat16* bf16, std::size_t count, TpSchedule schedule,
              unsigned op, cudaStream_t stream);
  void allreduce_bf16(__nv_bfloat16* bf16, std::size_t count, TpSchedule schedule,
                      unsigned op, cudaStream_t stream);
  void allreduce_f32(float* fp32, std::size_t count, TpSchedule schedule,
                     unsigned op, cudaStream_t stream);
  void head(const Weights& weights, const __nv_bfloat16* x, float* logits, int32_t* token, int m,
            TpSchedule schedule, cudaStream_t stream);
  void broadcast_proposal(void* device, std::size_t bytes, cudaStream_t stream);
  void exchange_draft_head(void* device, std::size_t bytes, void* gathered, cudaStream_t stream);
  bool agree(std::uint64_t value, cudaStream_t stream);
  void verify_request(std::uint64_t fingerprint, cudaStream_t stream);
  void wait(cudaStream_t stream);
  void check() const;
  void mark(unsigned boundary, cudaStream_t stream);
  DecodeTiming timing() const;
  void mark_stage(VerifyStage stage, int occurrence, bool end, cudaStream_t stream);
  void print_verify_profile(int width) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void tp_store_candidate(const float* logits, const int32_t* index, int begin, int m, int stride,
                        kernels::head::Candidate* local, cudaStream_t stream);
void tp_merge_candidates(const kernels::head::Candidate* ranks, int32_t* token, int m, cudaStream_t stream);
void tp_narrow_hidden(const float* in, __nv_bfloat16* out, std::size_t n, cudaStream_t stream);

}  // namespace ingine::model
