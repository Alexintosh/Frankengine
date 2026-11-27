// Batched decoding: request slots with their own caches, and one decode step over many sequences at once (one
// token each, or a speculative verify of several), for a server that runs many requests together.
#pragma once

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/host_memory.h"
#include "kernels/batch/batch.h"
#include "kernels/batch/dense.h"
#include "model/model_data.h"
#include "model/tp_runtime.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

namespace ingine::model {

// The caches of `slots` request slots (one model.State each) for contexts of up to `ctx` tokens (prompt and
// generated): sliding rings of min(ctx, WINDOW) rows (a ring below WINDOW never wraps), full-attention caches of ctx
// rows, conv states. Slot s's memory is slot 0's plus s times the slot stride of each tensor.
struct BatchState {
  int slots = 0, ctx = 0, ring_rows = 0;
  bool kv_fp8 = false, kv_q4 = false, kv_12 = false;
  LoopCdConfig loopcd;
  int hkv_local = KV_HEADS, kv_dim_local = KV_DIM;
  Tensor k_ring, v_ring;       // bf16 [slots * 40, ring_rows, 8, 128]
  Tensor ring_pos;             // int64 [slots * 40, ring_rows], -1 = empty
  Tensor k_full, v_full;       // [slots * full visits, ctx, local KV heads, head elements]
  Tensor k_full_scale, v_full_scale;  // FP16 Q4 scales or U8 KV12 overflow, same leading dimensions
  Tensor conv_k, conv_v;       // fp32 [slots * 48, 3, 1024]
  Tensor conv_attn, conv_mlp;  // fp32 [slots * 48, 3, H]
  std::vector<DeviceBuffer> storage;

  // slot strides, in elements
  std::int64_t ring_stride() const { return std::int64_t(loopcd.ring_visits()) * ring_rows * kv_dim_local; }
  std::int64_t kpos_stride() const { return std::int64_t(loopcd.ring_visits()) * ring_rows; }
  std::int64_t full_stride() const { return std::int64_t(loopcd.full_visits()) * ctx * hkv_local * k_full.ne[0]; }
  std::int64_t full_scale_stride() const { return std::int64_t(loopcd.full_visits()) * ctx * hkv_local * k_full_scale.ne[0]; }
  std::int64_t conv_kv_stride() const { return std::int64_t(loopcd.visits()) * (CONV_K - 1) * kv_dim_local; }
  std::int64_t conv_h_stride() const { return std::int64_t(loopcd.visits()) * (CONV_K - 1) * HIDDEN; }
  std::size_t device_bytes() const;

  // model.State over slot s's memory (the Prefiller runs on it); owns nothing
  State view(int s) const;
};

BatchState make_batch_state(int slots, int ctx, bool kv_fp8, const TpLayout& layout = TpLayout{}, bool kv_q4 = false,
                            bool kv_12 = false, LoopCdConfig loopcd = {});
// model.State.reset for one slot: ring positions -1, conv states 0. Stream-ordered.
void reset_slot(BatchState& st, int s, cudaStream_t stream);

// Host arrays in the same flattened row order as BatchDecoder::step.
struct BatchRejectInputs {
  const std::uint8_t* active = nullptr;  // [T], 1 for rows verifying a sampled draft
  const std::int32_t* ids = nullptr;    // [T, RejectSpec::kSupport]
  const float* probs = nullptr;         // [T, RejectSpec::kSupport]
  const std::int32_t* picked = nullptr;  // [T], index into the row's support
};

// One decode step for n sequences of M consecutive rows each (M = 1: plain decoding; M > 1: a speculative verify
// of the anchor and M - 1 drafts): rows b*M .. b*M+M-1 are tokens at positions pos[b] .. pos[b]+M-1 of slot
// slots[b]. The visit sequence is the Prefiller's with n*M rows (fast.FastDecoder.decode's block-state wiring, the
// mix and message-gate kernels, native FP8 dense GEMMs or BF16 GEMMs measured in place (kernels::batch::TunedGemm),
// and the grouped MoE GEMMs); the attention is the decode attention
// kernel over each sequence's slot (attn::DecodeArgs::batch, M rows causal), the short convolutions continue each
// slot's conv state. Then the head over every row and one token per row (kernels::batch::sample).
//
// With max_m > 1 a step with more than one row for some sequence also keeps what speculative decoding needs: the
// rows' DSpark taps (taps()) and the conv inputs; its convolutions leave the slots' conv states as they were, and
// accept() writes them once each sequence's kept rows are known (engine._rollback). Full cache rows past the
// committed length are overwritten before being read. LoopCD also preserves live rows of wrapped sliding rings.
class BatchDecoder {
 public:
  BatchDecoder(const Weights& w, const Ablation& abl, BatchState& st, int max_seqs, cudaStream_t stream,
               int max_m = 1, TpRuntime* tp = nullptr, const Dense* unembed_override = nullptr);
  ~BatchDecoder();
  BatchDecoder(const BatchDecoder&) = delete;
  BatchDecoder& operator=(const BatchDecoder&) = delete;

  // The step for n <= max_seqs sequences, sequence b with m[b] rows (1..max_m) at positions pos[b] .. pos[b] + m[b]
  // - 1 of slot slots[b] (distinct slots; positions below ctx); tokens [sum m] hold the sequences' rows back to back.
  // m, tokens, slots, pos and temperature [n] are host arrays; temperature <= 0: greedy. Returns the sampled tokens,
  // one per row (row r's token follows row r; host memory owned by the decoder, valid until the next step);
  // synchronises the stream. Optional reject inputs provide the proposal distribution for active rows.
  // In Auto mode, omegas[n] sets the weight per sequence. Omitting it gives every sequence zero weight.
  std::span<const std::int32_t> step(int n, const std::int32_t* m, const std::int32_t* tokens,
                                     const std::int32_t* slots, const std::int64_t* pos, const float* temperature,
                                     std::uint64_t seed, const BatchRejectInputs* reject = nullptr,
                                     const float* omegas = nullptr);
  // every sequence with M rows
  std::span<const std::int32_t> step(int n, int M, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, const float* temperature, std::uint64_t seed) {
    return step(n, std::vector<std::int32_t>(std::size_t(n), M).data(), tokens, slots, pos, temperature, seed);
  }
  // every sequence with M rows at the same temperature
  std::span<const std::int32_t> step(int n, int M, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, float temperature, std::uint64_t seed) {
    return step(n, M, tokens, slots, pos, std::vector<float>(std::size_t(n), temperature).data(), seed);
  }
  std::span<const std::int32_t> step(int n, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, float temperature, std::uint64_t seed) {
    return step(n, 1, tokens, slots, pos, temperature, seed);
  }
  // After every step with more than one row for some sequence (a no-op otherwise): sequence b keeps its first
  // kept[b] rows (1..m[b]; host array of the step's n), and its slot's conv states become those after them.
  // Stream-ordered.
  void accept(const std::int32_t* kept);
  // The last step's DSpark taps (max_m > 1, a step with more than one row for some sequence): device bf16 [rows, 4,
  // H] (model.TAP_VISITS partial blocks, then the final normed hidden), the Prefiller's layout.
  const __nv_bfloat16* taps() const { return taps_; }

  int max_seqs() const { return max_seqs_; }
  int max_m() const { return max_m_; }
  bool reject_enabled() const { return reject_enabled_; }
  std::size_t device_bytes() const;
  // With BF16 head rows the step picks its tokens in the head GEMM (kernels::batch::head_sample) and stores the
  // logits only after keep_logits(true) (allocates them: rows x kLogitsLd fp32); FP8 head rows always store them.
  void keep_logits(bool keep);
  // Last step's logits (device fp32, row b at b * kLogitsLd). Materialized automatically with a
  // reference coda. If logits_raw(), apply the model softcap before consuming these rows.
  static constexpr int kLogitsLd = (UNPADDED_VOCAB + 7) / 8 * 8;
  const float* logits() const { return keep_logits_ || !logits_raw() ? logits_ : nullptr; }
  bool logits_raw() const { return !st_.loopcd.uses_ref_coda() &&
      (use_nvfp4_head_ || !(unembed_override_ ? unembed_override_->fp8() : w_.unembed.fp8())); }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);
  void run_body(int n, int T, int M, bool verify, std::uint64_t seed,
                const kernels::batch::RejectSpec* reject);
  void verify_rings(int n, const std::int32_t* commit);
  kernels::batch::ConvStates conv_states() const;  // slot 0's conv states and the slot strides

  const Weights& w_;
  const Dense* unembed_override_ = nullptr;
  TpRuntime* tp_ = nullptr;
  TpLayout layout_{};
  const Ablation& abl_;
  BatchState& st_;
  int max_seqs_, max_m_;
  int last_n_ = 0;            // the last step's sequences (rollback)
  bool last_ring_backup_ = false;
  bool last_verify_ = false;  // ... and whether it kept the conv history (some sequence had more than one row)
  cudaStream_t stream_;
  std::unique_ptr<kernels::batch::TunedGemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  DeviceBuffer attn_ws_;
  bool use_nvfp4_head_ = false;
  bool batch_fp8_dense_ = false;
  bool batch_pico12_gemm_ = false;
  bool batch_gate_fuse_ = false;
  bool graph_enabled_ = true;
  int graph_warmup_ = 1;
  int graph_max_seqs_ = 32;
  int graph_max_tokens_ = 256;
  using GraphKey = std::tuple<int, int, int, bool, bool>;
  std::map<GraphKey, int> graph_seen_;
  std::map<GraphKey, DecodeGraphExecutable> graphs_;
  MappedHostBuffer pin_ring_backup_;
  MappedHostBuffer pin_tokens_, pin_slots_, pin_pos_, pin_rows_, pin_temps_, pin_seed_, pin_sampled_;
  MappedHostBuffer pin_omegas_;
  std::vector<std::int32_t> sampled_host_;
  std::vector<float> temps_host_;            // per row
  std::vector<std::int32_t> rows_host_, last_m_;  // the last step's row offsets [n + 1] and rows per sequence

  // inputs and activations (rows = max_seqs * max_m), see batch.cpp
  std::int32_t *tokens_, *slots_, *sampled_;
  std::int32_t* rows_;  // row offsets, device [max_seqs + 1]
  float* temps_;        // per row
  float* omegas_dev_ = nullptr;  // per row, Auto only
  std::int64_t *pos_, *ids_;
  std::uint64_t* d_seed_ = nullptr;
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  std::uint8_t *xin_fp8_q_ = nullptr, *attn_fp8_q_ = nullptr;
  float *xin_fp8_s_ = nullptr, *attn_fp8_s_ = nullptr;
  float *mixf_ = nullptr, *msgs_ = nullptr, *mixm_ = nullptr;
  __nv_bfloat16 *mixf_bf16_ = nullptr, *msgs_bf16_ = nullptr, *mixm_bf16_ = nullptr;
  __nv_bfloat16 *proj16_, *attn_out_, *wo_out_;
  __nv_bfloat16 *gin_, *g16_;
  float* fg_ = nullptr;  // the unfused gates' forget | gain products, fp32 [3, rows, 2H]
  // per layer (loop layers): the gates' forget16 | gain16 rows, bf16 [G, 2H, GATE_RANK] (G = 3 qkv, 1 mlp)
  std::array<const __nv_bfloat16*, LAYERS> fg_qkv_{}, fg_mlp_{};
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  float* logits_ = nullptr;
  float* ref_logits_ = nullptr;
  bool keep_logits_ = false;
  DeviceBuffer logits_buf_;  // logits_ when kept (BF16 head rows)
  int2* head_partials_ = nullptr;
  bool reject_enabled_ = false;
  MappedHostBuffer pin_reject_active_, pin_reject_ids_, pin_reject_probs_, pin_reject_picked_;
  std::uint8_t* reject_active_ = nullptr;
  std::int32_t *reject_ids_ = nullptr, *reject_picked_ = nullptr;
  float* reject_probs_ = nullptr;
  kernels::batch::RejectSpec reject_spec_;
  kernels::batch::RejectShardPartial *local_reject_ = nullptr, *gathered_reject_ = nullptr;
  float* partial_ = nullptr;
  float* down_partial_ = nullptr;
  kernels::head::Candidate* local_cand_ = nullptr;
  kernels::head::Candidate* gathered_cand_ = nullptr;  // kernels::batch::head_sample's
  __nv_bfloat16* wscratch_;
  // speculative verify (max_m > 1): taps, conv inputs per visit, conv rows the step overwrites, accepted rows
  __nv_bfloat16* taps_ = nullptr;
  __nv_bfloat16 *hist_k_ = nullptr, *hist_v_ = nullptr, *hist_attn_ = nullptr, *hist_mlp_ = nullptr;
  std::int32_t* commit_ = nullptr;
};

}  // namespace ingine::model
