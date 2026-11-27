// The batched engine behind batched serving (ingine/batch.h): the weights, the request slots, the Prefiller for
// long admissions, the BatchDecoder for the steps and, with speculative decoding, the BatchDrafter.
#pragma once

#include "core/arena.h"
#include "ingine/batch.h"
#include "model/batch.h"
#include "model/batch_dflash2.h"
#include "model/tp_runtime.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ingine::model {

class BatchDrafter;
class Prefiller;

struct BatchSpecConfig {
  int adaptive = 1;
  int dcut = 1;
  float dcut_theta = 0.f;
  int dcut_min_k = -1;
  int dcut_max_t = 0;
  int dcut_bucket = 16;
  float dcut_reject_ema_weight = 0.f;
};

// Read policy settings on rank zero so both ranks compute the same row counts.
BatchSpecConfig batch_spec_config(TpRuntime* tp, cudaStream_t stream);

// Host storage shared by the serving and benchmark batch paths. A slot's temperature describes its last proposal
// until the current verification has been assembled.
class BatchDraftSupport {
 public:
  explicit BatchDraftSupport(int slots = 0, TpRuntime* tp = nullptr);
  void reset(int slot);
  void set_temperature(int slot, float temperature);
  void clear_support(int slot);
  bool has_support(int slot, float temperature) const;
  std::vector<float> proposal_temperatures(std::span<const std::int32_t> slots) const;

  // With TP, both ranks enqueue the broadcast and download. Store after the stream has synchronized.
  void download(const BatchDFlash2::Drafts& drafts, std::size_t sequences, cudaStream_t stream);
  void store(std::span<const std::int32_t> slots);
  std::size_t device_bytes() const { return broadcast_.bytes; }

  const BatchRejectInputs* verification_inputs(std::span<const std::int32_t> slots,
                                               std::span<const std::int32_t> widths,
                                               std::span<const std::int32_t> tokens,
                                               std::span<const float> temperatures);

 private:
  std::vector<float> slot_temps_;
  std::vector<bool> slot_has_support_;
  std::vector<std::int32_t> slot_ids_, slot_picked_;
  std::vector<float> slot_probs_;
  std::vector<std::int32_t> download_ids_, download_picked_;
  std::vector<float> download_probs_;
  std::vector<std::uint8_t> row_active_;
  std::vector<std::int32_t> row_ids_, row_picked_;
  std::vector<float> row_probs_;
  BatchRejectInputs inputs_;
  TpRuntime* tp_;
  DeviceBuffer broadcast_;
};

class BatchSpecCost {
 public:
  static constexpr float initial_p = 0.75f;

  struct Slot {
    int k_max;
    const float* confidence;
    float ema_p;
    int steps;
    float temperature;
    bool reject = false;
  };

  BatchSpecCost(BatchSpecConfig config, int n, int slots, float threshold);
  void allocate_widths(std::span<const Slot> slots, int k_min, std::span<std::int32_t> out_k) const;
  bool use_proposal_budgets() const { return config_.adaptive || config_.dcut; }
  int proposal_budget(int k_max, int k_min, float ema_p, int steps) const;
  static void observe(int k, int accepted, float& ema_p, int& steps);

 private:
  int verification_width(int k_max, int k_min, const float* confidence, float ema_p, int steps) const;
  int minimum_width(int k_max, int k_min, float ema_p, int steps) const;
  int dcut_minimum_width(int k_max, int k_min) const;

  BatchSpecConfig config_;
  int n_;
  bool use_threshold_;
  bool calibrate_;
  float theta_;
  float dcut_theta_;
};

struct BatchEngineOptions {
  std::filesystem::path model_dir;
  std::filesystem::path ablation;  // empty: the original model
  int slots = 64;                  // requests decoded together at most
  int ctx = 4096;                  // positions per slot
  int prefill_chunk = 512;         // prompt tokens per Prefiller call
  int spec = 0;                    // DSpark drafts verified per step and sequence (1..7, at most); 0: plain decoding
  // Base cutoff for D-cut's concurrency scaling. The legacy policy uses fixed widths when spec_threshold is 0.
  float spec_threshold = 0.f;
  int spec_min = 0;
  bool kv_fp8 = false;             // FP8 E4M3 full-attention caches
  bool kv_q4 = false;              // Q4_G64 full-attention caches
  bool kv_12 = false;              // Bit-exact BF16 full-attention caches
  bool dense_pico12 = false;       // Bit-exact BF16 projections and dense unembedding
  bool head_nvfp4 = false;         // NVFP4 unembedding head
  // Batched speculative drafter: "dspark", "dflash2", or "hybrid" (empty: INGINE_BATCH_DRAFT_TYPE, or "dflash2" if
  // dflash2_dir is given, else "dspark").
  std::string batch_draft_type;
  std::filesystem::path dflash2_dir;
  int dflash2_max_ctx = 6144;      // hybrid mode: slots with pos <= dflash2_max_ctx use BatchDFlash2
  TpOptions tp;
  // speculative: a line per sequence and step, "<slot> <drafts verified> <accepted> <conf of the 7 drafts>"
  std::filesystem::path spec_trace;
};

// BF16 dense weights (the checkpoint's), BF16 sliding caches, the checkpoint's NVFP4 / MXFP8 experts.
//
// Speculative decoding (spec = K): a step verifies [pending token, its drafts] of every sequence in one BatchDecoder
// step (up to K drafts per sequence, chosen by confidence with spec_threshold, rows packed back to back) and keeps the
// longest prefix of drafts equal to the tokens the target samples at those rows (greedy: its argmax), plus the
// target's token after them: the target's own samples, so the output is distributed as plain decoding at every
// temperature. The kept rows' taps become draft context and the drafter drafts after the new pending tokens (greedy
// drafts: the most probable match of the target's sample).
class BatchEngine final : public BatchBackend {
 public:
  explicit BatchEngine(const BatchEngineOptions& options);
  ~BatchEngine() override;
  BatchEngine(const BatchEngine&) = delete;
  BatchEngine& operator=(const BatchEngine&) = delete;

  std::string model_id() const override;
  int slots() const override { return st_.slots; }
  int ctx() const override { return st_.ctx; }
  int lookahead() const override { return spec_; }
  // Prompts of up to kStepPrefill tokens: together, in decode steps of up to 8 rows per prompt; longer ones one after
  // the other through the Prefiller. Then the first tokens (and their drafts).
  static constexpr int kStepPrefill = 256;
  std::vector<std::int32_t> admit(std::span<const Admission> requests) override;
  void step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) override;

  std::size_t device_bytes() const;  // caches, drafter and scratch (not the target's weights)
  // Speculative decoding so far: verified sequence steps, the rows they verified and the drafts they accepted.
  std::uint64_t spec_steps() const { return spec_steps_; }
  std::uint64_t spec_rows() const { return spec_rows_; }
  std::uint64_t spec_accepted() const { return spec_accepted_; }
  double draft_seconds() const { return draft_seconds_; }
  double verify_seconds() const { return verify_seconds_; }
  std::uint64_t draft_calls() const { return draft_calls_; }

 private:
  struct Stream {  // destroyed after every member that uses it
    cudaStream_t s = nullptr;
    Stream();
    ~Stream();
  } stream_;
  const bool ablated_;
  const int prefill_chunk_, spec_;
  const float spec_threshold_;
  const int spec_min_;
  std::unique_ptr<TpRuntime> tp_;
  Weights w_;
  Ablation abl_;
  BatchState st_;
  std::unique_ptr<Prefiller> pf_;
  std::unique_ptr<BatchDecoder> dec_;  // 8 rows per sequence at most (admissions, verify steps)
  std::string batch_draft_type_;
  int dflash2_max_ctx_ = 6144;
  BatchSpecConfig batch_spec_config_;
  BatchDraftSupport draft_support_;
  std::unique_ptr<BatchDrafter> drafter_;
  std::vector<DeviceBuffer> dflash2_head_storage_;
  std::unique_ptr<BatchDFlash2Head> dflash2_head_;
  std::unique_ptr<BatchDFlash2> dflash2_;
  DeviceBuffer chunk_, logits_, first_;
  DeviceBuffer taps_chunk_;  // speculative: the Prefiller's taps of a chunk
  DeviceBuffer gathered_;    // speculative: the kept rows' taps, by sequence
  DeviceBuffer rows_;
  DeviceBuffer draft_bcast_;        // speculative: int32 rows of the step's taps to gather
  std::uint64_t draws_ = 0;  // seeds the noise of sampled tokens: one per sampling call
  std::uint64_t spec_steps_ = 0, spec_rows_ = 0, spec_accepted_ = 0;
  double draft_seconds_ = 0.0, verify_seconds_ = 0.0;
  std::uint64_t draft_calls_ = 0;
  std::vector<std::int32_t> drafts_;  // speculative: [slots, GAMMA] drafts after each slot's pending token
  std::vector<float> conf_;            // speculative: [slots, GAMMA] their confidences
  std::vector<float> batch_slot_ema_p_;
  std::vector<int> batch_slot_spec_steps_;
  std::vector<int> batch_slot_draft_budget_;  // available proposals for the next verification
  std::vector<std::int32_t> draft_host_;  // speculative: the drafter's [n, GAMMA] output
  std::vector<float> conf_host_;
  std::FILE* trace_ = nullptr;
  std::vector<std::int32_t> tok_, slot_, keep_, rows_host_, anchor_;
  std::vector<std::int32_t> m_, off_;  // the step's rows per sequence and their offsets [n + 1]
  std::vector<BatchSpecCost::Slot> spec_slots_;
  std::vector<std::int64_t> pos_, next_pos_;
  std::vector<float> temp_;

  void prefill_one(const Admission& a, std::int32_t& first);
  void prefill_together(std::span<const Admission> requests, std::span<const std::size_t> which,
                        std::vector<std::int32_t>& first);
  void offsets(int n);  // off_ from m_
  // the kept rows (keep_[i] of sequence i's rows from off_[i], at positions pos_[i]..) become draft context
  void commit_kept(int n);
  // drafts after anchor_[i] at next_pos_[i] for slot_[i], into drafts_
  void draft(int n);
};

}  // namespace ingine::model
