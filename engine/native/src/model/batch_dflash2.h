// Batched DFlash2 drafter: 5-layer block-diffusion drafter (block=8, GAMMA=7, sliding window W=2048)
// supporting c=1..32+ concurrent sequences with per-slot FP8/BF16 KV rings, clone_slot, and TP=2 sharded NVFP4 head.
#pragma once

#include "kernels/head/nvfp4_head.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ingine::model {

class TpRuntime;

class BatchDFlash2Head {
 public:
  static constexpr int GAMMA = 7;

  BatchDFlash2Head(TpRuntime& tp, kernels::head::Nvfp4Matrix matrix, int max_seqs, cudaStream_t stream);
  ~BatchDFlash2Head();
  BatchDFlash2Head(const BatchDFlash2Head&) = delete;
  BatchDFlash2Head& operator=(const BatchDFlash2Head&) = delete;

  // Rank 0 passes `hidden` [n * 7, H]; rank 1 passes nullptr.
  void run(int n, const __nv_bfloat16* hidden = nullptr);
  // Both ranks pass the same compacted row count.
  void run_rows(int total_rows, const __nv_bfloat16* hidden = nullptr);
  const std::int32_t* ids() const;
  const float* values() const;
  std::size_t device_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class BatchDFlash2 {
 public:
  static constexpr int BLOCK = 8;
  static constexpr int GAMMA = 7;
  static constexpr int WINDOW = 2048;
  static constexpr int MASK_TOKEN = 200056;
  static constexpr int COMMIT_PIECE = 2048;

  enum class Precision { MXFP8, BF16 };

  // checkpoint: directory containing DFlash2 model.safetensors.
  // w: target model Weights (provides embed, embed_norm, and optionally unembed_nvfp4).
  // head_override: optional full NVFP4 head if w.unembed_nvfp4 is not available (TP=1 without --head-nvfp4).
  // sharded_head: optional TP=2 sharded head (when non-null, takes precedence over w.unembed_nvfp4 / head_override).
  BatchDFlash2(const std::filesystem::path& checkpoint, const Weights& w, int slots, int max_seqs,
               cudaStream_t stream, Precision precision = Precision::MXFP8, bool kv_fp8 = true,
               const kernels::head::Nvfp4Matrix* head_override = nullptr,
               BatchDFlash2Head* sharded_head = nullptr);
  ~BatchDFlash2();
  BatchDFlash2(const BatchDFlash2&) = delete;
  BatchDFlash2& operator=(const BatchDFlash2&) = delete;

  void reset_slot(int slot);
  void clone_slot(int src_slot, int dst_slot, int len);

  // Target taps [sum count, 4, H] become draft context for each sequence i in 0..n-1.
  void commit(int n, const std::int32_t* slots, const std::int64_t* start, const std::int32_t* count,
              const __nv_bfloat16* taps);

  struct Drafts {
    const std::int32_t* tokens;  // device int32 [n, 7]
    const float* conf;           // device fp32 [n, 7]
  };
  // Optional per-sequence budgets are clamped to [1, GAMMA]. Unused output entries are zero.
  Drafts propose(int n, const std::int32_t* slots, const std::int32_t* anchors, const std::int64_t* pos,
                 const std::int32_t* max_drafts = nullptr);

  int slots() const;
  int max_seqs() const;
  std::size_t device_bytes() const;

  static Precision default_precision();
  static bool default_kv_fp8();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ingine::model
