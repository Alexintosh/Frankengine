#pragma once

#include "ingine/generator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

class BatchGenerator final : public ingine::model::Generator {
 public:
  explicit BatchGenerator(int32_t token) : token_(token) {}
  std::string model_id() const override { return "batch-test"; }
  int max_model_len() const override { return 8192; }
  int max_batch() const override { return 8; }
  int batch_spec() const override { return spec_; }
  int spec_ = 0;

  ingine::model::GenerateStats generate(
      std::span<const int32_t>, int limit, const ingine::model::Sampling&, std::span<const int32_t>,
      const std::function<bool(std::span<const int32_t>)>& emit) override {
    for (int i = 0; i < limit; ++i) {
      pause();
      if (!emit(std::span(&token_, 1))) break;
    }
    return {};
  }

  ingine::model::PrefillChunkResult prefill_slot_chunk(
      int, std::span<const int32_t> prompt, int offset, ingine::model::BatchSampling&, int stop_at = 0) override {
    pause();
    const int target = (stop_at > offset && stop_at < int(prompt.size())) ? stop_at : int(prompt.size());
    offset = std::min(offset + 2048, target);
    return {offset, 0, offset == int(prompt.size()) ? std::optional(token_) : std::nullopt};
  }

  bool clone_slot(int, int) override {
    ++cloned;
    return true;
  }

  std::vector<int32_t> decode_batch_step(std::span<const ingine::model::BatchSlotStep> rows) override {
    pause();
    peak.store(std::max(peak.load(), int(rows.size())));
    return std::vector<int32_t>(rows.size(), token_);
  }

  void release_slot(int) override { ++released; }

  std::atomic<int> peak{0}, released{0}, cloned{0};

 private:
  static void pause() { std::this_thread::sleep_for(std::chrono::milliseconds(4)); }
  const int32_t token_;
};
