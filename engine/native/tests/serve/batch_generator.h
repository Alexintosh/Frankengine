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
  int max_model_len() const override { return max_ctx_; }
  int batch_max_ctx() const override { return batch_ctx_ > 0 ? batch_ctx_ : max_ctx_; }
  int max_batch() const override { return 8; }
  int batch_spec() const override { return spec_; }
  int max_ctx_ = 8192;
  int batch_ctx_ = 0;
  int spec_ = 0;

  ingine::model::GenerateStats generate(
      std::span<const int32_t> prompt, int limit, const ingine::model::Sampling&, std::span<const int32_t>,
      const std::function<bool(std::span<const int32_t>)>& emit) override {
    ingine::model::GenerateStats st;
    st.prompt_tokens = int(prompt.size());
    for (int i = 0; i < limit; ++i) {
      pause();
      if (!emit(std::span(&token_, 1))) break;
    }
    return st;
  }

  int reusable_prefix(int slot, std::span<const int32_t> prompt) const override {
    if (slot < 0 || slot >= 8) return 0;
    const auto& h = histories_[std::size_t(slot)];
    return !h.empty() && h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin()) ? int(h.size()) : 0;
  }

  ingine::model::PrefillChunkResult prefill_slot_chunk(
      int slot, std::span<const int32_t> prompt, int offset, ingine::model::BatchSampling&, int stop_at = 0) override {
    pause();
    auto& h = histories_[std::size_t(slot)];
    int reused = 0;
    if (offset == 0) {
      reused = reusable_prefix(slot, prompt);
      offset = reused;
      if (!offset) h.clear();
    }
    if (stop_at > 0 && offset == stop_at && stop_at < int(prompt.size())) {
      return {offset, reused, std::nullopt};
    }
    const int target = (stop_at > offset && stop_at < int(prompt.size())) ? stop_at : int(prompt.size());
    const int next = std::min(offset + 2048, target);
    const int delta = next - offset;
    h.insert(h.end(), prompt.begin() + offset, prompt.begin() + next);
    prefilled_tokens += delta;
    if (delta > 0) chunk_sizes.push_back(delta);
    return {next, reused, next == int(prompt.size()) ? std::optional(token_) : std::nullopt};
  }

  bool clone_slot(int src, int dst) override {
    if (src < 0 || src >= 8 || dst < 0 || dst >= 8 || histories_[std::size_t(src)].empty()) return false;
    histories_[std::size_t(dst)] = histories_[std::size_t(src)];
    ++cloned;
    return true;
  }

  std::vector<int32_t> decode_batch_step(std::span<const ingine::model::BatchSlotStep> rows) override {
    pause();
    peak.store(std::max(peak.load(), int(rows.size())));
    return std::vector<int32_t>(rows.size(), token_);
  }

  void release_slot(int) override { ++released; }

  std::atomic<int> peak{0}, released{0}, cloned{0}, prefilled_tokens{0};
  std::vector<int> chunk_sizes;
  std::vector<int32_t> histories_[8];

 private:
  static void pause() { std::this_thread::sleep_for(std::chrono::milliseconds(4)); }
  const int32_t token_;
};
