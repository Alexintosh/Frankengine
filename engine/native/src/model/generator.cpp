// The resident model supports full turns and step-level continuous batching.
// In a full turn, Prefiller advances the prompt, then plain decode pipelines
// width-one steps or SpecLoop verifies draft proposals.
// Batched serving keeps a committed history for each slot and advances rows
// whose next input token is ready. The solo path remains available on slot 0
// and can hand its state back to the scheduler at a safe boundary.
#include "ingine/generator.h"

#include "core/arena.h"
#include "core/device.h"
#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/dflash2.h"

#include <cstdlib>
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/spec.h"
#include "model/vision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace head = kernels::head;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const RuntimeOptions& without_spec(const RuntimeOptions& options) {
  if (options.spec) throw std::invalid_argument("make_generator over given weights: spec needs the model directory");
  return options;
}

class EngineGenerator final : public Generator {
 public:
  explicit EngineGenerator(const RuntimeOptions& options)
      : EngineGenerator(options, std::make_unique<Runtime>(options)) {}
  EngineGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : EngineGenerator(without_spec(options),
                        std::make_unique<Runtime>(options, std::move(weights), std::move(ablation))) {}

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }

  bool supports_images() const override { return vision_ != nullptr; }

 private:
  EngineGenerator(const RuntimeOptions& options, std::unique_ptr<Runtime> runtime)
      : options_(options),
        runtime_(std::move(runtime)),
        rt_(*runtime_),
        prefiller_(rt_.weights(), rt_.ablation(), options.prefill_chunk, rt_.stream(), rt_.tp()),
        chunk_tokens_(std::size_t(options.prefill_chunk) * sizeof(std::int32_t)),
        logits_(rt_.tp() && !options.tp.serve ? sizeof(int32_t) : std::size_t(UNPADDED_VOCAB) * sizeof(float)),
        token_(sizeof(std::int32_t)),
        host_token_(sizeof(std::int32_t)) {
    if (rt_.tp() && options.tp.full_logits)
      throw std::invalid_argument("TP=2 generator uses the greedy head");
    if (options.prefill_chunk < 1) throw std::invalid_argument("prefill_chunk must be positive");
    if (options.draft_type != "dspark" && options.draft_type != "dflash2" && options.draft_type != "hybrid")
      throw std::invalid_argument("Unknown draft type");
    const bool use_dflash2 = options.draft_type != "dspark";
    if (options.dflash2_max_ctx < 1)
      throw std::invalid_argument("dflash2_max_ctx must be positive");
    if (!(options.hybrid_dflash2_threshold >= 0.f && options.hybrid_dflash2_threshold <= 1.f))
      throw std::invalid_argument("dflash2-threshold must be in [0, 1]");
    if (use_dflash2 && !options.spec)
      throw std::invalid_argument("DFlash2 requires speculative decoding");
    if (use_dflash2 && options.head_nvfp4 && rt_.tp() && !dflash2_full_head())
      dflash2_head_ = std::make_unique<DFlash2Head>(*rt_.tp(), rt_.weights().unembed_nvfp4, rt_.stream());
    if (use_dflash2 && options.tp.rank == 0) {
      auto dir = options.dflash2_dir;
      if (dir.empty())
        if (const char* env = std::getenv("INKLING_DFLASH2_DIR")) dir = env;
      if (dir.empty()) throw std::invalid_argument("DFlash2 requires a checkpoint directory");
      dflash2_ = std::make_unique<DFlash2>(dir, rt_.stream(), options.draft_type == "dflash2" && options.draft_nvfp4);
      const auto* embed = ptr<const __nv_bfloat16>(rt_.weights().embed);
      const auto* embed_norm = ptr<const float>(rt_.weights().embed_norm);
      if (dflash2_head_) {
        dflash2_->bind_sharded_head(*dflash2_head_, embed, embed_norm);
      } else if (!rt_.tp() && options.head_nvfp4) {
        dflash2_->bind_head(rt_.weights().unembed_nvfp4, embed, embed_norm);
      } else {
        const auto head = load_head(options.model_dir, false, true, rt_.stream(), dflash2_head_storage_);
        dflash2_->bind_head(head.nvfp4, embed, embed_norm);
      }
      dflash2_->warmup();
    } else if (dflash2_head_) {
      // Match the rank-0 startup proposal's head exchanges on the worker.
      dflash2_head_->run();
    }
    if (options.spec && options.draft_type != "dflash2" && options.tp.rank == 0) {
      // The folded head and context shortlist need every token row. Reuse the verifier's dense
      // unembedding when it already has that layout and retain a separate dense source otherwise.
      Dense head;
      if (rt_.tp() || options.head_nvfp4)
        head = load_head(options.model_dir, options.dense_fp8, false, rt_.stream(), draft_head_storage_).dense;
      drafter_ = std::make_unique<Drafter>(options.model_dir, rt_.weights(), options.max_ctx, rt_.stream(),
                                           Drafter::SHORTLIST_ROWS, options.draft_nvfp4, head.w.data ? &head : nullptr,
                                           rt_.tp() && rt_.tp()->profile);
      drafter_->capture();
    }
    if (drafter_ || dflash2_)
      taps_ = DeviceBuffer(std::size_t(options.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
    if (options.spec && options.draft_type != "dflash2")
      spec_ = std::make_unique<SpecLoop>(rt_, drafter_.get());
    if (use_dflash2)
      dflash2_spec_ = std::make_unique<SpecLoop>(rt_, nullptr, dflash2_.get(), dflash2_head_.get());
    // checkpoint layout: target state | drafter context | last prompt logits
    checkpoint_drafter_ = carved(rt_.checkpoint_bytes());
    checkpoint_logits_ = checkpoint_drafter_ + (drafter_ ? carved(drafter_->checkpoint_bytes()) : 0);
    checkpoint_ = DeviceBuffer(checkpoint_logits_ + logits_.bytes);
    // the vision tower's weights are resident: loaded before the park budget below is measured
    if (options.vision) vision_ = std::make_unique<VisionEncoder>(options.model_dir, rt_.stream());
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    if (dflash2_head_) rt_.tp()->check();
    if (rt_.tp() && options.park_mb < 0) {
      park_budget_ = 0;
    } else if (options.park_mb < 0) {  // leave room for the workspaces a long prefill adds later
      std::size_t free = 0, total = 0;
      CUDA_CHECK(cudaMemGetInfo(&free, &total));
      park_budget_ = std::min<std::size_t>(3072, std::max<std::size_t>(free >> 20, 4096) - 4096) << 20;
    } else {
      park_budget_ = std::size_t(options.park_mb) << 20;
    }
    // parked conversations are best-effort: a device allocation that fails releases them and retries
    release_hook_ = add_release_hook([this] { return release_parked(); });
  }

 public:
  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens);
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate_impl(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, nullptr);
  }

 private:
  GenerateStats generate_impl(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                              std::span<const int32_t> stop_ids,
                              const std::function<bool(std::span<const int32_t>)>& on_tokens,
                              uint64_t* counter) {
    const std::span<const std::int32_t> prompt = full_prompt.ids;
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    if (options_.draft_type == "dflash2" && sampling.temperature > 0 && sampling.forced.empty())
      throw std::invalid_argument("DFlash2 currently supports greedy generation");
    if (rt_.tp() && !options_.tp.serve && sampling.temperature != 0)
      throw std::invalid_argument("TP=2 generator requires temperature 0");
    const int n = int(prompt.size());
    const bool flash = options_.draft_type == "dflash2" ||
        (options_.draft_type == "hybrid" && n <= options_.dflash2_max_ctx && sampling.temperature == 0);
    auto* spec = flash ? dflash2_spec_.get() : spec_.get();
    const int image_tokens = check_images(full_prompt);
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    // a verify call writes up to max(verify widths) positions
    const int reserve = spec ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    GenerateStats stats;
    stats.used_dflash2 = flash;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::lock_guard<std::mutex> lock(mutex_);

    if (rt_.tp() && options_.tp.serve) rt_.tp()->full_logits = sampling.temperature > 0;

    auto t = std::chrono::steady_clock::now();
    try {
      std::uint64_t request = 0;
      if (rt_.tp()) {
        request = 14695981039346656037ull;
        auto mix = [&](std::span<const int32_t> tokens) {
          request = (request ^ tokens.size()) * 1099511628211ull;
          for (int32_t id : tokens) request = (request ^ std::uint32_t(id)) * 1099511628211ull;
        };
        mix(prompt);
        mix(std::span<const int32_t>(&max_new_tokens, 1));
        mix(sampling.forced);
        mix(stop_ids);
        request = (request ^ std::uint64_t(flash)) * 1099511628211ull;
      }
      if (flash && !previous_flash_ && !options_.cold_prefill) park_live();
      if (flash || previous_flash_) {
        tokens_in_state_.clear();
        checkpoint_ids_.clear();
      }
      previous_flash_ = flash;
      prefill(prompt, full_prompt.images, stats, request, flash, counter ? &on_tokens : nullptr);
      if (stats.interrupted_prefill_offset) return stats;
      stats.prompt_tokens = n;
      stats.image_tokens = image_tokens;
      if (spec) {
        std::int32_t token;
        if (rt_.tp() && rt_.tp()->full_logits) {
          token = sample_tp(logits_.p, sampling, rng);
        } else if (rt_.tp()) {
          CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), logits_.p, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                     rt_.stream()));
          rt_.tp()->wait(rt_.stream());
          token = *static_cast<const int32_t*>(host_token_.data());
        } else {
          token = sample(static_cast<const float*>(logits_.p), sampling, rng);
        }
        if (!sampling.forced.empty()) token = sampling.forced[0];
        stats.prefill_seconds = seconds_since(t);
        t = std::chrono::steady_clock::now();
        if (max_new_tokens > 0) {
          if (drafter_ && !flash) drafter_->set_seed(rng.seed);
          spec->generate(token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, tokens_in_state_);
        }
      } else if (rt_.tp()) {
        decode_tp(n, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, t);
      } else {
        decode(n, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, t);
      }
      if (!rt_.tp()) CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    } catch (...) {
      forget_state();
      throw;
    }
    if (counter) *counter = rng.counter;
    stats.decode_seconds = seconds_since(t);
    return stats;
  }

 public:
  int max_batch() const override { return options_.max_batch; }

  int reusable_prefix(int slot, std::span<const int32_t> prompt) const override {
    if (options_.cold_prefill) return 0;
    const auto& history = slot_history(slot);
    // Precompute the final token's head even when a previous turn used a different head mode.
    return !history.empty() && history.size() < prompt.size() &&
                   std::equal(history.begin(), history.end(), prompt.begin())
               ? int(history.size()) : 0;
  }

  PrefillChunkResult prefill_slot_chunk(int slot, std::span<const int32_t> prompt, int offset,
                                       BatchSampling& sampling) override {
    if (prompt.empty() || prompt.size() >= std::size_t(options_.max_ctx) ||
        std::any_of(prompt.begin(), prompt.end(), [](int32_t token) {
          return token < 0 || token >= UNPADDED_VOCAB;
        }))
      throw std::invalid_argument("invalid batch prompt");
    auto& state = rt_.state_slot(slot);
    auto& history = slot_history(slot);
    const auto stream = rt_.stream();
    int reused = 0;
    if (offset == 0) {
      reused = reusable_prefix(slot, prompt);
      if (rt_.tp() && !rt_.tp()->agree(reused, stream)) reused = 0;
      if (slot == 0) {
        park_live();
        checkpoint_ids_.clear();
        previous_flash_ = false;
        if (drafter_) drafter_->reset();
      }
      offset = reused;
      if (!offset) {
        reset_state(state, stream);
        history.clear();
      }
    }
    const int count = std::min({2048, options_.prefill_chunk, int(prompt.size()) - offset});
    const bool final = offset + count == int(prompt.size());
    const bool sampled = sampling.params.temperature > 0;
    auto* tp = rt_.tp();
    if (tp) tp->full_logits = sampled;
    CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, prompt.data() + offset, count * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    prefiller_.run(state, static_cast<const int32_t*>(chunk_tokens_.p), count, offset,
                   !tp || (sampled && options_.tp.rank == 0) ? static_cast<float*>(logits_.p) : nullptr,
                   nullptr, {}, tp && !sampled ? static_cast<int32_t*>(logits_.p) : nullptr, final);
    if (tp) tp->wait(stream);
    else CUDA_CHECK(cudaStreamSynchronize(stream));
    history.insert(history.end(), prompt.begin() + offset, prompt.begin() + offset + count);
    state.length = offset + count;
    PrefillChunkResult result{state.length, reused, std::nullopt};
    if (final && options_.tp.rank == 0)
      result.token = batch_token(logits_.p, tp && !sampled, sampling);
    return result;
  }

  std::vector<int32_t> decode_batch_step(std::span<const BatchSlotStep> slots) override {
    std::vector<Runtime::BatchInput> rows;
    bool sampled = false;
    for (const auto& slot : slots) {
      rt_.state_slot(slot.slot).length = int(slot_history(slot.slot).size());
      rows.push_back({slot.slot, slot.token});
      sampled |= slot.sampling->params.temperature > 0;
    }
    if (rows.size() == 1 && rows.front().slot == 0) {
      if (rt_.tp()) rt_.tp()->full_logits = sampled;
      const int32_t input = rows.front().token;
      rt_.set_inputs(std::span(&input, 1), int(tokens_in_state_.size()));
      const float* logits = rt_.run_step(1);
      tokens_in_state_.push_back(input);
      if (rt_.tp()) rt_.tp()->wait(rt_.stream());
      if (options_.tp.rank != 0) return {};
      const bool greedy = rt_.tp() && !sampled;
      const void* output = greedy ? rt_.buffers(1).greedy.data : logits;
      return {batch_token(output, greedy, *slots.front().sampling)};
    }
    auto& buffers = rt_.run_batch(rows, sampled);
    std::vector<int32_t> tokens;
    const bool greedy_head = rt_.tp() && !sampled;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      slot_history(slots[i].slot).push_back(slots[i].token);
      if (options_.tp.rank == 0) {
        const void* output = greedy_head
            ? static_cast<const void*>(ptr<int32_t>(buffers.greedy) + i)
            : static_cast<const void*>(ptr<float>(buffers.logits) + i * UNPADDED_VOCAB);
        tokens.push_back(batch_token(output, greedy_head, *slots[i].sampling));
      }
    }
    return tokens;
  }

  GenerateStats generate_interruptible(const Prompt& prompt, int limit, BatchSampling& sampling,
                                       std::span<const int32_t> stops,
                                       const std::function<bool(std::span<const int32_t>)>& emit) override {
    auto params = sampling.params;
    params.seed = sampling.seed;
    return generate_impl(prompt, limit, params, stops, emit, &sampling.counter);
  }

  ~EngineGenerator() override {
    remove_release_hook(release_hook_);
    // nothing queued may outlive the graphs and buffers of the members destroyed below
    if (rt_.stream()) cudaStreamSynchronize(rt_.stream());
  }

 private:
  // The images of a prompt lie inside it, in order and without overlap, and cover exactly its negative ids
  // (an image token's id; anything else negative would index outside the embedding table). Returns their tokens.
  int check_images(const Prompt& prompt) const {
    if (!prompt.images.empty() && !vision_) {
      throw std::invalid_argument("the prompt has images but the vision tower is not loaded (RuntimeOptions::vision)");
    }
    const std::int64_t n = std::int64_t(prompt.ids.size());
    std::int64_t end = 0, tokens = 0;
    for (const PromptImage& image : prompt.images) {
      if (!image.rgb || image.height < 1 || image.width < 1 ||
          image.rgb->size() != std::size_t(image.height) * std::size_t(image.width) * 3) {
        throw std::invalid_argument("a prompt image has no pixels or pixels of another size");
      }
      if (image.pos < end || std::int64_t(image.pos) + image.tokens() > n) {
        throw std::invalid_argument("the prompt's images overlap, are out of order or lie outside the prompt");
      }
      end = std::int64_t(image.pos) + image.tokens();
      tokens += image.tokens();
    }
    const std::int64_t negative = std::count_if(prompt.ids.begin(), prompt.ids.end(), [](std::int32_t id) { return id < 0; });
    bool covered = negative == tokens;
    for (const PromptImage& image : prompt.images) {
      for (int k = 0; covered && k < image.tokens(); ++k) covered = prompt.ids[std::size_t(image.pos + k)] < 0;
    }
    if (!covered) throw std::invalid_argument("negative token ids of the prompt are not exactly its image tokens");
    return int(tokens);
  }

  // After a failure anywhere in a request the device state may be ahead of or behind tokens_in_state_ (a step
  // in flight, a partial prefill, a round between launch and bookkeeping): wait for the stream and forget
  // everything reusable, so the next request starts from an empty state.
  void forget_state() noexcept {
    cudaStreamSynchronize(rt_.stream());
    tokens_in_state_.clear();
    checkpoint_ids_.clear();
    if (drafter_) drafter_->reset();  // stream-ordered; CUDA failures abort (CUDA_CHECK)
  }

  // The paired CLI processes submit the same prompts, forced tokens, and stop policy.
  // Both ranks obtain the merged token on device. Only execute a step after its input
  // has been accepted by the caller, so there is no speculative state to undo.
  int32_t sample_tp(const void* logits, const Sampling& sampling, head::SamplerRng& rng) {
    const auto stream = rt_.stream();
    if (options_.tp.rank == 0)
      head::sample(static_cast<const float*>(logits), 1, UNPADDED_VOCAB, UNPADDED_VOCAB,
                   {sampling.temperature, sampling.top_k, sampling.top_p}, rng,
                   static_cast<int32_t*>(token_.p), stream);
    rt_.tp()->broadcast_proposal(token_.p, sizeof(int32_t), stream);
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    rt_.tp()->wait(stream);
    return *static_cast<const int32_t*>(host_token_.data());
  }

  void decode_tp(int n, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
                 std::span<const int32_t> stop_ids,
                 const std::function<bool(std::span<const int32_t>)>& on_tokens, GenerateStats& stats,
                 std::chrono::steady_clock::time_point& t) {
    auto fetch = [&](const void* device) {
      CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), device, sizeof(int32_t), cudaMemcpyDeviceToHost, rt_.stream()));
      rt_.tp()->wait(rt_.stream());
      return *static_cast<const int32_t*>(host_token_.data());
    };
    const bool sampled = rt_.tp()->full_logits;
    int32_t token = sampled ? sample_tp(logits_.p, sampling, rng) : fetch(logits_.p);
    stats.prefill_seconds = seconds_since(t);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
      if (!sampling.forced.empty()) token = sampling.forced[i];
      stats.generated_tokens = i + 1;
      const bool go_on = on_tokens(std::span<const int32_t>(&token, 1));
      if (i + 1 == max_new_tokens || std::find(stop_ids.begin(), stop_ids.end(), token) != stop_ids.end()) break;
      if (options_.tp.serve) {
        if (!rt_.tp()->agree(go_on, rt_.stream()) || !go_on) break;
      } else if (!go_on) {
        break;
      }
      rt_.set_inputs(std::span<const int32_t>(&token, 1), n + i);
      const float* next_logits = rt_.run_step(1);
      tokens_in_state_.push_back(token);
      token = sampled ? sample_tp(next_logits, sampling, rng) : fetch(rt_.buffers(1).greedy.data);
      if (rt_.tp()->profile) {
        const auto step = rt_.tp()->timing();
        auto& total = stats.decode_timing;
        total.steps += step.steps;
        total.step_ms += step.step_ms;
        total.reduce_ms += step.reduce_ms;
        total.head_exchange_ms += step.head_exchange_ms;
      }
    }
  }

  // Plain decoding after the prefill of n tokens (logits in logits_): the token at position p is sampled on
  // the device, chain_advance makes it the width-1 step's input, and the step at p is launched before the
  // host reads that token, so the GPU never waits for the host. A step launched for a token that ends the
  // generation (stop token, cancel) is undone, leaving the state at tokens_in_state_. Sets prefill_seconds
  // (until the first token is known) and restarts `t` for the decode time.
  void decode(int n, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
              std::span<const std::int32_t> stop_ids,
              const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
              std::chrono::steady_clock::time_point& t) {
    const cudaStream_t stream = rt_.stream();
    const head::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
    auto* sampled = static_cast<std::int32_t*>(token_.p);
    // forced replay: the forced tokens on the device; each sampled token is replaced by its forced one
    DeviceBuffer forced;
    if (!sampling.forced.empty()) {
      forced = DeviceBuffer(sampling.forced.size_bytes());
      CUDA_CHECK(cudaMemcpyAsync(forced.p, sampling.forced.data(), sampling.forced.size_bytes(),
                                 cudaMemcpyHostToDevice, stream));
    }
    // engine._sample of the logits into `sampled`, then the inputs of the step at `position`
    auto sample_for = [&](const float* logits, int position) {
      head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
      if (forced.p)
        CUDA_CHECK(cudaMemcpyAsync(sampled, static_cast<const std::int32_t*>(forced.p) + (position - n),
                                   sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
      rt_.chain_advance(sampled, position);
    };
    // the step at `position` (its input token is on the device), then the next token
    auto launch = [&](int position) { sample_for(rt_.run_step(1), position + 1); };
    auto is_stop = [&](std::int32_t tok) { return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end(); };

    sample_for(static_cast<const float*>(logits_.p), n);
    if (max_new_tokens > 1) launch(n);
    std::int32_t token = rt_.chain_token(n);
    stats.prefill_seconds = seconds_since(t);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
      // `token` is at position n + i; its step is in flight unless it is the budget's last token
      const bool last = i + 1 == max_new_tokens;
      stats.generated_tokens = i + 1;
      const std::int32_t out[1] = {token};
      const bool go_on = on_tokens(out);
      if (last || is_stop(token) || !go_on) {
        if (!last) {
          rt_.chain_undo(n + i);
          if (sampling.temperature > 0) --rng.counter;
        }
        break;
      }
      tokens_in_state_.push_back(token);
      if (i + 2 < max_new_tokens) launch(n + i + 1);
      token = rt_.chain_token(n + i + 1);
    }
  }

  // engine._prefill with prompt checkpoints: brings the state to `ids` (last logits in logits_) and sets
  // stats.reused_tokens and stats.reuse.
  // * Extend: `ids` extends tokens_in_state_ (the previous prompt and the generated tokens processed since):
  //   prefill the new suffix.
  // * Checkpoint: the prompt of the request that saved the checkpoint (the state right after its prefill) is a
  //   prefix of `ids`: restore it and prefill the rest. This is the agent-loop case: the next prompt
  //   re-renders the previous answer differently from the tokens the model generated.
  // * Parked: a parked conversation's prompt is a longer prefix of `ids` than the live state offers: park the
  //   live conversation, restore the parked one's checkpoint and prefill the rest.
  // * None: park the live conversation, reset and prefill everything.
  // After a prefill the checkpoint moves to the end of `ids`.
  //
  // Images: an image token's id is a function of the image's content, so the comparisons above tell images
  // apart. The vision tower encodes an image when the first chunk that holds some of its tokens is reached
  // (images inside the reused prefix are not encoded again), and the Prefiller writes its rows over the
  // embeddings of its tokens; an image may span chunks.
  void prefill(std::span<const std::int32_t> ids, std::span<const PromptImage> images, GenerateStats& stats,
               std::uint64_t request, bool flash,
               const std::function<bool(std::span<const int32_t>)>* proceed = nullptr) {
    const bool cold = options_.cold_prefill || flash;
    const std::vector<std::int32_t>& prev = tokens_in_state_;
    const std::size_t n = ids.size();
    const bool drafter_in_step = !drafter_ || drafter_->length() == int(prev.size());  // else a row is missing
    State& st = rt_.state();
    const cudaStream_t stream = rt_.stream();
    const bool extend =
        !cold && drafter_in_step && !prev.empty() && prev.size() < n && std::equal(prev.begin(), prev.end(), ids.begin());
    // A different head mode can reuse the cached state if prefill computes fresh final logits.
    const bool sampled_head = rt_.tp() && rt_.tp()->full_logits;
    const bool checkpoint = !cold && !extend && !checkpoint_ids_.empty() && checkpoint_ids_.size() <= n &&
                            (checkpoint_ids_.size() < n || checkpoint_sampled_head_ == sampled_head) &&
                            std::equal(checkpoint_ids_.begin(), checkpoint_ids_.end(), ids.begin());
    Parked* parked = cold ? nullptr :
        find_parked(ids, extend ? prev.size() : checkpoint ? checkpoint_ids_.size() : 0);
    std::size_t start = 0;
    if (parked) {
      // parking the live conversation may move the parked list (never evicting the one to restore)
      const std::vector<std::int32_t> keep = parked->ids;
      park_live(&keep);
      unpark(*std::find_if(parked_.begin(), parked_.end(), [&](const Parked& p) { return p.ids == keep; }));
      restore_checkpoint();
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Parked;
    } else if (extend) {
      start = prev.size();
      stats.reuse = Reuse::Extend;
    } else if (checkpoint) {
      restore_checkpoint();
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Checkpoint;
    } else {
      if (!cold) park_live();
      reset_state(st, stream);
      if (flash && dflash2_) dflash2_->reset();
      if (!flash && drafter_) drafter_->reset();
      tokens_in_state_.clear();
      checkpoint_ids_.clear();
      stats.reuse = Reuse::None;
    }
    if (rt_.tp()) {
      if (options_.tp.serve && !rt_.tp()->agree(start, stream)) {
        forget_state();
        reset_state(st, stream);
        if (!flash && drafter_) drafter_->reset();
        start = 0;
        stats.reuse = Reuse::None;
      }
      rt_.tp()->verify_request((request ^ start) * 1099511628211ull, stream);
    }
    st.length = int(start);  // decode steps advance the state without its host-side length
    const std::size_t chunk = std::size_t(proceed ? std::min(2048, options_.prefill_chunk) : options_.prefill_chunk);
    struct Encoded {
      const PromptImage* image;
      DeviceBuffer rows;  // bf16 [>= tokens, HIDDEN]
    };
    std::deque<Encoded> encoded;  // images whose rows the chunks still need (or were queued to read)
    std::size_t next_image = 0;
    while (next_image < images.size() &&
           std::size_t(images[next_image].pos) + std::size_t(images[next_image].tokens()) <= start) {
      ++next_image;  // inside the reused prefix
    }
    std::vector<EmbedRows> rows;
    for (std::size_t s = start; s < n; s += chunk) {
      const int T = int(std::min(chunk, n - s));
      for (; next_image < images.size() && std::size_t(images[next_image].pos) < s + std::size_t(T); ++next_image) {
        const PromptImage& image = images[next_image];
        const auto t = std::chrono::steady_clock::now();
        DeviceBuffer out = vision_->encode(image.rgb->data(), image.height, image.width);
        // encode() returns with the stream idle: the rows of images that ended before this chunk have been read
        while (!encoded.empty() && std::size_t(encoded.front().image->pos + encoded.front().image->tokens()) <= s) {
          encoded.pop_front();
        }
        encoded.push_back(Encoded{&image, std::move(out)});
        stats.vision_seconds += seconds_since(t);
        ++stats.images_encoded;
      }
      rows.clear();
      for (const Encoded& e : encoded) {
        const std::int64_t first = std::max<std::int64_t>(e.image->pos, std::int64_t(s));
        const std::int64_t last = std::min<std::int64_t>(std::int64_t(e.image->pos) + e.image->tokens(), std::int64_t(s) + T);
        if (first < last) {
          rows.push_back(EmbedRows{int(first - std::int64_t(s)), int(last - first),
                                   static_cast<const __nv_bfloat16*>(e.rows.p) + std::size_t(first - e.image->pos) * HIDDEN});
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, ids.data() + s, std::size_t(T) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream));
      Tensor taps;
      if (flash ? bool(dflash2_) : bool(drafter_)) taps = rm(taps_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
      prefiller_.run(st, static_cast<const int32_t*>(chunk_tokens_.p), T, int(s),
                     rt_.tp() && (!rt_.tp()->full_logits || options_.tp.rank != 0) ? nullptr : static_cast<float*>(logits_.p), (flash ? bool(dflash2_) : bool(drafter_)) ? &taps : nullptr, rows,
                     rt_.tp() && !rt_.tp()->full_logits ? static_cast<int32_t*>(logits_.p) : nullptr, (!proceed && !rt_.tp()) || s + T == n);
      if (!flash && drafter_) {
        drafter_->note_tokens(ids.subspan(s, std::size_t(T)));  // the prompt's tokens join the drafter's shortlist
        drafter_->commit(taps, int(s));
      }
      if (flash && dflash2_) dflash2_->commit(ptr<const __nv_bfloat16>(taps), T, int(s));
      if (proceed && s + T < n && images.empty()) {
        const bool go_on = (*proceed)({});
        const bool agreed = !rt_.tp() || rt_.tp()->agree(go_on, stream);
        if (!agreed || !go_on) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
          tokens_in_state_.assign(ids.begin(), ids.begin() + s + T);
          if (!cold) park_live();
          checkpoint_ids_.clear();
          previous_flash_ = false;
          stats.reused_tokens = int(start);
          stats.interrupted_prefill_offset = int(s + T);
          return;
        }
      }
    }
    if (!encoded.empty()) CUDA_CHECK(cudaStreamSynchronize(stream));  // queued chunks read the rows freed here
    tokens_in_state_.assign(ids.begin(), ids.end());
    if (start < n && !cold) save_checkpoint();
    stats.reused_tokens = int(start);
  }

  // Conversation parking. A parked conversation is a packed copy of a prompt checkpoint (target state |
  // drafter checkpoint regions | last logits) plus the target's full-attention rows [0, n) of its n prompt
  // tokens, in device memory within park_budget_ (least recently used evicted), and the drafter's context
  // tokens of the checkpoint (its shortlist rows).
  struct Parked {
    std::vector<std::int32_t> ids, context;
    DeviceBuffer data;
    bool sampled_head = false;
    std::uint64_t used = 0;
  };
  static constexpr std::size_t kMinParked = 256;  // shorter prompts are cheaper to prefill than to keep

  // the layout of a parked copy of n tokens: {offset, bytes} of each drafter region, logits, full rows
  struct ParkLayout {
    std::array<Drafter::Region, 3> drafter{};
    std::size_t drafter_off[3] = {}, logits = 0, rows = 0, bytes = 0;
  };
  ParkLayout park_layout(int n) const {
    ParkLayout l;
    std::size_t off = checkpoint_drafter_;
    if (drafter_) {
      l.drafter = drafter_->checkpoint_regions(n);
      for (int i = 0; i < 3; ++i) l.drafter_off[i] = off, off += carved(l.drafter[i].bytes);
    }
    l.logits = off;
    l.rows = off + carved(logits_.bytes);
    l.bytes = l.rows + rt_.full_rows_bytes(n);
    return l;
  }

  // the parked conversation whose prompt is the longest prefix of `ids`, if longer than `live` (and kMinParked)
  Parked* find_parked(std::span<const std::int32_t> ids, std::size_t live) {
    Parked* best = nullptr;
    const bool sampled_head = rt_.tp() && rt_.tp()->full_logits;
    for (Parked& p : parked_)
      if (p.ids.size() >= kMinParked && p.ids.size() > live && p.ids.size() <= ids.size() &&
          (p.ids.size() < ids.size() || p.sampled_head == sampled_head) &&
          (!best || p.ids.size() > best->ids.size()) && std::equal(p.ids.begin(), p.ids.end(), ids.begin()))
        best = &p;
    return best;
  }

  // The release hook (core/arena.h): a device allocation failed; drop every parked conversation so that it can
  // be retried. Runs on the allocating thread, i.e. inside a request (the Generator allocates nothing else).
  bool release_parked() {
    if (parking_ || parked_.empty()) return false;
    cudaStreamSynchronize(rt_.stream());  // copies from or into parked buffers may be queued
    std::fprintf(stderr, "device memory low: released %zu parked conversation(s)\n", parked_.size());
    parked_.clear();
    return true;
  }

  // Parks the live conversation's checkpoint (the state after its last prompt; the full rows [0, n) are still
  // in the caches) unless it is short, already parked or larger than the budget; the parked conversation with
  // prompt `keep` (if given) stays. Never fails the request: a failed allocation disables parking.
  void park_live(const std::vector<std::int32_t>* keep = nullptr) {
    const std::size_t n = checkpoint_ids_.size();
    if (park_budget_ == 0 || n < kMinParked) return;
    for (Parked& p : parked_)
      if (p.ids == checkpoint_ids_ && p.sampled_head == checkpoint_sampled_head_) {
        p.used = ++clock_;
        return;
      }
    auto kept = [&](const Parked& p) { return keep && p.ids == *keep; };
    // a parked prefix of this conversation is superseded by it
    std::erase_if(parked_, [&](const Parked& p) {
      return !kept(p) && p.ids.size() <= n && std::equal(p.ids.begin(), p.ids.end(), checkpoint_ids_.begin());
    });
    const ParkLayout l = park_layout(int(n));
    std::size_t used = 0, kept_bytes = 0;
    for (const Parked& p : parked_) used += p.data.bytes, kept_bytes += kept(p) ? p.data.bytes : 0;
    if (kept_bytes + l.bytes > park_budget_) return;
    while (used + l.bytes > park_budget_) {  // least recently used first
      auto lru = parked_.end();
      for (auto it = parked_.begin(); it != parked_.end(); ++it)
        if (!kept(*it) && (lru == parked_.end() || it->used < lru->used)) lru = it;
      used -= lru->data.bytes;
      parked_.erase(lru);
    }
    Parked p;
    try {
      parking_ = true;  // a failure here must not release the parked conversations through the hook
      p.data = DeviceBuffer(l.bytes);
      parking_ = false;
    } catch (const std::exception& e) {
      parking_ = false;
      std::fprintf(stderr, "conversation parking disabled: %s\n", e.what());
      cudaGetLastError();  // the failed allocation is not a sticky error
      park_budget_ = 0;
      std::erase_if(parked_, [&](const Parked& q) { return !kept(q); });
      return;
    }
    char* d = static_cast<char*>(p.data.p);
    const char* c = static_cast<const char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(d, c, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    for (int i = 0; i < 3 && drafter_; ++i)
      CUDA_CHECK(cudaMemcpyAsync(d + l.drafter_off[i], c + checkpoint_drafter_ + l.drafter[i].offset,
                                 l.drafter[i].bytes, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d + l.logits, c + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.save_full_rows(d + l.rows, int(n));
    p.ids = checkpoint_ids_;
    p.sampled_head = checkpoint_sampled_head_;
    if (drafter_) p.context = drafter_->checkpoint_context();
    p.used = ++clock_;
    parked_.push_back(std::move(p));
  }

  // Puts a parked conversation back as the prompt checkpoint (restore_checkpoint() then restores it) and its
  // full rows into the caches. It stays parked.
  void unpark(Parked& p) {
    const int n = int(p.ids.size());
    const ParkLayout l = park_layout(n);
    const char* d = static_cast<const char*>(p.data.p);
    char* c = static_cast<char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(c, d, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    for (int i = 0; i < 3 && drafter_; ++i)
      CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_drafter_ + l.drafter[i].offset, d + l.drafter_off[i],
                                 l.drafter[i].bytes, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_logits_, d + l.logits, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.restore_full_rows(d + l.rows, n);
    if (drafter_) drafter_->adopt_checkpoint(c + checkpoint_drafter_, n, p.context);
    checkpoint_ids_ = p.ids;
    checkpoint_sampled_head_ = p.sampled_head;
    p.used = ++clock_;
  }

  // The checkpoint (on the device): target state (Runtime::save_checkpoint) | drafter context
  // (Drafter::save_checkpoint) | logits of the last prompt token.
  void save_checkpoint() {
    char* p = static_cast<char*>(checkpoint_.p);
    rt_.save_checkpoint(p);
    if (drafter_) drafter_->save_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    checkpoint_ids_ = tokens_in_state_;
    checkpoint_sampled_head_ = rt_.tp() && rt_.tp()->full_logits;
  }

  void restore_checkpoint() {
    const char* p = static_cast<const char*>(checkpoint_.p);
    rt_.restore_checkpoint(p);
    if (drafter_) drafter_->restore_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(logits_.p, p + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    tokens_in_state_ = checkpoint_ids_;
  }

  static std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

  // engine._sample with a host synchronisation (the speculative loop's first anchor): argmax at
  // temperature <= 0, else one draw from engine._probs.
  std::int32_t sample(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    head::SamplingParams p;
    p.temperature = sampling.temperature;
    p.top_k = sampling.top_k;
    p.top_p = sampling.top_p;
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_.p),
                 rt_.stream());
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               rt_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  const std::vector<int32_t>& slot_history(int slot) const {
    return slot ? batch_histories_.at(slot - 1) : tokens_in_state_;
  }

  std::vector<int32_t>& slot_history(int slot) {
    return slot ? batch_histories_.at(slot - 1) : tokens_in_state_;
  }

  int32_t batch_token(const void* output, bool greedy_head, BatchSampling& sampling) {
    if (greedy_head) {
      int32_t token;
      CUDA_CHECK(cudaMemcpy(&token, output, sizeof(token), cudaMemcpyDeviceToHost));
      return token;
    }
    head::SamplerRng rng{sampling.seed, sampling.counter};
    const int32_t token = sample(static_cast<const float*>(output), sampling.params, rng);
    sampling.counter = rng.counter;
    return token;
  }

  std::array<std::vector<int32_t>, 7> batch_histories_;
  RuntimeOptions options_;
  std::unique_ptr<Runtime> runtime_;
  Runtime& rt_;
  Prefiller prefiller_;
  std::vector<DeviceBuffer> draft_head_storage_;
  std::vector<DeviceBuffer> dflash2_head_storage_;
  std::unique_ptr<SpecLoop> dflash2_spec_;
  bool previous_flash_ = false;
  std::unique_ptr<Drafter> drafter_;
  std::unique_ptr<DFlash2Head> dflash2_head_;
  std::unique_ptr<DFlash2> dflash2_;
  std::unique_ptr<SpecLoop> spec_;
  std::unique_ptr<VisionEncoder> vision_;  // image input (RuntimeOptions::vision)
  DeviceBuffer chunk_tokens_, logits_, token_, taps_;
  PinnedHostBuffer host_token_;
  // prompt checkpoint (save_checkpoint) and the prompt it holds (empty: none)
  DeviceBuffer checkpoint_;
  std::size_t checkpoint_drafter_ = 0, checkpoint_logits_ = 0;  // byte offsets in checkpoint_
  std::vector<std::int32_t> checkpoint_ids_;
  std::vector<std::int32_t> tokens_in_state_;
  bool checkpoint_sampled_head_ = false;
  std::vector<Parked> parked_;
  std::size_t park_budget_ = 0;  // bytes
  std::uint64_t clock_ = 0;      // last use of a parked conversation
  bool parking_ = false;         // park_live is allocating
  int release_hook_ = -1;
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options) {
  return std::make_unique<EngineGenerator>(options);
}

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  return std::make_unique<EngineGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model
