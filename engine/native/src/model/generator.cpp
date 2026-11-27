// The resident model supports full turns and step-level continuous batching.
// In a full turn, Prefiller advances the prompt, then plain decode pipelines
// width-one steps or SpecLoop verifies draft proposals.
// Batched serving keeps a committed history for each slot and advances rows
// whose next input token is ready. The solo path remains available on slot 0
// and can hand its state back to the scheduler at a safe boundary.
#include "ingine/generator.h"

#include "core/arena.h"
#include "core/device.h"
#include "kernels/batch/batch.h"
#include "kernels/head/head.h"
#include "model/batch.h"
#include "model/batch_dflash2.h"
#include "model/batch_drafter.h"
#include "model/batch_engine.h"
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

std::uint64_t batch_seed(std::uint64_t draw) { return draw * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull; }

const RuntimeOptions& without_spec(const RuntimeOptions& options) {
  if (options.spec) throw std::invalid_argument("make_generator over given weights: spec needs the model directory");
  return options;
}

// Copy committed rows between cache views that may have different context capacities.
void copy_full_prefix(const State& src, const State& dst, int len, cudaStream_t stream) {
  if (len == 0) return;
  auto copy = [&](const Tensor& from, const Tensor& to) {
    const std::size_t bytes = std::size_t(len) * from.nb[2];
    for (int visit = 0; visit < src.loopcd.full_visits(); ++visit) {
      const auto* src_rows = ptr<const char>(from) + std::size_t(visit) * from.nb[3];
      auto* dst_rows = ptr<char>(to) + std::size_t(visit) * to.nb[3];
      CUDA_CHECK(cudaMemcpyAsync(dst_rows, src_rows, bytes, cudaMemcpyDeviceToDevice, stream));
    }
  };
  copy(src.k_full, dst.k_full);
  copy(src.v_full, dst.v_full);
  if (src.kv_q4 || src.kv_12) {
    copy(src.k_full_scale, dst.k_full_scale);
    copy(src.v_full_scale, dst.v_full_scale);
  }
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
        prefiller_(rt_.weights(), rt_.ablation(), options.prefill_chunk, rt_.stream(), rt_.tp(), rt_.state().loopcd),
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
    const bool shard_dflash2 = use_dflash2 && rt_.tp() && dflash2_shard_backbone();
    if (use_dflash2 && (options.tp.rank == 0 || shard_dflash2)) {
      auto dir = options.dflash2_dir;
      if (dir.empty())
        if (const char* env = std::getenv("INKLING_DFLASH2_DIR")) dir = env;
      if (dir.empty()) throw std::invalid_argument("DFlash2 requires a checkpoint directory");
      dflash2_ = std::make_unique<DFlash2>(dir, rt_.stream(), options.draft_type == "dflash2" && options.draft_nvfp4,
                                           shard_dflash2 ? rt_.tp() : nullptr);
      const auto* embed = ptr<const __nv_bfloat16>(rt_.weights().embed);
      const auto* embed_norm = ptr<const float>(rt_.weights().embed_norm);
      if (dflash2_head_) {
        dflash2_->bind_sharded_head(*dflash2_head_, embed, embed_norm);
      } else if (options.tp.rank != 0) {
        dflash2_->bind_backbone_only(embed, embed_norm);
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
        head = load_head(options.model_dir, options.dense_fp8, false, rt_.stream(), draft_head_storage_,
                         TpLayout{}, options.dense_pico12).dense;
      drafter_ = std::make_unique<Drafter>(options.model_dir, rt_.weights(), options.max_ctx, rt_.stream(),
                                           Drafter::SHORTLIST_ROWS, options.draft_nvfp4, head.w.data ? &head : nullptr,
                                           rt_.tp() && rt_.tp()->profile);
      drafter_->capture();
    }
    if (drafter_ || dflash2_ || options.batch_spec > 0)
      taps_ = DeviceBuffer(std::size_t(options.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
    if (options.batch_spec > 0) {
      if (options.batch_spec > BatchDrafter::GAMMA)
        throw std::invalid_argument("batch_spec must be in 1..7");
      if (options.batch_ctx < WINDOW || options.batch_ctx > options.max_ctx)
        throw std::invalid_argument("batch_ctx must be in [1024, max_ctx]");
      const TpLayout layout = rt_.tp() ? rt_.tp()->layout : TpLayout{};
      const cudaStream_t stream = rt_.stream();
      batch_spec_config_ = batch_spec_config(rt_.tp(), stream);
      if (!rt_.weights().unembed_nvfp4.codes && (!rt_.weights().unembed.w.data || rt_.weights().unembed.fp8())) {
        batch_unembed_ = load_head(options.model_dir, /*dense_fp8=*/false, /*head_nvfp4=*/false, stream,
                                   batch_head_storage_, layout).dense;
      }
      batch_state_ = make_batch_state(options.max_batch, options.batch_ctx, options.kv_fp8, layout,
                                     options.kv_q4, options.kv_12, rt_.state().loopcd);
      batch_dec_ = std::make_unique<BatchDecoder>(rt_.weights(), rt_.ablation(), batch_state_, options.max_batch,
                                                  stream, 8, rt_.tp(),
                                                  batch_unembed_.w.data ? &batch_unembed_ : nullptr);
      batch_draft_type_ = options.batch_draft_type;
      if (batch_draft_type_.empty()) {
        if (const char* env = std::getenv("INGINE_BATCH_DRAFT_TYPE"); env && *env) batch_draft_type_ = env;
        else if (!options.dflash2_dir.empty() && options.batch_spec >= 5)
          batch_draft_type_ = (options.draft_type == "hybrid") ? "hybrid" : "dflash2";
        else batch_draft_type_ = "dspark";
      }
      if (batch_draft_type_ != "dspark" && batch_draft_type_ != "dflash2" && batch_draft_type_ != "hybrid")
        throw std::invalid_argument("batch_draft_type must be dspark, dflash2, or hybrid");
      const bool hybrid = batch_draft_type_ == "hybrid";
      if (hybrid) batch_dflash2_max_ctx_ = batch_dflash2_max_ctx(options.dflash2_max_ctx, rt_.tp(), stream);
      const bool use_batch_dspark = batch_draft_type_ == "dspark" || hybrid;
      const bool use_batch_dflash2 = batch_draft_type_ == "dflash2" || (hybrid && batch_dflash2_max_ctx_ > 0);
      if (use_batch_dspark) {
        const bool shard_batch_drafter = [] {
          const char* e = std::getenv("INGINE_SHARD_BATCH_DRAFTER");
          return !e || !*e || std::atoi(e) != 0;
        }();
        if (!rt_.tp() || shard_batch_drafter || layout.tp_rank == 0) {
          std::vector<DeviceBuffer> tmp_head;
          Dense full_head;
          if (rt_.tp() || !rt_.weights().unembed.w.data || rt_.weights().unembed.fp8())
            full_head = load_head(options.model_dir, false, false, stream, tmp_head).dense;
          batch_drafter_ = std::make_unique<BatchDrafter>(
              options.model_dir, rt_.weights(), options.max_batch, options.batch_ctx, options.max_batch, stream,
              BatchDrafter::Precision::MXFP8, full_head.w.data ? &full_head : nullptr,
              shard_batch_drafter ? rt_.tp() : nullptr);
        }
      }
      if (use_batch_dflash2) {
        if (rt_.tp() && options.head_nvfp4 && !dflash2_full_head()) {
          batch_dflash2_head_ = std::make_unique<BatchDFlash2Head>(
              *rt_.tp(), rt_.weights().unembed_nvfp4, options.max_batch, stream);
        }
        if (!rt_.tp() || layout.tp_rank == 0) {
          auto dir = options.dflash2_dir;
          if (dir.empty())
            if (const char* env = std::getenv("INKLING_DFLASH2_DIR")) dir = env;
          if (dir.empty()) {
            const auto sibling = options.model_dir.parent_path() / "Inkling-2-Mini-DFlash2";
            if (std::filesystem::exists(sibling / "model.safetensors")) dir = sibling;
          }
          if (dir.empty()) throw std::invalid_argument("BatchDFlash2 requires --dflash2-dir or INKLING_DFLASH2_DIR");
          const kernels::head::Nvfp4Matrix* head_ovr = nullptr;
          kernels::head::Nvfp4Matrix loaded_head{};
          if (!batch_dflash2_head_ && (rt_.tp() || !rt_.weights().unembed_nvfp4.codes)) {
            loaded_head = load_head(options.model_dir, false, true, stream, batch_dflash2_head_storage_).nvfp4;
            head_ovr = &loaded_head;
          }
          batch_dflash2_ = std::make_unique<BatchDFlash2>(
              dir, rt_.weights(), options.max_batch, options.max_batch, stream,
              BatchDFlash2::default_precision(), BatchDFlash2::default_kv_fp8(),
              head_ovr, batch_dflash2_head_.get());
        }
      }
      if (use_batch_dflash2 && batch_dec_->reject_enabled())
        batch_draft_support_ = BatchDraftSupport(options.max_batch, rt_.tp());
      constexpr int G = BatchDrafter::GAMMA;
      constexpr int TAP_ROW = NUM_TAPS * HIDDEN;
      batch_gathered_ = DeviceBuffer(std::size_t(options.max_batch) * 8 * TAP_ROW * 2);
      batch_rows_ = DeviceBuffer(std::size_t(options.max_batch) * 8 * 4);
      if (rt_.tp()) batch_draft_bcast_ = DeviceBuffer(std::size_t(options.max_batch) * G * 8);
      batch_drafts_.assign(std::size_t(options.max_batch) * G, 0);
      batch_conf_.assign(std::size_t(options.max_batch) * G, 0.f);
      batch_spec_histories_.resize(std::size_t(options.max_batch));
      batch_slot_ema_p_.assign(std::size_t(options.max_batch), BatchSpecCost::initial_p);
      batch_slot_spec_steps_.assign(std::size_t(options.max_batch), 0);
      batch_slot_draft_budget_.assign(std::size_t(options.max_batch), options.batch_spec);
    }
    if (options.spec && options.draft_type != "dflash2")
      spec_ = std::make_unique<SpecLoop>(rt_, drafter_.get());
    if (use_dflash2)
      dflash2_spec_ = std::make_unique<SpecLoop>(rt_, nullptr, dflash2_.get(), dflash2_head_.get());
    // checkpoint layout: target state | drafter context | last prompt logits
    checkpoint_drafter_ = carved(rt_.checkpoint_bytes());
    checkpoint_logits_ = checkpoint_drafter_ + (drafter_ ? carved(drafter_->checkpoint_bytes()) : 0);
    checkpoint_ = DeviceBuffer(checkpoint_logits_ + logits_.bytes);
    if (dflash2_) dflash2_checkpoint_ = DeviceBuffer(dflash2_->checkpoint_bytes());
    // the vision tower's weights are resident: loaded before the park budget below is measured
    if (options.vision) vision_ = std::make_unique<VisionEncoder>(options.model_dir, rt_.stream());
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    if (dflash2_head_ || (dflash2_ && dflash2_->is_sharded())) rt_.tp()->check();
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
    const bool allow_dflash2_sampled = [] {
      const char* e = std::getenv("INGINE_DFLASH2_SAMPLED");
      return !e || !*e || std::atoi(e) != 0;
    }();
    if (options_.draft_type == "dflash2" && sampling.temperature > 0 && sampling.forced.empty() && !allow_dflash2_sampled)
      throw std::invalid_argument("DFlash2 sampled generation disabled by INGINE_DFLASH2_SAMPLED=0");
    if (rt_.tp() && !options_.tp.serve && sampling.temperature != 0)
      throw std::invalid_argument("TP=2 generator requires temperature 0");
    const int n = int(prompt.size());
    const int image_tokens = check_images(full_prompt);
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::lock_guard<std::mutex> lock(mutex_);

    if (rt_.tp() && options_.tp.serve) rt_.tp()->full_logits = sampling.temperature > 0;

    bool flash = options_.draft_type == "dflash2";
    if (options_.draft_type == "hybrid") {
      if (n <= options_.dflash2_max_ctx && (sampling.temperature <= 0.f || allow_dflash2_sampled)) {
        const bool warm_flash = !options_.cold_prefill &&
            (live_prefix(prompt, true) || checkpoint_prefix(prompt, true) || find_parked(prompt, 0, true));
        // Favor a substantial warm DSpark prefix when DFlash2 would need a full prefill.
        constexpr std::size_t kMinHybridReuse = 512;
        const bool warm_dspark = !options_.cold_prefill &&
            (live_prefix(prompt, false) >= kMinHybridReuse ||
             checkpoint_prefix(prompt, false) >= kMinHybridReuse ||
             find_parked(prompt, kMinHybridReuse - 1, false));
        flash = warm_flash || !warm_dspark;
      }
      if (rt_.tp() && !rt_.tp()->agree(uint64_t(flash), rt_.stream())) flash = false;
    }
    auto* spec = flash ? dflash2_spec_.get() : spec_.get();
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
      if (flash != previous_flash_) {
        if (!options_.cold_prefill) park_live();
        tokens_in_state_.clear();
        clear_checkpoint();
        if (!flash && dflash2_) dflash2_->reset();
      }
      prefill(prompt, full_prompt.images, stop_ids, stats, request, flash, counter ? &on_tokens : nullptr);
      previous_flash_ = flash;
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
  int batch_max_ctx() const override { return options_.batch_spec > 0 ? options_.batch_ctx : options_.max_ctx; }
  int batch_spec() const override { return options_.batch_spec; }

  int reusable_prefix(int slot, std::span<const int32_t> prompt) const override {
    if (options_.cold_prefill) return 0;
    auto match = [&](const std::vector<int32_t>& h) {
      return !h.empty() && h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin()) ? int(h.size()) : 0;
    };
    if (options_.batch_spec > 0) {
      const int bm = match(batch_spec_histories_.at(std::size_t(slot)));
      return slot == 0 ? std::max(bm, match(tokens_in_state_)) : bm;
    }
    // Precompute the final token's head even when a previous turn used a different head mode.
    return match(slot_history(slot));
  }

  bool promote_solo_to_batch(int slot, int32_t pending_token) override {
    if (options_.batch_spec == 0) return true;
    const int len = int(tokens_in_state_.size());
    if (slot < 0 || slot >= options_.max_batch) return false;
    if (len + options_.batch_spec + 2 >= options_.batch_ctx) return false;
    const cudaStream_t stream = rt_.stream();
    const State& src = rt_.state();
    State dst = batch_state_.view(slot);
    auto copy_same = [&](const Tensor& s_t, const Tensor& d_t) {
      CUDA_CHECK(cudaMemcpyAsync(d_t.data, s_t.data, s_t.bytes(), cudaMemcpyDeviceToDevice, stream));
    };
    copy_same(src.k_ring, dst.k_ring);
    copy_same(src.v_ring, dst.v_ring);
    copy_same(src.ring_pos, dst.ring_pos);
    copy_same(src.conv_k, dst.conv_k);
    copy_same(src.conv_v, dst.conv_v);
    copy_same(src.conv_attn, dst.conv_attn);
    copy_same(src.conv_mlp, dst.conv_mlp);
    copy_full_prefix(src, dst, len, stream);
    batch_spec_histories_.at(std::size_t(slot)) = tokens_in_state_;
    reset_batch_spec_cost(slot);
    if (batch_drafter_) batch_drafter_->reset_slot(slot);
    if (batch_dflash2_) batch_dflash2_->reset_slot(slot);
    if (pending_token >= 0) {
      const std::int32_t slot32 = slot;
      const std::int64_t pos64 = len;
      batch_draft_slots(1, &slot32, &pending_token, &pos64);
    } else {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return true;
  }

  bool clone_slot(int src_slot, int dst_slot) override {
    if (options_.batch_spec == 0) return false;
    if (src_slot < 0 || src_slot >= options_.max_batch || dst_slot < 0 || dst_slot >= options_.max_batch)
      return false;
    if (src_slot == dst_slot) return true;
    const auto& src_hist = batch_spec_histories_.at(std::size_t(src_slot));
    const int len = int(src_hist.size());
    if (len <= 0) return false;
    const cudaStream_t stream = rt_.stream();
    const State src = batch_state_.view(src_slot);
    State dst = batch_state_.view(dst_slot);
    auto copy_same = [&](const Tensor& s_t, const Tensor& d_t) {
      CUDA_CHECK(cudaMemcpyAsync(d_t.data, s_t.data, s_t.bytes(), cudaMemcpyDeviceToDevice, stream));
    };
    copy_same(src.k_ring, dst.k_ring);
    copy_same(src.v_ring, dst.v_ring);
    copy_same(src.ring_pos, dst.ring_pos);
    copy_same(src.conv_k, dst.conv_k);
    copy_same(src.conv_v, dst.conv_v);
    copy_same(src.conv_attn, dst.conv_attn);
    copy_same(src.conv_mlp, dst.conv_mlp);
    copy_full_prefix(src, dst, len, stream);
    batch_spec_histories_.at(std::size_t(dst_slot)) = src_hist;
    // The cloned prefix has no pending proposal.
    reset_batch_spec_cost(dst_slot);
    if (batch_drafter_) batch_drafter_->clone_slot(src_slot, dst_slot, len);
    if (batch_dflash2_) batch_dflash2_->clone_slot(src_slot, dst_slot, len);
    return true;
  }

  PrefillChunkResult prefill_slot_chunk(int slot, std::span<const int32_t> prompt, int offset,
                                       BatchSampling& sampling, int stop_at = 0) override {
    const int ctx_lim = options_.batch_spec > 0 ? options_.batch_ctx : options_.max_ctx;
    if (prompt.empty() || int(prompt.size()) + options_.batch_spec + 1 >= ctx_lim ||
        std::any_of(prompt.begin(), prompt.end(), [](int32_t token) {
          return token < 0 || token >= UNPADDED_VOCAB;
        }))
      throw std::invalid_argument("invalid batch prompt");
    const auto stream = rt_.stream();
    auto* tp = rt_.tp();
    if (options_.batch_spec > 0) {
      State view = batch_state_.view(slot);
      auto& history = batch_spec_histories_.at(std::size_t(slot));
      const int reused = offset == 0 ? prepare_batch_prefill(slot, prompt) : 0;
      offset += reused;
      if (stop_at > 0 && offset == stop_at && stop_at < int(prompt.size())) {
        return PrefillChunkResult{offset, reused, std::nullopt};
      }
      const int target_end = (stop_at > offset && stop_at < int(prompt.size())) ? stop_at : int(prompt.size());
      const int count = std::min({2048, options_.prefill_chunk, target_end - offset});
      const bool final = offset + count == int(prompt.size());
      const bool sampled = sampling.params.temperature > 0;
      if (tp) tp->full_logits = sampled;
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, prompt.data() + offset, std::size_t(count) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
      const bool need_batch_taps = batch_drafter_ || batch_dflash2_;
      Tensor taps;
      if (need_batch_taps) taps = rm(taps_.p, DType::BF16, {count, NUM_TAPS, HIDDEN});
      prefiller_.run(view, static_cast<const int32_t*>(chunk_tokens_.p), count, offset,
                     !tp || (sampled && options_.tp.rank == 0) ? static_cast<float*>(logits_.p) : nullptr,
                     need_batch_taps ? &taps : nullptr, {},
                     tp && !sampled ? static_cast<int32_t*>(logits_.p) : nullptr, final);
      if (need_batch_taps) {
        const std::int32_t slot32 = slot, count32 = count;
        const std::int64_t start64 = offset;
        const auto* tp_ptr = static_cast<const __nv_bfloat16*>(taps_.p);
        if (batch_drafter_) batch_drafter_->commit(1, &slot32, &start64, &count32, tp_ptr);
        if (batch_dflash2_) batch_dflash2_->commit(1, &slot32, &start64, &count32, tp_ptr);
      }
      if (tp) tp->wait(stream);
      else CUDA_CHECK(cudaStreamSynchronize(stream));
      history.insert(history.end(), prompt.begin() + offset, prompt.begin() + offset + count);
      PrefillChunkResult result{offset + count, reused, std::nullopt};
      if (final) {
        std::int32_t first_tok;
        if (tp && sampled) {
          head::SamplerRng rng{sampling.seed, sampling.counter};
          first_tok = sample_tp(logits_.p, sampling.params, rng);
          sampling.counter = rng.counter;
        } else {
          first_tok = batch_token(logits_.p, tp != nullptr, sampling);
        }
        if (options_.tp.rank == 0) result.token = first_tok;
        const std::int32_t slot32 = slot;
        const std::int64_t pos64 = std::int64_t(prompt.size());
        batch_draft_support_.set_temperature(slot, sampling.params.temperature);
        batch_draft_slots(1, &slot32, &first_tok, &pos64);
      }
      return result;
    }

    auto& state = rt_.state_slot(slot);
    auto& history = slot_history(slot);
    int reused = 0;
    if (offset == 0) {
      reused = reusable_prefix(slot, prompt);
      if (tp && !tp->agree(reused, stream)) reused = 0;
      if (slot == 0) {
        park_live();
        clear_checkpoint();
        if (previous_flash_ && dflash2_) dflash2_->reset();
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

  bool supports_batch_short_prefill() const override {
    if (options_.batch_spec <= 0 || !batch_dec_) return false;
    const char* enabled = std::getenv("INGINE_BATCH_SHORT_PREFILL");
    // Default ON for TP=1; opt-in (=1) for TP=2 because 8-tok BatchDecoder steps multiply 96 RoCE
    // AllReduces per step and raise c=24 TTFT (+0.29..0.55 s vs chunked Prefiller).
    if (rt_.tp()) return enabled && std::string_view(enabled) == "1";
    return !enabled || std::string_view(enabled) != "0";
  }

  std::vector<PrefillChunkResult> prefill_batch_short(std::span<const BatchPrefillSlot> slots,
                                                     int max_steps = 0) override {
    if (!supports_batch_short_prefill()) throw std::logic_error("short batch prefill unavailable");
    if (max_steps < 0 || slots.size() > batch_spec_histories_.size())
      throw std::invalid_argument("invalid short batch prefill dimensions");
    std::vector<bool> seen(batch_spec_histories_.size());
    for (const auto& input : slots) {
      if (input.slot < 0 || std::size_t(input.slot) >= seen.size() || seen[input.slot] ||
          !input.sampling || input.offset < 0 || std::size_t(input.offset) >= input.prompt.size() ||
          input.prompt.size() > std::size_t(options_.batch_ctx - 8) ||
          input.sampling->params.top_k != 0 || !(input.sampling->params.top_p >= 1.0f) ||
          std::any_of(input.prompt.begin(), input.prompt.end(), [](int32_t token) {
            return token < 0 || token >= UNPADDED_VOCAB;
          }))
        throw std::invalid_argument("invalid short batch prompt");
      if (input.offset > 0 && batch_spec_histories_[input.slot].size() != std::size_t(input.offset))
        throw std::invalid_argument("short batch prefill offset does not match slot history");
      seen[input.slot] = true;
    }
    std::vector<PrefillChunkResult> results;
    results.reserve(slots.size());
    std::vector<std::size_t> members;
    std::vector<std::int32_t> m, tok, slot_ids, off, finished_slots, anchors;
    std::vector<std::int64_t> pos, finished_pos;
    std::vector<float> temp;
    try {
      for (const auto& input : slots) {
        const int reused = input.offset == 0 ? prepare_batch_prefill(input.slot, input.prompt) : 0;
        results.push_back({input.offset + reused, reused, std::nullopt});
      }
      for (int step = 0; max_steps == 0 || step < max_steps; ++step) {
        members.clear();
        m.clear();
        tok.clear();
        slot_ids.clear();
        off.assign(1, 0);
        pos.clear();
        temp.clear();
        for (std::size_t j = 0; j < slots.size(); ++j) {
          const auto& input = slots[j];
          const int offset = results[j].offset;
          const int count = std::min(8, int(input.prompt.size()) - offset);
          if (count == 0) continue;
          members.push_back(j);
          m.push_back(count);
          tok.insert(tok.end(), input.prompt.begin() + offset, input.prompt.begin() + offset + count);
          slot_ids.push_back(input.slot);
          off.push_back(int(tok.size()));
          pos.push_back(offset);
          temp.push_back(input.sampling->params.temperature);
        }
        if (members.empty()) break;
        const auto got = batch_dec_->step(int(members.size()), m.data(), tok.data(), slot_ids.data(), pos.data(),
                                          temp.data(), batch_seed(++batch_draws_), nullptr);
        // A multi-row step defers conv updates for every lane, including one-row lanes.
        batch_dec_->accept(m.data());
        commit_batch_taps(slot_ids, pos, m, off);
        for (std::size_t i = 0; i < members.size(); ++i) {
          const std::size_t j = members[i];
          auto& history = batch_spec_histories_[slot_ids[i]];
          history.insert(history.end(), tok.begin() + off[i], tok.begin() + off[i + 1]);
          auto& result = results[j];
          result.offset += m[i];
          if (result.offset == int(slots[j].prompt.size())) {
            result.token = got[off[i + 1] - 1];
            batch_draft_support_.set_temperature(slot_ids[i], temp[i]);
            finished_slots.push_back(slot_ids[i]);
            anchors.push_back(*result.token);
            finished_pos.push_back(result.offset);
          }
        }
      }
      if (!finished_slots.empty())
        batch_draft_slots(int(finished_slots.size()), finished_slots.data(), anchors.data(), finished_pos.data());
      else if (rt_.tp())
        rt_.tp()->wait(rt_.stream());
      else
        CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    } catch (...) {
      // A failed step may have written part of a slot's state. Force a cold restart on reuse.
      for (const auto& input : slots) batch_spec_histories_[input.slot].clear();
      throw;
    }
    return results;
  }

  std::vector<std::vector<int32_t>> decode_batch_step_multi(std::span<const BatchSlotStep> slots) override {
    if (options_.batch_spec == 0) return Generator::decode_batch_step_multi(slots);
    const int n = int(slots.size());
    const std::size_t ns = std::size_t(n);
    std::vector<std::vector<int32_t>> emitted(ns);
    if (n == 0) return emitted;
    constexpr int G = BatchDrafter::GAMMA;
    const BatchSpecCost cost(batch_spec_config_, n, options_.max_batch, options_.batch_spec_threshold);
    std::vector<std::int32_t> m(ns), slot_ids(ns), tok;
    std::vector<std::int64_t> pos(ns);
    std::vector<float> temp(ns);
    std::vector<BatchSpecCost::Slot> spec_slots(ns);
    for (std::size_t i = 0; i < ns; ++i) {
      const int s = slots[i].slot;
      const int p_cur = int(batch_spec_histories_.at(std::size_t(s)).size());
      const int avail = std::max(0, options_.batch_ctx - 1 - p_cur);
      const int k_max = std::min(batch_slot_draft_budget_[s], std::max(0, avail - 1));
      slot_ids[i] = s;
      pos[i] = p_cur;
      temp[i] = slots[i].sampling->params.temperature;
      spec_slots[i] = {k_max, batch_conf_.data() + std::size_t(s) * G,
                      batch_slot_ema_p_[s], batch_slot_spec_steps_[s], temp[i],
                      batch_draft_support_.has_support(s, temp[i])};
    }
    cost.allocate_widths(spec_slots, options_.batch_spec_min, m);
    tok.reserve(ns * std::size_t(options_.batch_spec + 1));
    for (std::size_t i = 0; i < ns; ++i) {
      const int s = slot_ids[i];
      const int k = m[i];
      m[i] = k + 1;
      tok.push_back(slots[i].token);
      for (int d = 0; d < k; ++d) tok.push_back(batch_drafts_[std::size_t(s) * G + std::size_t(d)]);
    }
    const auto* reject = batch_draft_support_.verification_inputs(slot_ids, m, tok, temp);
    const auto got = batch_dec_->step(n, m.data(), tok.data(), slot_ids.data(), pos.data(), temp.data(),
                                      batch_seed(++batch_draws_), reject);
    std::vector<std::int32_t> off(ns + 1, 0), keep(ns);
    for (int i = 0; i < n; ++i) off[std::size_t(i) + 1] = off[std::size_t(i)] + m[std::size_t(i)];
    std::vector<std::int32_t> d_slots, d_anchors;
    std::vector<std::int64_t> d_pos;
    d_slots.reserve(std::size_t(n)), d_anchors.reserve(std::size_t(n)), d_pos.reserve(std::size_t(n));
    for (int i = 0; i < n; ++i) {
      const int K = m[std::size_t(i)] - 1;
      const std::int32_t* row = tok.data() + off[std::size_t(i)];
      const std::int32_t* out = got.data() + off[std::size_t(i)];
      int a = 0;
      while (a < K && row[1 + a] == out[a]) ++a;
      const int slot = slot_ids[std::size_t(i)];
      BatchSpecCost::observe(K, a, batch_slot_ema_p_[slot], batch_slot_spec_steps_[slot]);
      batch_draft_support_.set_temperature(slot, temp[std::size_t(i)]);
      emitted[std::size_t(i)].assign(row + 1, row + 1 + a);
      emitted[std::size_t(i)].push_back(out[a]);
      keep[std::size_t(i)] = a + 1;
      auto& hist = batch_spec_histories_.at(std::size_t(slot));
      hist.insert(hist.end(), row, row + a + 1);
      const std::int64_t np = pos[std::size_t(i)] + a + 1;
      if (np + 1 < options_.batch_ctx) {
        d_slots.push_back(slot);
        d_anchors.push_back(out[a]);
        d_pos.push_back(np);
      }
    }
    batch_dec_->accept(keep.data());
    commit_batch_taps(slot_ids, pos, keep, off);
    if (!d_slots.empty()) batch_draft_slots(int(d_slots.size()), d_slots.data(), d_anchors.data(), d_pos.data());
    else if (rt_.tp()) rt_.tp()->wait(rt_.stream());
    else CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return emitted;
  }

  std::vector<int32_t> decode_batch_step(std::span<const BatchSlotStep> slots) override {
    std::vector<Runtime::BatchInput> rows;
    bool sampled = false, greedy_only = true;
    for (const auto& slot : slots) {
      rt_.state_slot(slot.slot).length = int(slot_history(slot.slot).size());
      rows.push_back({slot.slot, slot.token});
      sampled |= slot.sampling->params.temperature > 0;
      greedy_only &= slot.sampling->params.temperature <= 0;
    }
    if (rows.size() == 1 && rows.front().slot == 0) {
      if (rt_.tp()) rt_.tp()->full_logits = sampled;
      const int32_t input = rows.front().token;
      rt_.set_inputs(std::span(&input, 1), int(tokens_in_state_.size()));
      const float* logits = rt_.run_step(1, greedy_only);
      tokens_in_state_.push_back(input);
      if (rt_.tp()) rt_.tp()->wait(rt_.stream());
      if (options_.tp.rank != 0) return {};
      const bool greedy = rt_.tp() ? !sampled : rt_.has_cert_head() && greedy_only;
      const void* output = greedy ? rt_.buffers(1).greedy.data : logits;
      return {batch_token(output, greedy, *slots.front().sampling)};
    }
    auto& buffers = rt_.run_batch(rows, sampled, greedy_only);
    std::vector<int32_t> tokens;
    const bool greedy_head = rt_.tp() ? !sampled : rt_.has_cert_head() && greedy_only;
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

  void clear_checkpoint() {
    checkpoint_ids_.clear();
    checkpoint_is_flash_ = false;
    checkpoint_has_logits_ = false;
  }

  // After a failure anywhere in a request the device state may be ahead of or behind tokens_in_state_ (a step
  // in flight, a partial prefill, a round between launch and bookkeeping): wait for the stream and forget
  // everything reusable, so the next request starts from an empty state.
  void forget_state() noexcept {
    cudaStreamSynchronize(rt_.stream());
    tokens_in_state_.clear();
    clear_checkpoint();
    previous_flash_ = false;
    if (dflash2_) dflash2_->reset();
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
    // forced replay: the forced tokens on the device; each sampled token is replaced by its forced one
    DeviceBuffer forced;
    if (!sampling.forced.empty()) {
      forced = DeviceBuffer(sampling.forced.size_bytes());
      CUDA_CHECK(cudaMemcpyAsync(forced.p, sampling.forced.data(), sampling.forced.size_bytes(),
                                 cudaMemcpyHostToDevice, stream));
    }
    // engine._sample of the logits into `sampled`, then the inputs of the step at `position`
    auto sample_for = [&](const float* logits, int position) {
      auto* sampled = logits ? static_cast<int32_t*>(token_.p) : ptr<int32_t>(rt_.buffers(1).greedy);
      if (logits) head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
      if (forced.p)
        CUDA_CHECK(cudaMemcpyAsync(sampled, static_cast<const std::int32_t*>(forced.p) + (position - n),
                                   sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
      rt_.chain_advance(sampled, position);
    };
    // the step at `position` (its input token is on the device), then the next token
    auto launch = [&](int position) { sample_for(rt_.run_step(1, params.temperature <= 0.f), position + 1); };
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

  std::size_t live_prefix(std::span<const std::int32_t> ids, bool flash) const {
    const auto& prev = tokens_in_state_;
    if (options_.cold_prefill || previous_flash_ != flash || prev.empty() || prev.size() >= ids.size())
      return 0;
    // Only ranks that own the active drafter have a local draft length to check.
    const int len = int(prev.size());
    if (flash ? (dflash2_ && dflash2_->length() != len) : (drafter_ && drafter_->length() != len))
      return 0;
    return std::equal(prev.begin(), prev.end(), ids.begin()) ? prev.size() : 0;
  }

  std::size_t checkpoint_prefix(std::span<const std::int32_t> ids, bool flash) const {
    const auto& saved = checkpoint_ids_;
    if (options_.cold_prefill || checkpoint_is_flash_ != flash || saved.empty() || saved.size() > ids.size())
      return 0;
    // Reusing the entire prompt also reuses its saved logits.
    const bool sampled_head = rt_.tp() && rt_.tp()->full_logits;
    if (saved.size() == ids.size() && (!checkpoint_has_logits_ || checkpoint_sampled_head_ != sampled_head))
      return 0;
    return std::equal(saved.begin(), saved.end(), ids.begin()) ? saved.size() : 0;
  }

  std::size_t find_anchor_boundary(std::span<const std::int32_t> ids, std::span<const PromptImage> images,
                                   std::span<const std::int32_t> stop_ids) const {
    if (options_.cold_prefill || park_budget_ == 0 || stop_ids.empty() || ids.size() < kMinParked + 16)
      return 0;
    for (std::size_t i = kMinParked; i + 16 <= ids.size(); ++i) {
      if (std::find(stop_ids.begin(), stop_ids.end(), ids[i]) == stop_ids.end()) continue;
      for (const PromptImage& image : images) {
        const auto pos = std::size_t(image.pos);
        if (pos < i && pos + std::size_t(image.tokens()) > i) return 0;
      }
      return i;
    }
    return 0;
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
  void prefill(std::span<const std::int32_t> ids, std::span<const PromptImage> images,
               std::span<const std::int32_t> stop_ids, GenerateStats& stats, std::uint64_t request, bool flash,
               const std::function<bool(std::span<const int32_t>)>* proceed = nullptr) {
    const bool cold = options_.cold_prefill;
    const std::size_t n = ids.size();
    State& st = rt_.state();
    const cudaStream_t stream = rt_.stream();
    const std::size_t extend = live_prefix(ids, flash);
    const std::size_t checkpoint = extend ? 0 : checkpoint_prefix(ids, flash);
    Parked* parked = cold ? nullptr : find_parked(ids, std::max(extend, checkpoint), flash);
    std::size_t start = 0;
    if (parked) {
      // parking the live conversation may move the parked list (never evicting the one to restore)
      const void* keep = parked->data.p;
      park_live(keep);
      unpark(*std::find_if(parked_.begin(), parked_.end(), [keep](const Parked& p) { return p.data.p == keep; }));
      restore_checkpoint();
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Parked;
    } else if (extend) {
      start = extend;
      stats.reuse = Reuse::Extend;
    } else if (checkpoint) {
      restore_checkpoint();
      start = checkpoint;
      stats.reuse = Reuse::Checkpoint;
    } else {
      if (!cold) park_live();
      reset_state(st, stream);
      if (flash && dflash2_) dflash2_->reset();
      if (!flash && drafter_) drafter_->reset();
      tokens_in_state_.clear();
      clear_checkpoint();
      stats.reuse = Reuse::None;
    }
    if (rt_.tp()) {
      // Equal offsets can still come from different target and draft snapshots.
      const auto reuse = (std::uint64_t(start) << 32) | std::uint32_t(stats.reuse);
      if (!rt_.tp()->agree(reuse, stream)) {
        forget_state();
        reset_state(st, stream);
        start = 0;
        stats.reuse = Reuse::None;
      }
      rt_.tp()->verify_request((request ^ start) * 1099511628211ull, stream);
    }
    std::size_t boundary = find_anchor_boundary(ids, images, stop_ids);
    if (boundary <= start || park_layout(int(boundary), flash).bytes > park_budget_) boundary = 0;
    std::size_t anchor_at = boundary;
    if (anchor_at) {
      for (Parked& p : parked_) {
        if (p.is_flash == flash && p.ids.size() == anchor_at && std::equal(p.ids.begin(), p.ids.end(), ids.begin())) {
          p.is_anchor = true;
          p.used = ++clock_;
          anchor_at = 0;
          break;
        }
      }
    }
    const bool capture_anchor = anchor_at != 0;
    // A rank that still has the anchor can split with the other rank while it rebuilds its copy.
    if (rt_.tp() && options_.park_mb > 0 && !cold && !stop_ids.empty() && n >= kMinParked + 16 &&
        !rt_.tp()->agree(anchor_at, stream))
      anchor_at = rt_.tp()->agree(boundary, stream) ? boundary : 0;

    st.length = int(start);  // decode steps advance the state without its host-side length
    const std::size_t chunk = std::size_t(options_.prefill_chunk);
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
    for (std::size_t s = start; s < n;) {
      int T = int(std::min(chunk, n - s));
      if (anchor_at > s && s + std::size_t(T) > anchor_at) T = int(anchor_at - s);
      const std::size_t end = s + std::size_t(T);
      const bool at_anchor = anchor_at == end;
      for (; next_image < images.size() && std::size_t(images[next_image].pos) < end; ++next_image) {
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
                     rt_.tp() && !rt_.tp()->full_logits ? static_cast<int32_t*>(logits_.p) : nullptr,
                     (!at_anchor && !proceed && !rt_.tp()) || end == n);
      if (!flash && drafter_) {
        drafter_->note_tokens(ids.subspan(s, std::size_t(T)));  // the prompt's tokens join the drafter's shortlist
        drafter_->commit(taps, int(s));
      }
      if (flash && dflash2_) dflash2_->commit(ptr<const __nv_bfloat16>(taps), T, int(s));
      if (at_anchor && capture_anchor) park_anchor(ids.first(anchor_at), flash);
      if (proceed && end < n && images.empty()) {
        const bool go_on = (*proceed)({});
        const bool agreed = !rt_.tp() || rt_.tp()->agree(go_on, stream);
        if (!agreed || !go_on) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
          tokens_in_state_.assign(ids.begin(), ids.begin() + end);
          if (!cold) park_live();
          clear_checkpoint();
          stats.reused_tokens = int(start);
          stats.interrupted_prefill_offset = int(end);
          return;
        }
      }
      s = end;
    }
    if (!encoded.empty()) CUDA_CHECK(cudaStreamSynchronize(stream));  // queued chunks read the rows freed here
    tokens_in_state_.assign(ids.begin(), ids.end());
    if (start < n && !cold) save_checkpoint(flash);
    stats.reused_tokens = int(start);
  }

  // Parked prompts hold target state, draft state, and the full-attention rows [0, n). An anchor saved
  // mid-prefill has no logits and can only be reused when there's a suffix to prefill.
  struct Parked {
    std::vector<std::int32_t> ids, context;
    DeviceBuffer data;
    bool is_flash = false;
    bool is_anchor = false;
    bool has_logits = true;
    bool sampled_head = false;
    std::uint64_t used = 0;
  };
  static constexpr std::size_t kMinParked = 256;  // shorter prompts are cheaper to prefill than to keep
  static constexpr std::size_t kShieldTokens = 4096;

  // The packed draft state is either DSpark's live regions or DFlash2's fixed-size ring.
  struct ParkLayout {
    std::array<Drafter::Region, 3> drafter{};
    std::size_t drafter_off[3] = {}, dflash2_off = 0, dflash2_bytes = 0, logits = 0, rows = 0, bytes = 0;
  };
  ParkLayout park_layout(int n, bool flash) const {
    ParkLayout l;
    std::size_t off = checkpoint_drafter_;
    if (flash) {
      l.dflash2_off = off;
      l.dflash2_bytes = dflash2_ ? dflash2_->checkpoint_bytes() : 0;
      off += carved(l.dflash2_bytes);
    } else if (drafter_) {
      l.drafter = drafter_->checkpoint_regions(n);
      for (int i = 0; i < 3; ++i) l.drafter_off[i] = off, off += carved(l.drafter[i].bytes);
    }
    l.logits = off;
    l.rows = off + carved(logits_.bytes);
    l.bytes = l.rows + rt_.full_rows_bytes(n);
    return l;
  }

  // the parked conversation whose prompt is the longest prefix of `ids`, if longer than `live` (and kMinParked)
  Parked* find_parked(std::span<const std::int32_t> ids, std::size_t live, bool flash) {
    Parked* best = nullptr;
    const bool sampled_head = rt_.tp() && rt_.tp()->full_logits;
    for (Parked& p : parked_)
      if (p.is_flash == flash && p.ids.size() >= kMinParked && p.ids.size() > live && p.ids.size() <= ids.size() &&
          (p.ids.size() < ids.size() || (p.has_logits && p.sampled_head == sampled_head)) &&
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

  DeviceBuffer allocate_parked(std::size_t bytes, std::size_t n, bool is_anchor, const void* keep = nullptr) {
    const bool shield = !is_anchor && n < kShieldTokens;
    auto evictable = [&](const Parked& p) { return p.data.p != keep && (!shield || !p.is_anchor); };
    std::size_t used = 0, evictable_bytes = 0;
    for (const Parked& p : parked_) {
      used += p.data.bytes;
      if (evictable(p)) evictable_bytes += p.data.bytes;
    }
    if (used - evictable_bytes + bytes > park_budget_) return {};
    while (used + bytes > park_budget_) {
      auto lru = parked_.end();
      for (auto it = parked_.begin(); it != parked_.end(); ++it) {
        if (!evictable(*it)) continue;
        if (lru == parked_.end() || (lru->is_anchor && !it->is_anchor) ||
            (lru->is_anchor == it->is_anchor && it->used < lru->used))
          lru = it;
      }
      used -= lru->data.bytes;
      parked_.erase(lru);
    }
    try {
      parking_ = true;  // a failure here must not release the parked conversations through the hook
      DeviceBuffer data(bytes);
      parking_ = false;
      return data;
    } catch (const std::exception& e) {
      parking_ = false;
      std::fprintf(stderr, "conversation parking disabled: %s\n", e.what());
      cudaGetLastError();  // the failed allocation is not a sticky error
      park_budget_ = 0;
      std::erase_if(parked_, evictable);
      return {};
    }
  }

  void pack_dspark(Parked& p, const ParkLayout& l) {
    char* d = static_cast<char*>(p.data.p);
    const char* c = static_cast<const char*>(checkpoint_.p) + checkpoint_drafter_;
    for (int i = 0; i < 3; ++i)
      CUDA_CHECK(cudaMemcpyAsync(d + l.drafter_off[i], c + l.drafter[i].offset, l.drafter[i].bytes,
                                 cudaMemcpyDeviceToDevice, rt_.stream()));
    p.context = drafter_->checkpoint_context();
  }

  // Parks the last prompt checkpoint while its full rows are still in the target cache.
  // `keep` identifies a parked buffer selected for the upcoming restore.
  void park_live(const void* keep = nullptr) {
    const std::size_t n = checkpoint_ids_.size();
    if (!checkpoint_has_logits_ || park_budget_ == 0 || n < kMinParked) return;
    const bool flash = checkpoint_is_flash_;
    for (Parked& p : parked_)
      if (p.is_flash == flash && p.has_logits && p.ids == checkpoint_ids_ &&
          p.sampled_head == checkpoint_sampled_head_) {
        p.used = ++clock_;
        return;
      }
    const ParkLayout l = park_layout(int(n), flash);
    if (l.bytes > park_budget_) return;
    // A longer checkpoint supersedes ordinary prefixes of the same draft type.
    std::erase_if(parked_, [&](const Parked& p) {
      return p.data.p != keep && !p.is_anchor && p.is_flash == flash && p.ids.size() <= n &&
             std::equal(p.ids.begin(), p.ids.end(), checkpoint_ids_.begin());
    });
    Parked p;
    p.data = allocate_parked(l.bytes, n, false, keep);
    if (!p.data.p) return;
    char* d = static_cast<char*>(p.data.p);
    const char* c = static_cast<const char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(d, c, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    if (flash) {
      if (dflash2_)
        CUDA_CHECK(cudaMemcpyAsync(d + l.dflash2_off, dflash2_checkpoint_.p, l.dflash2_bytes,
                                   cudaMemcpyDeviceToDevice, s));
    } else if (drafter_) {
      pack_dspark(p, l);
    }
    CUDA_CHECK(cudaMemcpyAsync(d + l.logits, c + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.save_full_rows(d + l.rows, int(n));
    p.ids = checkpoint_ids_;
    p.is_flash = flash;
    p.sampled_head = checkpoint_sampled_head_;
    p.used = ++clock_;
    parked_.push_back(std::move(p));
  }

  void park_anchor(std::span<const std::int32_t> ids, bool flash) {
    const int n = int(ids.size());
    const ParkLayout l = park_layout(n, flash);
    // Saving either drafter replaces its checkpoint metadata. Preserve the previous prompt first.
    if (!checkpoint_ids_.empty()) park_live();
    Parked p;
    p.data = allocate_parked(l.bytes, ids.size(), true);
    if (!p.data.p) return;
    clear_checkpoint();
    char* d = static_cast<char*>(p.data.p);
    rt_.save_checkpoint(d);
    if (flash) {
      if (dflash2_) dflash2_->save_checkpoint(d + l.dflash2_off);
    } else if (drafter_) {
      drafter_->save_checkpoint(static_cast<char*>(checkpoint_.p) + checkpoint_drafter_);
      pack_dspark(p, l);
    }
    rt_.save_full_rows(d + l.rows, n);
    p.ids.assign(ids.begin(), ids.end());
    p.is_flash = flash;
    p.is_anchor = true;
    p.has_logits = false;
    p.used = ++clock_;
    parked_.push_back(std::move(p));
  }

  // Puts a parked conversation back as the prompt checkpoint (restore_checkpoint() then restores it) and its
  // full rows into the caches. It stays parked.
  void unpark(Parked& p) {
    const int n = int(p.ids.size());
    const ParkLayout l = park_layout(n, p.is_flash);
    const char* d = static_cast<const char*>(p.data.p);
    char* c = static_cast<char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(c, d, checkpoint_drafter_, cudaMemcpyDeviceToDevice, s));
    if (p.is_flash) {
      if (dflash2_) {
        CUDA_CHECK(cudaMemcpyAsync(dflash2_checkpoint_.p, d + l.dflash2_off, l.dflash2_bytes,
                                   cudaMemcpyDeviceToDevice, s));
        dflash2_->adopt_checkpoint(n);
      }
    } else if (drafter_) {
      for (int i = 0; i < 3; ++i)
        CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_drafter_ + l.drafter[i].offset, d + l.drafter_off[i],
                                   l.drafter[i].bytes, cudaMemcpyDeviceToDevice, s));
      drafter_->adopt_checkpoint(c + checkpoint_drafter_, n, p.context);
    }
    if (p.has_logits)
      CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_logits_, d + l.logits, logits_.bytes, cudaMemcpyDeviceToDevice, s));
    rt_.restore_full_rows(d + l.rows, n);
    checkpoint_ids_ = p.ids;
    checkpoint_is_flash_ = p.is_flash;
    checkpoint_has_logits_ = p.has_logits;
    checkpoint_sampled_head_ = p.sampled_head;
    p.used = ++clock_;
  }

  // The target state and last prompt logits share checkpoint_. DFlash2 owns a separate ring snapshot.
  void save_checkpoint(bool flash) {
    char* p = static_cast<char*>(checkpoint_.p);
    rt_.save_checkpoint(p);
    if (flash) {
      if (dflash2_) dflash2_->save_checkpoint(dflash2_checkpoint_.p);
    } else if (drafter_) {
      drafter_->save_checkpoint(p + checkpoint_drafter_);
    }
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    checkpoint_ids_ = tokens_in_state_;
    checkpoint_is_flash_ = flash;
    checkpoint_has_logits_ = true;
    checkpoint_sampled_head_ = rt_.tp() && rt_.tp()->full_logits;
  }

  void restore_checkpoint() {
    const char* p = static_cast<const char*>(checkpoint_.p);
    rt_.restore_checkpoint(p);
    if (checkpoint_is_flash_) {
      if (dflash2_) dflash2_->restore_checkpoint(dflash2_checkpoint_.p, int(checkpoint_ids_.size()));
    } else if (drafter_) {
      drafter_->restore_checkpoint(p + checkpoint_drafter_);
    }
    if (checkpoint_has_logits_)
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

  int prepare_batch_prefill(int slot, std::span<const int32_t> prompt) {
    auto& history = batch_spec_histories_.at(std::size_t(slot));
    reset_batch_spec_cost(slot);
    int reused = 0;
    if (!options_.cold_prefill && !history.empty() && history.size() < prompt.size() &&
        std::equal(history.begin(), history.end(), prompt.begin())) {
      reused = int(history.size());
    }
    const auto stream = rt_.stream();
    if (auto* tp = rt_.tp(); tp && !tp->agree(reused, stream)) reused = 0;
    if (reused == 0) {
      reset_slot(batch_state_, slot, stream);
      if (batch_drafter_) batch_drafter_->reset_slot(slot);
      if (batch_dflash2_) batch_dflash2_->reset_slot(slot);
      history.clear();
    }
    return reused;
  }

  void commit_batch_taps(std::span<const std::int32_t> slots, std::span<const std::int64_t> pos,
                         std::span<const std::int32_t> keep, std::span<const std::int32_t> off) {
    if (!batch_drafter_ && !batch_dflash2_) return;
    constexpr int TAP_ROW = NUM_TAPS * HIDDEN;
    const auto stream = rt_.stream();
    std::vector<std::int32_t> rows;
    rows.reserve(off.back());
    for (std::size_t i = 0; i < slots.size(); ++i)
      for (int r = 0; r < keep[i]; ++r) rows.push_back(off[i] + r);
    CUDA_CHECK(cudaMemcpyAsync(batch_rows_.p, rows.data(), rows.size() * sizeof(rows[0]), cudaMemcpyHostToDevice, stream));
    kernels::batch::gather_rows(batch_dec_->taps(), TAP_ROW, static_cast<const std::int32_t*>(batch_rows_.p),
                                int(rows.size()), static_cast<__nv_bfloat16*>(batch_gathered_.p), stream);
    const auto* taps = static_cast<const __nv_bfloat16*>(batch_gathered_.p);
    const int n = int(slots.size());
    if (batch_drafter_) batch_drafter_->commit(n, slots.data(), pos.data(), keep.data(), taps);
    if (batch_dflash2_) batch_dflash2_->commit(n, slots.data(), pos.data(), keep.data(), taps);
  }

  void reset_batch_spec_cost(int slot) {
    batch_slot_ema_p_[slot] = BatchSpecCost::initial_p;
    batch_slot_spec_steps_[slot] = 0;
    batch_slot_draft_budget_[slot] = options_.batch_spec;
    batch_draft_support_.reset(slot);
  }

  void batch_draft_slots(int n, const std::int32_t* slots, const std::int32_t* anchors, const std::int64_t* next_pos) {
    constexpr int G = BatchDrafter::GAMMA;
    const cudaStream_t stream = rt_.stream();
    std::vector<int> df_idx, ds_idx;
    if (batch_draft_type_ == "dflash2") {
      df_idx.resize(std::size_t(n));
      for (int i = 0; i < n; ++i) df_idx[std::size_t(i)] = i;
    } else if (batch_draft_type_ == "hybrid") {
      for (int i = 0; i < n; ++i) {
        if (batch_dflash2_max_ctx_ > 0 && next_pos[i] <= batch_dflash2_max_ctx_) df_idx.push_back(i);
        else ds_idx.push_back(i);
      }
    } else {
      ds_idx.resize(std::size_t(n));
      for (int i = 0; i < n; ++i) ds_idx[std::size_t(i)] = i;
    }

    auto gather_sub = [&](const std::vector<int>& idx, std::vector<std::int32_t>& sl, std::vector<std::int32_t>& an,
                          std::vector<std::int64_t>& np) {
      sl.resize(idx.size()), an.resize(idx.size()), np.resize(idx.size());
      for (std::size_t k = 0; k < idx.size(); ++k) {
        const int i = idx[k];
        sl[k] = slots[i], an[k] = anchors[i], np[k] = next_pos[i];
      }
    };

    auto scatter_sync = [&](const std::vector<int>& idx, const std::int32_t* dev_tok, const float* dev_conf) {
      const std::size_t cnt = idx.size() * G;
      std::vector<std::int32_t> dh(cnt);
      std::vector<float> ch(cnt);
      if (rt_.tp()) {
        auto* base = static_cast<std::byte*>(batch_draft_bcast_.p);
        if (rt_.tp()->layout.tp_rank == 0 && dev_tok) {
          CUDA_CHECK(cudaMemcpyAsync(base, dev_tok, cnt * 4, cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(base + cnt * 4, dev_conf, cnt * 4, cudaMemcpyDeviceToDevice, stream));
        }
        rt_.tp()->broadcast_proposal(batch_draft_bcast_.p, cnt * 8, stream);
        CUDA_CHECK(cudaMemcpyAsync(dh.data(), base, cnt * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ch.data(), base + cnt * 4, cnt * 4, cudaMemcpyDeviceToHost, stream));
        rt_.tp()->wait(stream);
      } else {
        CUDA_CHECK(cudaMemcpyAsync(dh.data(), dev_tok, cnt * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ch.data(), dev_conf, cnt * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      for (std::size_t k = 0; k < idx.size(); ++k) {
        const int sl = slots[idx[k]];
        std::copy_n(dh.begin() + std::ptrdiff_t(k) * G, G, batch_drafts_.begin() + std::ptrdiff_t(sl) * G);
        std::copy_n(ch.begin() + std::ptrdiff_t(k) * G, G, batch_conf_.begin() + std::ptrdiff_t(sl) * G);
      }
    };

    if (!df_idx.empty()) {
      const int ndf = int(df_idx.size());
      std::vector<std::int32_t> df_max_drafts;
      int total_rows = ndf * G;
      const BatchSpecCost cost(batch_spec_config_, n, options_.max_batch, options_.batch_spec_threshold);
      if (cost.use_proposal_budgets()) {
        df_max_drafts.reserve(df_idx.size());
        total_rows = 0;
        for (const int i : df_idx) {
          const int slot = slots[i];
          const int budget = cost.proposal_budget(options_.batch_spec, options_.batch_spec_min, batch_slot_ema_p_[slot],
                                                  batch_slot_spec_steps_[slot]);
          batch_slot_draft_budget_[slot] = budget;
          df_max_drafts.push_back(budget);
          total_rows += budget;
        }
      }
      std::vector<std::int32_t> sl, an;
      std::vector<std::int64_t> np;
      gather_sub(df_idx, sl, an, np);
      const auto temps = batch_draft_support_.proposal_temperatures(sl);
      const bool sampled = !temps.empty();
      const auto seed = sampled ? batch_seed(++batch_draws_) : 0;
      BatchDFlash2::Drafts d{};
      if (batch_dflash2_) {
        d = batch_dflash2_->propose(ndf, sl.data(), an.data(), np.data(),
                                    df_max_drafts.empty() ? nullptr : df_max_drafts.data(),
                                    sampled ? temps.data() : nullptr, seed);
      } else if (batch_dflash2_head_) {
        batch_dflash2_head_->run_rows(total_rows);
      }
      if (sampled) batch_draft_support_.download(d, sl.size(), stream);
      scatter_sync(df_idx, d.tokens, d.conf);
      if (sampled) batch_draft_support_.store(sl);
    }
    if (!ds_idx.empty()) {
      const int nds = int(ds_idx.size());
      for (const int i : ds_idx) {
        batch_slot_draft_budget_[slots[i]] = options_.batch_spec;
        batch_draft_support_.clear_support(slots[i]);
      }
      if (batch_drafter_) {
        std::vector<std::int32_t> sl, an;
        std::vector<std::int64_t> np;
        gather_sub(ds_idx, sl, an, np);
        const BatchDrafter::Drafts d =
            batch_drafter_->draft(nds, sl.data(), an.data(), np.data(), 0.f, batch_seed(++batch_draws_),
                                  options_.batch_spec);
        scatter_sync(ds_idx, d.tokens, d.conf);
      } else if (rt_.tp()) {
        ++batch_draws_;
        scatter_sync(ds_idx, nullptr, nullptr);
      }
    }
  }

  std::array<std::vector<int32_t>, 7> batch_histories_;
  std::vector<std::vector<int32_t>> batch_spec_histories_;
  std::vector<DeviceBuffer> batch_head_storage_;
  Dense batch_unembed_;
  BatchState batch_state_;
  std::string batch_draft_type_;
  int batch_dflash2_max_ctx_ = 0;
  BatchSpecConfig batch_spec_config_;
  BatchDraftSupport batch_draft_support_;
  std::unique_ptr<BatchDecoder> batch_dec_;
  std::unique_ptr<BatchDrafter> batch_drafter_;
  std::vector<DeviceBuffer> batch_dflash2_head_storage_;
  std::unique_ptr<BatchDFlash2Head> batch_dflash2_head_;
  std::unique_ptr<BatchDFlash2> batch_dflash2_;
  DeviceBuffer batch_gathered_, batch_rows_, batch_draft_bcast_;
  std::vector<std::int32_t> batch_drafts_;
  std::vector<float> batch_conf_;
  std::vector<float> batch_slot_ema_p_;
  std::vector<int> batch_slot_spec_steps_;
  std::vector<int> batch_slot_draft_budget_;  // available proposals for the next verification
  std::uint64_t batch_draws_ = 0;
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
  DeviceBuffer checkpoint_, dflash2_checkpoint_;
  std::size_t checkpoint_drafter_ = 0, checkpoint_logits_ = 0;  // byte offsets in checkpoint_
  std::vector<std::int32_t> checkpoint_ids_;
  std::vector<std::int32_t> tokens_in_state_;
  bool checkpoint_is_flash_ = false;
  bool checkpoint_has_logits_ = false;
  bool checkpoint_sampled_head_ = false;
  std::vector<Parked> parked_;
  std::size_t park_budget_ = 0;  // bytes
  std::uint64_t clock_ = 0;      // last use of a parked conversation
  bool parking_ = false;         // a parked buffer is being allocated
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
