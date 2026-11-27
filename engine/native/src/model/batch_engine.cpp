#include "model/batch_engine.h"

#include "core/device.h"
#include "kernels/batch/batch.h"
#include "model/batch_dflash2.h"
#include "model/batch_drafter.h"
#include "model/dflash2.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace ingine::model {
namespace {

constexpr int G = BatchDrafter::GAMMA;
static_assert(G == BatchDFlash2::GAMMA);
constexpr int kSupport = kernels::batch::RejectSpec::kSupport;
constexpr int TAP_ROW = NUM_TAPS * HIDDEN;  // bf16 per row of taps

// the Gumbel noise of sampling call `draw`
std::uint64_t seed_of(std::uint64_t draw) { return draw * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull; }

std::string resolve_batch_draft_type(const BatchEngineOptions& o) {
  std::string dt = o.batch_draft_type;
  if (dt.empty()) {
    if (const char* env = std::getenv("INGINE_BATCH_DRAFT_TYPE"); env && *env) dt = env;
    else if (!o.dflash2_dir.empty()) dt = "dflash2";
    else dt = "dspark";
  }
  if (dt != "dspark" && dt != "dflash2" && dt != "hybrid")
    throw std::invalid_argument("BatchEngine: batch_draft_type must be dspark, dflash2, or hybrid");
  return dt;
}

std::filesystem::path resolve_dflash2_dir(const BatchEngineOptions& o) {
  if (!o.dflash2_dir.empty()) return o.dflash2_dir;
  if (const char* env = std::getenv("INKLING_DFLASH2_DIR"); env && *env) return env;
  const auto sibling = o.model_dir.parent_path() / "Inkling-2-Mini-DFlash2";
  if (std::filesystem::exists(sibling / "model.safetensors")) return sibling;
  throw std::invalid_argument("BatchEngine: DFlash2 requires --dflash2-dir or INKLING_DFLASH2_DIR");
}

int batch_spec_env_int(const char* name, int fallback, int minimum) {
  const char* env = std::getenv(name);
  if (!env) return fallback;
  const char* end = env + std::string_view(env).size();
  int value = 0;
  const auto [ptr, ec] = std::from_chars(env, end, value);
  return ec == std::errc{} && ptr == end && value >= minimum ? value : fallback;
}

}  // namespace

BatchSpecConfig batch_spec_config(TpRuntime* tp, cudaStream_t stream) {
  static_assert(std::is_trivially_copyable_v<BatchSpecConfig>);
  const bool root = !tp || tp->layout.tp_rank == 0;
  BatchSpecConfig config;
  if (root) {
    if (const char* env = std::getenv("INGINE_BATCH_SPEC_ADAPTIVE"); env && std::string_view(env) == "0")
      config.adaptive = 0;
    config.dcut = batch_spec_env_int("INGINE_BATCH_DCUT", config.dcut, 0) != 0;
    if (const char* env = std::getenv("INGINE_BATCH_DCUT_THETA")) {
      char* end = nullptr;
      const float theta = std::strtof(env, &end);
      if (*end == '\0' && theta > 0.f && theta < 1.f) config.dcut_theta = theta;
    }
    config.dcut_min_k = batch_spec_env_int("INGINE_BATCH_DCUT_MIN_K", config.dcut_min_k, -1);
    config.dcut_max_t = batch_spec_env_int("INGINE_BATCH_DCUT_MAX_T", config.dcut_max_t, 0);
    config.dcut_bucket = batch_spec_env_int("INGINE_BATCH_DCUT_BUCKET", config.dcut_bucket, 0);
    if (const char* env = std::getenv("INGINE_BATCH_DCUT_REJECT_EMA_WEIGHT")) {
      char* end = nullptr;
      const float weight = std::strtof(env, &end);
      if (end != env && *end == '\0' && weight >= 0.f && weight <= 1.f)
        config.dcut_reject_ema_weight = weight;
    }
  }
  if (tp) {
    DeviceBuffer device(sizeof(config));
    if (root)
      CUDA_CHECK(cudaMemcpyAsync(device.p, &config, sizeof(config), cudaMemcpyHostToDevice, stream));
    tp->broadcast_proposal(device.p, sizeof(config), stream);
    CUDA_CHECK(cudaMemcpyAsync(&config, device.p, sizeof(config), cudaMemcpyDeviceToHost, stream));
    tp->wait(stream);
  }
  return config;
}

BatchDraftSupport::BatchDraftSupport(int slots)
    : slot_temps_(slots), slot_has_support_(slots),
      slot_ids_(std::size_t(slots) * G * kSupport), slot_picked_(std::size_t(slots) * G),
      slot_probs_(slot_ids_.size()), download_ids_(slot_ids_.size()), download_picked_(slot_picked_.size()),
      download_probs_(slot_ids_.size()), row_active_(std::size_t(slots) * (G + 1)),
      row_ids_(row_active_.size() * kSupport), row_picked_(row_active_.size()), row_probs_(row_ids_.size()) {}

void BatchDraftSupport::reset(int slot) {
  if (slot_temps_.empty()) return;
  slot_temps_[slot] = 0.f;
  slot_has_support_[slot] = false;
}

void BatchDraftSupport::set_temperature(int slot, float temperature) {
  if (slot_temps_.empty()) return;
  if (slot_temps_[slot] != temperature) slot_has_support_[slot] = false;
  slot_temps_[slot] = temperature;
}

void BatchDraftSupport::clear_support(int slot) {
  if (!slot_has_support_.empty()) slot_has_support_[slot] = false;
}

bool BatchDraftSupport::has_support(int slot, float temperature) const {
  return !slot_has_support_.empty() && slot_has_support_[slot] && temperature > 0.f &&
         slot_temps_[slot] == temperature;
}

std::vector<float> BatchDraftSupport::proposal_temperatures(std::span<const std::int32_t> slots) const {
  if (slot_temps_.empty() ||
      std::none_of(slots.begin(), slots.end(), [&](int slot) { return slot_temps_[slot] > 0.f; }))
    return {};
  std::vector<float> temperatures;
  temperatures.reserve(slots.size());
  for (const int slot : slots) temperatures.push_back(slot_temps_[slot]);
  return temperatures;
}

void BatchDraftSupport::download(const BatchDFlash2::Drafts& drafts, std::size_t sequences, cudaStream_t stream) {
  const std::size_t count = sequences * G;
  CUDA_CHECK(cudaMemcpyAsync(download_ids_.data(), drafts.support_ids, count * kSupport * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(download_probs_.data(), drafts.support_probs, count * kSupport * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(download_picked_.data(), drafts.support_picked, count * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream));
}

void BatchDraftSupport::store(std::span<const std::int32_t> slots) {
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const int slot = slots[i];
    slot_has_support_[slot] = slot_temps_[slot] > 0.f;
    if (!slot_has_support_[slot]) continue;
    const std::size_t src = i * G, dst = std::size_t(slot) * G;
    std::copy_n(download_ids_.data() + src * kSupport, G * kSupport, slot_ids_.data() + dst * kSupport);
    std::copy_n(download_probs_.data() + src * kSupport, G * kSupport, slot_probs_.data() + dst * kSupport);
    std::copy_n(download_picked_.data() + src, G, slot_picked_.data() + dst);
  }
}

const BatchRejectInputs* BatchDraftSupport::verification_inputs(std::span<const std::int32_t> slots,
                                                                std::span<const std::int32_t> widths,
                                                                std::span<const std::int32_t> tokens,
                                                                std::span<const float> temperatures) {
  if (slot_temps_.empty()) return nullptr;
  bool any = false;
  std::size_t off = 0;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const int slot = slots[i];
    if (has_support(slot, temperatures[i])) {
      // The anchor predicts draft zero. The final row predicts a bonus token and stays inactive.
      for (int d = 0; d < widths[i] - 1; ++d) {
        const std::size_t draft = std::size_t(slot) * G + d;
        const std::size_t r = off + d;
        const auto* ids = slot_ids_.data() + draft * kSupport;
        const auto* probs = slot_probs_.data() + draft * kSupport;
        const int picked = slot_picked_[draft];
        if (picked < 0 || picked >= kSupport || ids[picked] != tokens[r + 1] || !(probs[picked] > 0.f)) break;
        if (!any) {
          std::fill_n(row_active_.data(), tokens.size(), std::uint8_t(0));
          any = true;
        }
        row_active_[r] = 1;
        std::copy_n(ids, kSupport, row_ids_.data() + r * kSupport);
        std::copy_n(probs, kSupport, row_probs_.data() + r * kSupport);
        row_picked_[r] = picked;
      }
    }
    off += widths[i];
  }
  if (!any) return nullptr;
  inputs_ = {row_active_.data(), row_ids_.data(), row_probs_.data(), row_picked_.data()};
  return &inputs_;
}

BatchSpecCost::BatchSpecCost(BatchSpecConfig config, int n, int slots, float threshold)
    : config_(config),
      n_(n),
      use_threshold_(threshold > 0.f),
      calibrate_(config.adaptive && n >= 5 && use_threshold_),
      theta_(std::max(0.05f, threshold * float(n) / float(slots))),
      dcut_theta_(config.dcut_theta) {
  if (calibrate_)
    theta_ = std::max(theta_, threshold * std::pow(float(n) / float(slots), 0.75f));
  if (config_.dcut && dcut_theta_ == 0.f) {
    // GB10 measurements with rejection sampling favored a 0.15 cutoff at high concurrency.
    const float target = 0.15f;
    const float base = std::min(theta_, target);
    const float ramp = std::clamp(float(n - 4) / 16.f, 0.f, 1.f);
    dcut_theta_ = std::lerp(base, target, ramp);
  }
}

int BatchSpecCost::minimum_width(int k_max, int k_min, float ema_p, int steps) const {
  if (calibrate_ && n_ >= 8 && steps >= 3 && ema_p < 0.35f) return std::min(1, k_max);
  return std::min(k_min, k_max);
}

int BatchSpecCost::verification_width(int k_max, int k_min, const float* confidence, float ema_p, int steps) const {
  if (!use_threshold_ || k_max == 0) return k_max;
  const int k_min_eff = minimum_width(k_max, k_min, ema_p, steps);
  const bool calibrated = calibrate_ && steps >= 2 && ema_p < 0.70f;
  const float cal = std::clamp(ema_p, 0.15f, 1.f);
  auto p_step = [&](int k) {
    const float c = confidence[k];
    return calibrated ? std::min(c, std::sqrt(c * cal)) : c;
  };
  float prod = 1.f;
  int k = 0;
  while (k < k_max && ((prod *= p_step(k)) > theta_ || k < k_min_eff)) ++k;
  return k;
}

int BatchSpecCost::dcut_minimum_width(int k_max, int k_min) const {
  const int minimum = config_.dcut_min_k >= 0 ? config_.dcut_min_k : (n_ < 8 ? k_min : 0);
  return std::min(minimum, k_max);
}

void BatchSpecCost::allocate_widths(std::span<const Slot> slots, int k_min, std::span<std::int32_t> out_k) const {
  if (!config_.dcut) {
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const Slot& slot = slots[i];
      out_k[i] = verification_width(slot.k_max, k_min, slot.confidence, slot.ema_p, slot.steps);
    }
    return;
  }

  struct Candidate {
    float gain;
    int slot;
    int depth;
    bool required;
  };
  const int n = int(slots.size());
  std::vector<Candidate> candidates;
  candidates.reserve(slots.size() * G);
  int min_drafts = 0;
  int selected = 0;
  for (int i = 0; i < n; ++i) {
    const Slot& slot = slots[i];
    const int minimum = dcut_minimum_width(slot.k_max, k_min);
    min_drafts += minimum;
    const bool sampled = slot.temperature > 0.f;
    const float initial_scale = sampled && !slot.reject && slot.steps == 0
                                    ? 1.f / (1.f + 0.5f * slot.temperature)
                                    : 1.f;
    const float ema_weight = sampled ? 0.40f : 0.25f;
    float gain = 1.f;
    for (int k = 0; k < slot.k_max; ++k) {
      const float confidence = std::isfinite(slot.confidence[k]) ? std::clamp(slot.confidence[k], 0.f, 1.f) : 0.f;
      float p_step;
      if (slot.reject && config_.dcut_reject_ema_weight > 0.f) {
        // Optional unclamped EMA blend for experimentation (INGINE_BATCH_DCUT_REJECT_EMA_WEIGHT > 0).
        const float target = slot.steps == 0 ? 0.78f : slot.ema_p;
        const float weight = slot.steps == 0 ? 0.50f : config_.dcut_reject_ema_weight;
        p_step = std::clamp(std::lerp(confidence, target, weight), 0.f, 1.f);
      } else {
        p_step = slot.steps == 0 ? confidence * initial_scale
                                 : std::min(confidence, std::lerp(confidence, slot.ema_p, ema_weight));
      }
      gain *= p_step;
      const bool required = k < minimum;
      candidates.push_back({gain, i, k, required});
      if (required || gain >= dcut_theta_) ++selected;
    }
  }

  const int available = int(candidates.size());
  const int limit = config_.dcut_max_t > 0 ? std::min(available, std::max(0, config_.dcut_max_t - n)) : available;
  // When the cap binds, spend the budget on the strongest prefixes across the entire batch.
  const bool cap_binds = selected > limit;
  if (cap_binds) min_drafts = 0;
  selected = std::min(selected, limit);
  std::fill(out_k.begin(), out_k.end(), 0);
  if (selected == 0) return;
  // Both required rows and cumulative gains are prefix-ordered within a slot. Break ties by depth and slot
  // so any prefix of the sorted candidates gives valid widths and identical selections on both ranks.
  std::sort(candidates.begin(), candidates.end(), [cap_binds](const Candidate& a, const Candidate& b) {
    if (!cap_binds && a.required != b.required) return a.required > b.required;
    if (a.gain != b.gain) return a.gain > b.gain;
    if (a.depth != b.depth) return a.depth < b.depth;
    return a.slot < b.slot;
  });

  const int bucket = config_.dcut_bucket;
  if (bucket > 1 && n >= 4) {
    const int remainder = (n + selected) % bucket;
    if (remainder != 0) {
      const int lo = std::max(0, selected - remainder);
      const int added = std::min(limit - selected, bucket - remainder);
      float added_gain = 0.f;
      for (int j = 0; j < added; ++j) added_gain += candidates[selected + j].gain;
      if (lo < min_drafts || added <= bucket / 2 || added_gain >= float(added) * 0.70f * dcut_theta_)
        selected += added;
      else
        selected = lo;
    }
  }

  for (int j = 0; j < selected; ++j) ++out_k[candidates[j].slot];
}

int BatchSpecCost::proposal_budget(int k_max, int k_min, float ema_p, int steps) const {
  if (config_.dcut) {
    const float p = std::sqrt(std::clamp(steps == 0 ? initial_p : ema_p, 0.15f, 0.95f));
    const int floor = std::min(k_max, std::max(2, dcut_minimum_width(k_max, k_min) + 1));
    int k = 0;
    for (float gain = p; k < k_max && gain >= dcut_theta_; gain *= p) ++k;
    // Leave one proposal beyond the predicted cutoff so improving slots can recover.
    return std::clamp(k + 1, floor, k_max);
  }
  if (!calibrate_ || steps < 2 || ema_p >= 0.65f) return k_max;
  const float cal = std::clamp(ema_p, 0.15f, 0.95f);
  const int est = int(std::ceil(std::log(std::max(1e-4f, theta_)) / std::log(std::sqrt(cal))));
  const int k_min_eff = minimum_width(k_max, k_min, ema_p, steps);
  return std::clamp(est + 1, std::min(k_max, std::max(2, k_min_eff + 1)), k_max);
}

void BatchSpecCost::observe(int k, int accepted, float& ema_p, int& steps) {
  if (k <= 0) return;
  // A fully accepted prefix is right-censored, so it contributes no failure.
  const float p_hat = float(accepted) / float(accepted + (accepted < k ? 1 : 0));
  ema_p = 0.75f * ema_p + 0.25f * p_hat;
  ++steps;
}

BatchEngine::Stream::Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
BatchEngine::Stream::~Stream() {
  if (s) cudaStreamDestroy(s);
}

static std::unique_ptr<TpRuntime> make_batch_tp(const BatchEngineOptions& o) {
  validate_kv_formats(o.kv_fp8, o.kv_q4, o.kv_12);
  if (o.tp.size <= 1) return nullptr;
  RuntimeOptions ro;
  ro.model_dir = o.model_dir;
  ro.max_ctx = o.ctx;
  ro.dense_fp8 = false;
  ro.dense_pico12 = o.dense_pico12;
  ro.kv_fp8 = o.kv_fp8;
  ro.kv_q4 = o.kv_q4;
  ro.kv_12 = o.kv_12;
  ro.head_nvfp4 = o.head_nvfp4;
  ro.spec = o.spec > 0;
  ro.tp = o.tp;
  ro.tp.full_logits = false;
  ro.tp.serve = false;
  return std::make_unique<TpRuntime>(ro);
}

BatchEngine::BatchEngine(const BatchEngineOptions& o)
    : ablated_(!o.ablation.empty()),
      prefill_chunk_(o.prefill_chunk),
      spec_(o.spec),
      spec_threshold_(o.spec_threshold),
      spec_min_(std::max(0, std::min(o.spec_min, o.spec))),
      tp_(make_batch_tp(o)),
      w_(load_weights(o.model_dir, /*dense_fp8=*/false, stream_.s, o.head_nvfp4,
                      tp_ ? tp_->layout : TpLayout{}, o.dense_pico12)),
      abl_(load_ablation(o.ablation)),
      st_(make_batch_state(o.slots, o.ctx, o.kv_fp8, tp_ ? tp_->layout : TpLayout{}, o.kv_q4, o.kv_12)),
      pf_(std::make_unique<Prefiller>(w_, abl_, o.prefill_chunk, stream_.s, tp_.get())),
      dec_(std::make_unique<BatchDecoder>(w_, abl_, st_, o.slots, stream_.s, 8, tp_.get())),
      chunk_(std::size_t(std::max(o.prefill_chunk, 1)) * 4),
      logits_(std::size_t(UNPADDED_VOCAB) * 4),
      first_(4) {
  if (spec_ < 0 || spec_ > G) throw std::invalid_argument("BatchEngine: spec must be in 0..7");
  if (!(spec_threshold_ >= 0.f && spec_threshold_ < 1.f))
    throw std::invalid_argument("BatchEngine: spec_threshold must be in [0, 1)");
  if (spec_ > 0) {
    batch_spec_config_ = batch_spec_config(tp_.get(), stream_.s);
    batch_draft_type_ = resolve_batch_draft_type(o);
    const bool hybrid = batch_draft_type_ == "hybrid";
    if (hybrid) dflash2_max_ctx_ = batch_dflash2_max_ctx(std::max(1, o.dflash2_max_ctx), tp_.get(), stream_.s);
    const bool use_dspark = batch_draft_type_ == "dspark" || hybrid;
    const bool use_dflash2 = batch_draft_type_ == "dflash2" || (hybrid && dflash2_max_ctx_ > 0);
    if (use_dspark) {
      const bool shard_batch_drafter = [] {
        const char* e = std::getenv("INGINE_SHARD_BATCH_DRAFTER");
        return !e || !*e || std::atoi(e) != 0;
      }();
      if (!tp_ || shard_batch_drafter || tp_->layout.tp_rank == 0) {
        std::vector<DeviceBuffer> tmp_head;
        Dense full_head;
        if (tp_ || !w_.unembed.w.data || w_.unembed.fp8())
          full_head = load_head(o.model_dir, false, false, stream_.s, tmp_head).dense;
        drafter_ = std::make_unique<BatchDrafter>(o.model_dir, w_, o.slots, o.ctx, o.slots, stream_.s,
                                                  BatchDrafter::Precision::MXFP8, full_head.w.data ? &full_head : nullptr,
                                                  shard_batch_drafter ? tp_.get() : nullptr);
      }
    }
    if (use_dflash2) {
      if (tp_ && o.head_nvfp4 && !dflash2_full_head()) {
        dflash2_head_ = std::make_unique<BatchDFlash2Head>(*tp_, w_.unembed_nvfp4, o.slots, stream_.s);
      }
      if (!tp_ || tp_->layout.tp_rank == 0) {
        const auto dir = resolve_dflash2_dir(o);
        const kernels::head::Nvfp4Matrix* head_ovr = nullptr;
        kernels::head::Nvfp4Matrix loaded_head{};
        if (!dflash2_head_ && (tp_ || !w_.unembed_nvfp4.codes)) {
          loaded_head = load_head(o.model_dir, false, true, stream_.s, dflash2_head_storage_).nvfp4;
          head_ovr = &loaded_head;
        }
        dflash2_ = std::make_unique<BatchDFlash2>(dir, w_, o.slots, o.slots, stream_.s,
                                                  BatchDFlash2::default_precision(), BatchDFlash2::default_kv_fp8(),
                                                  head_ovr, dflash2_head_.get());
      }
    }
    if (dflash2_ && dec_->reject_enabled()) draft_support_ = BatchDraftSupport(o.slots);
    taps_chunk_ = DeviceBuffer(std::size_t(prefill_chunk_) * TAP_ROW * 2);
    gathered_ = DeviceBuffer(std::size_t(o.slots) * 8 * TAP_ROW * 2);
    rows_ = DeviceBuffer(std::size_t(o.slots) * 8 * 4);
    if (tp_) draft_bcast_ = DeviceBuffer(std::size_t(o.slots) * G * 8);
    drafts_.assign(std::size_t(o.slots) * G, 0);
    conf_.assign(std::size_t(o.slots) * G, 0.f);
    batch_slot_ema_p_.assign(std::size_t(o.slots), BatchSpecCost::initial_p);
    batch_slot_spec_steps_.assign(std::size_t(o.slots), 0);
    batch_slot_draft_budget_.assign(std::size_t(o.slots), spec_);
    if (!o.spec_trace.empty() && !(trace_ = std::fopen(o.spec_trace.c_str(), "w")))
      throw std::runtime_error("BatchEngine: cannot write " + o.spec_trace.string());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
}

BatchEngine::~BatchEngine() {
  cudaStreamSynchronize(stream_.s);
  if (trace_) std::fclose(trace_);
}

std::string BatchEngine::model_id() const { return ablated_ ? "inkling-2-mini-abliterated" : "inkling-2-mini"; }

std::size_t BatchEngine::device_bytes() const {
  std::size_t extra = 0;
  for (const auto& b : dflash2_head_storage_) extra += b.bytes;
  return st_.device_bytes() + dec_->device_bytes() + pf_->device_bytes() + chunk_.bytes + logits_.bytes + first_.bytes +
         (drafter_ ? drafter_->device_bytes() : 0) + (dflash2_ ? dflash2_->device_bytes() : 0) +
         (dflash2_head_ ? dflash2_head_->device_bytes() : 0) + extra + taps_chunk_.bytes + gathered_.bytes + rows_.bytes;
}

std::vector<std::int32_t> BatchEngine::admit(std::span<const Admission> requests) {
  std::vector<std::int32_t> first(requests.size());
  std::vector<std::size_t> together;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const Admission& a = requests[i];
    if (a.slot < 0 || a.slot >= st_.slots) throw std::invalid_argument("BatchEngine::admit: slot out of range");
    const int P = int(a.prompt.size());
    if (P < 1 || P + lookahead() + 1 > st_.ctx)
      throw std::invalid_argument("BatchEngine::admit: a prompt of " + std::to_string(P) +
                                  " tokens does not fit a slot of " + std::to_string(st_.ctx) + " positions");
    reset_slot(st_, a.slot, stream_.s);
    if (spec_ > 0) {
      batch_slot_ema_p_[a.slot] = BatchSpecCost::initial_p;
      batch_slot_spec_steps_[a.slot] = 0;
      batch_slot_draft_budget_[a.slot] = spec_;
      draft_support_.reset(a.slot);
      draft_support_.set_temperature(a.slot, a.temperature);
    }
    if (drafter_) drafter_->reset_slot(a.slot);
    if (dflash2_) dflash2_->reset_slot(a.slot);
    if (P <= kStepPrefill && P + 8 <= st_.ctx) together.push_back(i);
    else prefill_one(a, first[i]);
  }
  if (!together.empty()) prefill_together(requests, together, first);
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
  if (spec_ > 0 && !requests.empty()) {  // drafts after the first tokens
    const int n = int(requests.size());
    slot_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
    for (int i = 0; i < n; ++i)
      slot_[i] = requests[i].slot, anchor_[i] = first[i], next_pos_[i] = std::int64_t(requests[i].prompt.size());
    draft(n);
  }
  return first;
}

void BatchEngine::prefill_one(const Admission& a, std::int32_t& first) {
  const cudaStream_t s = stream_.s;
  const int P = int(a.prompt.size());
  State v = st_.view(a.slot);
  const bool sampled = a.temperature > 0.f;
  for (int p = 0; p < P; p += prefill_chunk_) {
    const int T = std::min(prefill_chunk_, P - p);
    const bool final_chunk = (p + T == P);
    CUDA_CHECK(cudaMemcpyAsync(chunk_.p, a.prompt.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    Tensor taps = rm(taps_chunk_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
    if (tp_) tp_->full_logits = sampled;
    float* log_dst = (!tp_ || (sampled && tp_->layout.tp_rank == 0)) ? static_cast<float*>(logits_.p) : nullptr;
    std::int32_t* grd_dst = (tp_ && !sampled) ? static_cast<std::int32_t*>(first_.p) : nullptr;
    const bool need_taps = drafter_ || dflash2_;
    pf_->run(v, static_cast<const std::int32_t*>(chunk_.p), T, p, log_dst,
             need_taps ? &taps : nullptr, {}, grd_dst, final_chunk);
    if (need_taps) {
      const std::int32_t slot = a.slot, count = T;
      const std::int64_t start = p;
      const auto* tp_ptr = static_cast<const __nv_bfloat16*>(taps_chunk_.p);
      if (drafter_) drafter_->commit(1, &slot, &start, &count, tp_ptr);
      if (dflash2_) dflash2_->commit(1, &slot, &start, &count, tp_ptr);
    }
  }
  const std::uint64_t seed = seed_of(++draws_);
  if (!tp_) {
    kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false, a.temperature,
                           seed, static_cast<std::int32_t*>(first_.p), s);
  } else if (sampled) {
    if (tp_->layout.tp_rank == 0)
      kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false, a.temperature,
                             seed, static_cast<std::int32_t*>(first_.p), s);
    tp_->broadcast_proposal(first_.p, 4, s);
  }
  CUDA_CHECK(cudaMemcpyAsync(&first, first_.p, 4, cudaMemcpyDeviceToHost, s));
  if (tp_) tp_->wait(s);
}

void BatchEngine::prefill_together(std::span<const Admission> requests, std::span<const std::size_t> which,
                                   std::vector<std::int32_t>& first) {
  // per step, every prompt with tokens left feeds its next min(8, left) of them
  std::vector<int> done(which.size(), 0);
  std::vector<std::size_t> member;  // the step's sequence i is which[member[i]]
  for (;;) {
    tok_.clear(), slot_.clear(), pos_.clear(), temp_.clear(), m_.clear(), member.clear();
    for (std::size_t j = 0; j < which.size(); ++j) {
      const Admission& a = requests[which[j]];
      const int left = int(a.prompt.size()) - done[j];
      if (left <= 0) continue;
      const int k = std::min(left, 8);
      tok_.insert(tok_.end(), a.prompt.begin() + done[j], a.prompt.begin() + done[j] + k);
      slot_.push_back(a.slot), pos_.push_back(done[j]), temp_.push_back(a.temperature), m_.push_back(k);
      member.push_back(j);
    }
    const int n = int(slot_.size());
    if (n == 0) break;
    const auto got = dec_->step(n, m_.data(), tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
    offsets(n);
    for (int i = 0; i < n; ++i) {
      const std::size_t j = member[std::size_t(i)];
      done[j] += m_[std::size_t(i)];
      if (done[j] == int(requests[which[j]].prompt.size()))
        first[which[j]] = got[std::size_t(off_[std::size_t(i) + 1]) - 1];
    }
    keep_ = m_;
    dec_->accept(keep_.data());  // every prompt row is kept
    if (drafter_ || dflash2_) commit_kept(n);
  }
}

void BatchEngine::offsets(int n) {
  off_.resize(std::size_t(n) + 1);
  off_[0] = 0;
  for (int i = 0; i < n; ++i) off_[std::size_t(i) + 1] = off_[std::size_t(i)] + m_[std::size_t(i)];
}

void BatchEngine::commit_kept(int n) {
  if (!drafter_ && !dflash2_) return;
  rows_host_.clear();
  for (int i = 0; i < n; ++i)
    for (int r = 0; r < keep_[std::size_t(i)]; ++r) rows_host_.push_back(off_[std::size_t(i)] + r);
  CUDA_CHECK(cudaMemcpyAsync(rows_.p, rows_host_.data(), rows_host_.size() * 4, cudaMemcpyHostToDevice, stream_.s));
  kernels::batch::gather_rows(dec_->taps(), TAP_ROW, static_cast<const std::int32_t*>(rows_.p), int(rows_host_.size()),
                              static_cast<__nv_bfloat16*>(gathered_.p), stream_.s);
  const auto* gptr = static_cast<const __nv_bfloat16*>(gathered_.p);
  if (drafter_) drafter_->commit(n, slot_.data(), pos_.data(), keep_.data(), gptr);
  if (dflash2_) dflash2_->commit(n, slot_.data(), pos_.data(), keep_.data(), gptr);
}

void BatchEngine::draft(int n) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t cnt = std::size_t(n) * G;
  draft_host_.resize(cnt), conf_host_.resize(cnt);

  // Partition indices into DFlash2 vs DSpark subsets (in dflash2 mode all -> df, in dspark mode all -> ds).
  std::vector<int> df_idx, ds_idx;
  if (batch_draft_type_ == "dflash2") {
    df_idx.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) df_idx[std::size_t(i)] = i;
  } else if (batch_draft_type_ == "hybrid") {
    for (int i = 0; i < n; ++i) {
      if (dflash2_max_ctx_ > 0 && next_pos_[std::size_t(i)] <= dflash2_max_ctx_) df_idx.push_back(i);
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
      sl[k] = slot_[std::size_t(i)];
      an[k] = anchor_[std::size_t(i)];
      np[k] = next_pos_[std::size_t(i)];
    }
  };

  auto scatter_sync = [&](const std::vector<int>& idx, const std::int32_t* dev_tok, const float* dev_conf) {
    const std::size_t sub_cnt = idx.size() * G;
    std::vector<std::int32_t> dt(sub_cnt);
    std::vector<float> dc(sub_cnt);
    if (tp_) {
      auto* base = static_cast<std::byte*>(draft_bcast_.p);
      if (tp_->layout.tp_rank == 0 && dev_tok) {
        CUDA_CHECK(cudaMemcpyAsync(base, dev_tok, sub_cnt * 4, cudaMemcpyDeviceToDevice, stream_.s));
        CUDA_CHECK(cudaMemcpyAsync(base + sub_cnt * 4, dev_conf, sub_cnt * 4, cudaMemcpyDeviceToDevice, stream_.s));
      }
      tp_->broadcast_proposal(draft_bcast_.p, sub_cnt * 8, stream_.s);
      CUDA_CHECK(cudaMemcpyAsync(dt.data(), base, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaMemcpyAsync(dc.data(), base + sub_cnt * 4, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      tp_->wait(stream_.s);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dt.data(), dev_tok, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaMemcpyAsync(dc.data(), dev_conf, sub_cnt * 4, cudaMemcpyDeviceToHost, stream_.s));
      CUDA_CHECK(cudaStreamSynchronize(stream_.s));
    }
    for (std::size_t k = 0; k < idx.size(); ++k) {
      const int sl = slot_[std::size_t(idx[k])];
      std::copy_n(dt.begin() + std::ptrdiff_t(k) * G, G, drafts_.begin() + std::ptrdiff_t(sl) * G);
      std::copy_n(dc.begin() + std::ptrdiff_t(k) * G, G, conf_.begin() + std::ptrdiff_t(sl) * G);
    }
  };

  if (!df_idx.empty()) {
    const int ndf = int(df_idx.size());
    std::vector<std::int32_t> df_max_drafts;
    int total_rows = ndf * G;
    const BatchSpecCost cost(batch_spec_config_, n, st_.slots, spec_threshold_);
    if (cost.use_proposal_budgets()) {
      df_max_drafts.reserve(df_idx.size());
      total_rows = 0;
      for (const int i : df_idx) {
        const int slot = slot_[i];
        const int budget = cost.proposal_budget(spec_, spec_min_, batch_slot_ema_p_[slot],
                                                batch_slot_spec_steps_[slot]);
        batch_slot_draft_budget_[slot] = budget;
        df_max_drafts.push_back(budget);
        total_rows += budget;
      }
    }
    if (dflash2_) {
      std::vector<std::int32_t> sl, an;
      std::vector<std::int64_t> np;
      gather_sub(df_idx, sl, an, np);
      const auto temps = draft_support_.proposal_temperatures(sl);
      const bool sampled = !temps.empty();
      const BatchDFlash2::Drafts d = dflash2_->propose(
          ndf, sl.data(), an.data(), np.data(), df_max_drafts.empty() ? nullptr : df_max_drafts.data(),
          sampled ? temps.data() : nullptr, sampled ? seed_of(++draws_) : 0);
      if (sampled) draft_support_.download(d, sl.size(), stream_.s);
      scatter_sync(df_idx, d.tokens, d.conf);
      if (sampled) draft_support_.store(sl);
    } else if (tp_) {
      if (dflash2_head_) dflash2_head_->run_rows(total_rows);
      scatter_sync(df_idx, nullptr, nullptr);
    }
  }

  if (!ds_idx.empty()) {
    const int nds = int(ds_idx.size());
    for (const int i : ds_idx) {
      batch_slot_draft_budget_[slot_[i]] = spec_;
      draft_support_.clear_support(slot_[i]);
    }
    if (drafter_) {
      std::vector<std::int32_t> sl, an;
      std::vector<std::int64_t> np;
      gather_sub(ds_idx, sl, an, np);
      const BatchDrafter::Drafts d =
          drafter_->draft(nds, sl.data(), an.data(), np.data(), 0.f, seed_of(++draws_), spec_);
      scatter_sync(ds_idx, d.tokens, d.conf);
    } else if (tp_) {
      ++draws_;
      scatter_sync(ds_idx, nullptr, nullptr);
    }
  }
  draft_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++draft_calls_;
}

void BatchEngine::step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) {
  const int n = int(seqs.size());
  emitted.resize(std::size_t(n));
  if (n == 0) return;
  m_.resize(std::size_t(n));
  if (spec_ > 0) {
    const BatchSpecCost cost(batch_spec_config_, n, st_.slots, spec_threshold_);
    spec_slots_.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) {
      const int slot = seqs[i].slot;
      spec_slots_[i] = {batch_slot_draft_budget_[slot], conf_.data() + std::size_t(slot) * G,
                        batch_slot_ema_p_[slot], batch_slot_spec_steps_[slot], seqs[i].temperature,
                        draft_support_.has_support(slot, seqs[i].temperature)};
    }
    cost.allocate_widths(spec_slots_, spec_min_, m_);
    for (auto& m : m_) ++m;
  } else {
    std::fill(m_.begin(), m_.end(), 1);
  }
  tok_.clear(), slot_.resize(std::size_t(n)), pos_.resize(std::size_t(n)), temp_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const Seq& q = seqs[i];
    tok_.push_back(q.pending);
    for (int k = 1; k < m_[std::size_t(i)]; ++k) tok_.push_back(drafts_[std::size_t(q.slot) * G + std::size_t(k - 1)]);
    slot_[i] = q.slot, pos_[i] = q.pos, temp_[i] = q.temperature;
  }
  const auto* reject = draft_support_.verification_inputs(slot_, m_, tok_, temp_);
  const auto tv0 = std::chrono::steady_clock::now();
  const auto got =
      dec_->step(n, m_.data(), tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_), reject);
  verify_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
  offsets(n);
  if (spec_ == 0) {
    for (int i = 0; i < n; ++i) {
      emitted[i].assign(1, got[std::size_t(i)]);
      seqs[i].pending = got[std::size_t(i)];
      seqs[i].pos += 1;
    }
    return;
  }
  keep_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const int K = m_[std::size_t(i)] - 1;                            // this sequence's drafts
    const std::int32_t* row = tok_.data() + off_[std::size_t(i)];    // pending, drafts
    const std::int32_t* out = got.data() + off_[std::size_t(i)];     // the target's token after each row
    int a = 0;
    while (a < K && row[1 + a] == out[a]) ++a;
    BatchSpecCost::observe(K, a, batch_slot_ema_p_[slot_[i]], batch_slot_spec_steps_[slot_[i]]);
    emitted[i].assign(row + 1, row + 1 + a);
    emitted[i].push_back(out[a]);
    keep_[i] = a + 1;
    spec_accepted_ += std::uint64_t(a);
    spec_rows_ += std::uint64_t(K + 1);
    if (trace_) {
      const float* c = conf_.data() + std::size_t(slot_[i]) * G;
      std::fprintf(trace_, "%d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", slot_[i], K, a, c[0], c[1], c[2], c[3], c[4],
                   c[5], c[6]);
    }
    draft_support_.set_temperature(slot_[i], temp_[i]);
    seqs[i].pending = anchor_[i] = out[a];
    seqs[i].pos = next_pos_[i] = pos_[i] + a + 1;
  }
  spec_steps_ += std::uint64_t(n);
  dec_->accept(keep_.data());
  commit_kept(n);
  draft(n);
}

}  // namespace ingine::model
