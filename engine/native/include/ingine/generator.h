// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::model {

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
};

// The token whose embedding an image token replaces (<|unused_200054|>, the chat template's image placeholder).
inline constexpr std::int32_t IMAGE_PLACEHOLDER_ID = 200054;

// One image of a prompt. The chat template marks an image with one placeholder token; the prompt holds
// tokens() ids in its place, one per 40x40-pixel tile in raster order, whose embeddings the vision tower
// computes from the pixels. These ids are negative (never a vocabulary id) and a function of the image
// content and the tile index (image::token_id), so prompts compare equal exactly when their images do.
struct PromptImage {
  int pos = 0;                                           // prompt position of the image's first token
  int height = 0, width = 0;                             // pixels
  std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // [height, width, 3]
  [[nodiscard]] int rows() const { return (height + 39) / 40; }
  [[nodiscard]] int cols() const { return (width + 39) / 40; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// A prompt: token ids and its images, in the order of their positions (without overlap).
struct Prompt {
  std::span<const std::int32_t> ids;
  std::span<const PromptImage> images;
};

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,        // empty state, the whole prompt prefilled
  Extend,      // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,  // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,      // restored a parked conversation's checkpoint (a prefix of this prompt; RuntimeOptions::park_mb)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0, control_ms = 0;
};

struct DecodeTiming {
  double step_ms = 0, reduce_ms = 0, head_exchange_ms = 0;
  int steps = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  int interrupted_prefill_offset = 0;
  bool used_dflash2 = false;
  Reuse reuse = Reuse::None;
  double prefill_seconds = 0, decode_seconds = 0;
  DecodeTiming decode_timing;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay or opt-in timing
  // Images: tokens of the prompt's images, how many images the vision tower encoded for this request (those
  // not inside the reused prefix) and its time (part of prefill_seconds).
  int image_tokens = 0, images_encoded = 0;
  double vision_seconds = 0;
};

enum class SpecWidths { Coarse, Fine, Fine2 };

inline SpecWidths parse_spec_widths(std::string_view value) {
  if (value == "coarse") return SpecWidths::Coarse;
  if (value == "fine") return SpecWidths::Fine;
  if (value == "fine2") return SpecWidths::Fine2;
  throw std::invalid_argument("--spec-widths must be coarse, fine or fine2");
}

inline std::span<const int> verify_width_grid(SpecWidths policy) {
  static constexpr int coarse[] = {2, 4, 8};
  static constexpr int fine[] = {1, 2, 3, 4, 5, 6, 7, 8};
  if (policy == SpecWidths::Coarse) return coarse;
  return std::span(fine).subspan(policy == SpecWidths::Fine2 ? 1 : 0);
}

enum class TpWire { Fp32, Bf16, Bf16Moe };

struct TpOptions {
  int size = 1, rank = 0;
  TpWire wire = TpWire::Fp32;
  bool full_logits = false;  // gathered only to rank 0
  bool profile = false;
  bool serve = false;  // Capture both heads and synchronize callback cancellation.
};

// Command-line values override INKLING_TP_SIZE, INKLING_TP_RANK and INKLING_TP_WIRE.
TpOptions tp_options(std::optional<int> size = {}, std::optional<int> rank = {}, std::string wire = {});

inline void validate_dense_formats(bool fp8, bool pico12) {
  if (fp8 && pico12) throw std::invalid_argument("--dense-fp8 and --dense-pico12 are mutually exclusive");
}

inline void validate_kv_formats(bool fp8, bool q4, bool kv12) {
  if (kv12 && (fp8 || q4))
    throw std::invalid_argument("--kv-12 is mutually exclusive with --kv-fp8 and --kv-q4");
  if (fp8 && q4) throw std::invalid_argument("--kv-fp8 and --kv-q4 are mutually exclusive");
}

struct RuntimeOptions {
  int max_batch = 1;
  int batch_ctx = 32768;
  int batch_spec = 0;  // 0: Track 2a plain batch (1..8); 1..7: Track 3 batched speculative decoding (1..128)
  float batch_spec_threshold = 0.50f;
  int batch_spec_min = 2;
  // Batched speculative drafter: "dspark", "dflash2", or "hybrid" (empty: INGINE_BATCH_DRAFT_TYPE or "dspark").
  std::string batch_draft_type;
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  // Tokens per prefill chunk. 3072 (the largest the sliding attention takes) reads a long prompt 6-10% faster
  // than 2048 for 0.38 GB more scratch; results differ between chunk sizes only by rounding.
  int prefill_chunk = 3072;
  bool dense_fp8 = true, kv_fp8 = false, spec = false;
  bool dense_pico12 = false;  // Bit-exact BF16 projections and dense unembedding
  bool kv_q4 = false;  // Q4_G64 full-attention caches
  bool kv_12 = false;  // Bit-exact BF16 full-attention caches
  float spec_threshold = 0.8f;
  SpecWidths spec_widths = SpecWidths::Coarse;
  std::string draft_type = "dspark";
  std::filesystem::path dflash2_dir;
  float hybrid_dflash2_threshold = 0.50f;
  int dflash2_max_ctx = 1536;
  bool dflash2_adaptive = false;
  bool spec_trace = false;
  bool draft_nvfp4 = false;
  int spec_verify_width = 0;  // fixed-width TP=2 forced replay benchmark, or 0 for confidence selection
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = -1;
  // Image input: load the vision tower (636 MB of the checkpoint, resident) and accept prompts with images.
  bool vision = false;
  bool cold_prefill = false;  // benchmark without prompt reuse
  TpOptions tp;
  bool head_nvfp4 = false;  // packed NVFP4 unembedding, independent of projection precision
  bool cert_head = false;  // TP=1 exact greedy head; auto-enabled for PICO12; INGINE_CERT_HEAD overrides
};

struct BatchSampling {
  Sampling params;
  uint64_t seed = 0, counter = 0;
};

struct BatchSlotStep {
  int slot;
  int32_t token;
  BatchSampling* sampling;
};

struct PrefillChunkResult {
  int offset;
  int reused_tokens;
  std::optional<int32_t> token;
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  // The same for a prompt with images. Generators without image input take prompts without images only.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    if (!prompt.images.empty()) throw std::invalid_argument("this generator does not take images");
    return generate(prompt.ids, max_new_tokens, sampling, stop_ids, on_tokens);
  }
  virtual int max_batch() const { return 1; }
  virtual int batch_max_ctx() const { return max_model_len(); }
  virtual int batch_spec() const { return 0; }
  virtual int reusable_prefix(int, std::span<const int32_t>) const { return 0; }
  virtual PrefillChunkResult prefill_slot_chunk(int, std::span<const int32_t>, int, BatchSampling&, int /*stop_at*/ = 0) {
    throw std::logic_error("batch prefill unavailable");
  }
  // Clone a batch slot KV/conv/drafter prefix state into dst_slot (shared-prefix forking).
  virtual bool clone_slot(int /*src_slot*/, int /*dst_slot*/) { return false; }
  virtual std::vector<int32_t> decode_batch_step(std::span<const BatchSlotStep>) {
    throw std::logic_error("batch decode unavailable");
  }
  virtual std::vector<std::vector<int32_t>> decode_batch_step_multi(std::span<const BatchSlotStep> rows) {
    const auto one = decode_batch_step(rows);
    std::vector<std::vector<int32_t>> out;
    out.reserve(one.size());
    for (int32_t t : one) out.push_back({t});
    return out;
  }
  virtual bool promote_solo_to_batch(int, int32_t) { return true; }
  virtual void release_slot(int) {}
  // The callback runs at every committed speculative emission boundary. Returning false leaves
  // the last emitted token as the next decode input and transfers the request's sampler state.
  virtual GenerateStats generate_interruptible(const Prompt& prompt, int limit, BatchSampling& sampling,
                                               std::span<const int32_t> stops,
                                               const std::function<bool(std::span<const int32_t>)>& emit) {
    return generate(prompt, limit, sampling.params, stops, emit);
  }
  virtual bool supports_images() const { return false; }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
