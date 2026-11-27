// ingine-bench: end-to-end speed, prefill tokens/s and decode tokens/s for one request (port of
// engine/bench/bench_e2e.py; same flags, lines and JSON).
//
//     ingine-bench --tokens 512 --dense-fp8
//     ingine-bench --spec --forced agentic.jsonl   (teacher-forced replay, see forced.h)
//     ingine-bench --spec --forced agentic.jsonl --forced-free 1 --temperature 1   (free generation from each
//         item's prompt for len(output) tokens at the temperature: sampled A/B, where forced acceptance by
//         equality under-counts)
#include "forced.h"
#include "ingine/generator.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::pair<std::string, std::string>> kPrompts = {
    {"chat", "Explain how a hash map handles collisions, with a short Python example."},
    {"code", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests."},
};

// Python repr(float).
std::string py_float(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[64];
  auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string s(buf, end);  // [-]d[.ddd]e[+-]XX, shortest round-trip digits
  std::string sign;
  if (s[0] == '-') {
    sign = "-";
    s.erase(0, 1);
  }
  const size_t e = s.find('e');
  const int exp = std::stoi(s.substr(e + 1));
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  if (exp < -4 || exp >= 16) {
    std::string mant = digits.substr(0, 1);
    if (digits.size() > 1) mant += "." + digits.substr(1);
    char ebuf[16];
    std::snprintf(ebuf, sizeof ebuf, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    return sign + mant + ebuf;
  }
  std::string out;
  if (exp < 0) {
    out = "0." + std::string(size_t(-exp - 1), '0') + digits;
  } else if (size_t(exp) + 1 >= digits.size()) {
    out = digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
  } else {
    out = digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
  }
  return sign + out;
}

// Python repr(str) of (possibly partial) UTF-8 text; invalid bytes decode to U+FFFD as in tokenizer.decode.
std::string py_repr(const std::string& text) {
  std::vector<int32_t> cps;
  const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
  utf8proc_ssize_t left = utf8proc_ssize_t(text.size());
  while (left > 0) {
    utf8proc_int32_t cp;
    utf8proc_ssize_t n = utf8proc_iterate(p, left, &cp);
    if (n <= 0) {
      cp = 0xFFFD;
      n = 1;
    }
    cps.push_back(cp);
    p += n;
    left -= n;
  }
  const bool has_single = std::find(cps.begin(), cps.end(), '\'') != cps.end();
  const bool has_double = std::find(cps.begin(), cps.end(), '"') != cps.end();
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (int32_t cp : cps) {
    if (cp == quote || cp == '\\') {
      out += '\\';
      out += char(cp);
    } else if (cp == '\n') {
      out += "\\n";
    } else if (cp == '\r') {
      out += "\\r";
    } else if (cp == '\t') {
      out += "\\t";
    } else {
      const utf8proc_category_t cat = utf8proc_category(cp);
      const bool printable =
          cp == ' ' || !(cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_CS ||
                         cat == UTF8PROC_CATEGORY_CO || cat == UTF8PROC_CATEGORY_CN || cat == UTF8PROC_CATEGORY_ZL ||
                         cat == UTF8PROC_CATEGORY_ZP || cat == UTF8PROC_CATEGORY_ZS);
      if (printable) {
        utf8proc_uint8_t enc[4];
        const utf8proc_ssize_t n = utf8proc_encode_char(cp, enc);
        out.append(reinterpret_cast<const char*>(enc), size_t(n));
      } else {
        char buf[16];
        if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
        else if (cp < 0x10000) std::snprintf(buf, sizeof buf, "\\u%04x", cp);
        else std::snprintf(buf, sizeof buf, "\\U%08x", cp);
        out += buf;
      }
    }
  }
  out += quote;
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) return env;
  return std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir(), backend = "fast", ablation, effort = "low", forced, forced_trace;
  int max_ctx = 32768, tokens = 512, runs = 3, long_prompt = 0, forced_items = 0, forced_skip = 0, forced_free = -1;
  int spec_verify_width = 0;
  int tp_size = 0, tp_rank = -1, prefill_chunk = 0;
  std::string tp_wire;
  bool tp_profile = false, cold_prefill = false;
  bool dense_fp8 = false, dense_pico12 = false, head_nvfp4 = false, cert_head = false;
  bool kv_fp8 = false, kv_q4 = false, kv_12 = false, spec = false;
  bool draft_nvfp4 = false;
  std::string spec_widths = "coarse", loopcd = "off";
  double loopcd_omega = 0.0;
  int loopcd_ref = 1;
  bool spec_timing = false;
  double hybrid_dflash2_threshold = 0.50;
  int dflash2_max_ctx = 1536;
  bool dflash2_adaptive = false;
  std::string draft_type = "dspark", dflash2_dir;
  double spec_threshold = 0.8, temperature = 0.0;

  ingine::serve::ArgParser ap("ingine-bench",
                              "End-to-end ingine speed: prefill tokens/s and greedy decode tokens/s for one request.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--backend", "{fast}", "", &backend, {"fast"});
  ap.option("--ablation", "", "", &ablation);
  ap.option("--max-ctx", "", "", &max_ctx);
  ap.option("--tokens", "", "", &tokens);
  ap.option("--runs", "", "", &runs);
  ap.option("--effort", "", "", &effort);
  ap.option("--tp-size", "", "tensor parallel size (default INKLING_TP_SIZE or 1)", &tp_size);
  ap.option("--tp-rank", "", "tensor parallel rank (default INKLING_TP_RANK or 0)", &tp_rank);
  ap.option("--tp-wire", "{bf16,fp32,bf16-moe}", "reduction wire precision (default INKLING_TP_WIRE or fp32)",
            &tp_wire, {"bf16", "fp32", "bf16-moe"});
  ap.flag("--tp-profile", "Time every CUDA-graph reduction and the head exchange", &tp_profile);
  ap.option("--prefill-chunk", "", "tokens per prefill chunk", &prefill_chunk);
  ap.flag("--cold-prefill", "Disable prompt checkpoint and prefix reuse", &cold_prefill);
  ap.flag("--dense-fp8", "", &dense_fp8);
  ap.flag("--dense-pico12", "Bit-exact PICO12 dense projections and head", &dense_pico12);
  ap.flag("--cert-head", "Certified TP=1 greedy head (automatic for PICO12, INGINE_CERT_HEAD=0/1 overrides)", &cert_head);
  ap.flag("--head-nvfp4", "Packed NVFP4 unembedding", &head_nvfp4);
  ap.flag("--kv-fp8", "FP8 full-attention KV caches", &kv_fp8);
  ap.flag("--kv-q4", "Q4_G64 INT4 full-attention KV caches", &kv_q4);
  ap.flag("--kv-12", "Bit-exact KV12 full-attention KV caches", &kv_12);
  ap.flag("--spec", "DSpark speculative decoding", &spec);
  ap.option("--loopcd", "{off,hidden,logits,adaptive,auto}", "LoopCD contrastive decoding (auto uses hidden when thinking is off, default off)", &loopcd,
            {"off", "hidden", "logits", "adaptive"});
  ap.option("--loopcd-omega", "<float>", "LoopCD weight (default 0.3 for hidden/logits/auto, 0.5 for adaptive)", &loopcd_omega);
  ap.option("--loopcd-ref", "<int>", "Reference loop pass (1 or 2, default 1)", &loopcd_ref);
  ap.option("--spec-widths", "{coarse,fine,fine2}", "verify widths: 2/4/8, 1..8 or 2..8", &spec_widths, {"coarse", "fine", "fine2"});
  ap.option("--draft-type", "", "Drafter implementation", &draft_type, {"dspark", "dflash2", "hybrid"});
  ap.flag("--dflash2-adaptive", "Choose verify width using DFlash2 selector confidence", &dflash2_adaptive);
  ap.option("--dflash2-threshold", "", "Adaptive DFlash2 confidence threshold in hybrid mode", &hybrid_dflash2_threshold);
  ap.option("--dflash2-max-ctx", "", "Maximum prompt tokens for DFlash2 in hybrid mode", &dflash2_max_ctx);
  ap.option("--dflash2-dir", "", "DFlash2 checkpoint directory", &dflash2_dir);
  ap.flag("--spec-timing", "Collect per-round draft and verify timing", &spec_timing);
  ap.flag("--draft-nvfp4", "Use packed NVFP4 for the DSpark MLPs", &draft_nvfp4);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--spec-verify-width", "", "Force a width from the selected grid for DFlash2 or TP=2 teacher replay", &spec_verify_width);
  ap.option("--temperature", "", "", &temperature);
  ap.option("--long-prompt", "", "also time prefill of a synthetic prompt of N tokens", &long_prompt);
  ap.option("--forced", "", "teacher-forced replay of the outputs in FILE (JSON lines {name, prompt, output})",
            &forced);
  ap.option("--forced-items", "", "replay at most N items of --forced", &forced_items);
  ap.option("--forced-skip", "", "skip the first N items of --forced", &forced_skip);
  ap.option("--forced-trace", "", "append every replayed item and round to FILE (JSON lines)", &forced_trace);
  ap.option("--forced-free", "", "with --forced: sample freely from each item's prompt with this seed (no replay)", &forced_free);
  ap.parse(argc, argv);

  try {
    ingine::model::validate_dense_formats(dense_fp8, dense_pico12);
    ingine::model::validate_kv_formats(kv_fp8, kv_q4, kv_12);
    ingine::model::RuntimeOptions options;
    ingine::model::set_loopcd_options(options, loopcd,
                                     ap.given("--loopcd-omega") ? std::optional(loopcd_omega) : std::nullopt,
                                     loopcd_ref);
    using Clock = std::chrono::steady_clock;
    cudaFree(nullptr);  // create the context before measuring the model's memory
    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    const auto t0 = Clock::now();
    options.tp = ingine::model::tp_options(ap.given("--tp-size") ? std::optional<int>(tp_size) : std::nullopt,
                                           ap.given("--tp-rank") ? std::optional<int>(tp_rank) : std::nullopt, tp_wire);
    options.tp.profile = tp_profile;
    options.cold_prefill = cold_prefill;
    if (ap.given("--prefill-chunk")) options.prefill_chunk = prefill_chunk;
    else if (options.tp.size == 2) options.prefill_chunk = 2048;
    if (options.tp.size == 2 && temperature != 0)
      throw std::invalid_argument("TP=2 benchmark requires --temperature 0");
    options.model_dir = model_dir;
    options.ablation = ablation;
    options.max_ctx = max_ctx;
    options.dense_fp8 = dense_fp8;
    options.dense_pico12 = dense_pico12;
    options.head_nvfp4 = head_nvfp4;
    options.cert_head = cert_head;
    options.kv_fp8 = kv_fp8;
    options.kv_q4 = kv_q4;
    options.kv_12 = kv_12;
    options.spec = spec;
    options.spec_threshold = float(spec_threshold);
    options.spec_widths = ingine::model::parse_spec_widths(
        draft_type == "hybrid" && !ap.given("--spec-widths") ? "fine2" : spec_widths);
    options.draft_nvfp4 = draft_nvfp4;
    options.spec_trace = spec_timing;
    options.draft_type = draft_type;
    options.dflash2_adaptive = dflash2_adaptive;
    options.hybrid_dflash2_threshold = float(hybrid_dflash2_threshold);
    options.dflash2_max_ctx = dflash2_max_ctx;
    options.dflash2_dir = dflash2_dir;
    options.spec_verify_width = spec_verify_width;
    if (spec_verify_width && draft_type != "dflash2" && (forced.empty() || forced_free >= 0))
      throw std::invalid_argument("--spec-verify-width requires teacher-forced replay");
    auto gen = ingine::model::make_generator(options);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    std::printf("load+capture %.1fs, GPU %.2f GB\n", load_s, double(free0 > free1 ? free0 - free1 : 0) / 1e9);
    std::fflush(stdout);
    if (!forced.empty()) {
      const auto result = ingine::bench::run_forced(*gen, forced, temperature, forced_skip, forced_items,
                                                    options.tp.rank == 0 ? forced_trace : "", forced_free,
                                                    options.spec_widths);
      std::printf("%s\n", result.c_str());
      return 0;
    }

    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

    std::string json = "{";
    for (const auto& [name, prompt] : kPrompts) {
      ingine::text::ChatOptions chat;
      chat.reasoning_effort = effort;
      const nlohmann::ordered_json messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompt}}});
      const std::vector<int32_t> ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
      const auto normalized_effort = ingine::text::normalize_effort(chat.reasoning_effort);
      std::vector<double> rates, prefill_seconds, prefill_rates;
      ingine::model::GenerateStats st;
      ingine::model::DecodeTiming timing;
      for (int r = 0; r < runs; ++r) {
        std::vector<int32_t> out;
        ingine::model::Sampling sampling;
        sampling.temperature = float(temperature);
        sampling.seed = uint64_t(r);
        sampling.thinking_enabled = normalized_effort != "none" && normalized_effort != 0.0;
        // every run stops at the end-of-turn token: tokens after it are not what a request generates (and
        // speculative decoding accepts few drafts there, which made later runs look slow)
        st = gen->generate(ids, tokens, sampling, std::span<const int32_t>(stop),
                           [&](std::span<const int32_t> t) {
                             out.insert(out.end(), t.begin(), t.end());
                             return true;
                           });
        timing.steps += st.decode_timing.steps;
        timing.step_ms += st.decode_timing.step_ms;
        timing.reduce_ms += st.decode_timing.reduce_ms;
        timing.head_exchange_ms += st.decode_timing.head_exchange_ms;
        prefill_seconds.push_back(st.prefill_seconds);
        prefill_rates.push_back((st.prompt_tokens - st.reused_tokens) / st.prefill_seconds);
        rates.push_back((st.generated_tokens - 1) / st.decode_seconds);
        if (r == 0) {
          out.resize(std::min<size_t>(out.size(), 120));
          std::printf("--- %s: %s\n", name.c_str(), py_repr(tok.decode_lossy(out)).c_str());
          std::fflush(stdout);
        }
      }
      const double med = median(rates);
      std::string runs_json, runs_text;
      for (double x : rates) {
        runs_json += (runs_json.empty() ? "" : ", ") + py_float(x);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s%.1f", runs_text.empty() ? "" : ", ", x);
        runs_text += buf;
      }
      const std::string spec_json =
          spec ? "{\"rounds\": " + std::to_string(st.spec_rounds) + ", \"accepted\": " + std::to_string(st.spec_accepted) + "}"
               : "{}";
      if (json.size() > 1) json += ", ";
      json += "\"" + name + "\": {\"decode_tok_s\": " + py_float(med) +
              ", \"prefill_seconds\": " + py_float(median(prefill_seconds)) +
              ", \"prefill_tok_s\": " + py_float(median(prefill_rates)) + ", \"runs\": [" + runs_json +
              "], \"spec\": " + spec_json + "}";
      std::string spec_text;
      if (spec && st.spec_rounds) {
        char buf[96];
        std::snprintf(buf, sizeof buf, ", %.2f tokens/verify (last run)",
                      double(st.spec_accepted + st.spec_rounds) / st.spec_rounds);
        spec_text = buf;
      }
      std::printf("%s: decode %.1f tok/s (runs %s)%s\n", name.c_str(), med, runs_text.c_str(), spec_text.c_str());
      if (!st.spec_trace.empty()) {
        double draft_ms = 0, verify_ms = 0;
        for (const auto& round : st.spec_trace) {
          draft_ms += round.draft_ms;
          verify_ms += round.verify_ms;
        }
        const auto n = st.spec_trace.size();
        std::printf("%s: mean draft %.6f ms, mean verify %.6f ms, %zu rounds (last run)\n",
                    name.c_str(), draft_ms / n, verify_ms / n, n);
      }
      if (timing.steps) {
        const double compute = timing.step_ms - timing.reduce_ms - timing.head_exchange_ms;
        std::printf("%s: %d steps, graph %.6f ms, compute %.6f ms, 96 SUMs %.6f ms, head exchange %.6f ms\n",
                    name.c_str(), timing.steps, timing.step_ms / timing.steps, compute / timing.steps,
                    timing.reduce_ms / timing.steps, timing.head_exchange_ms / timing.steps);
      }
      std::fflush(stdout);
    }
    if (long_prompt) {
      std::ifstream in(__FILE__, std::ios::binary);
      std::stringstream source;
      source << in.rdbuf();
      const std::vector<int32_t> once = tok.encode(source.str(), /*special=*/true);
      std::vector<int32_t> ids;
      for (int i = 0; i < 1000 && int(ids.size()) < long_prompt; ++i) ids.insert(ids.end(), once.begin(), once.end());
      ids.resize(std::min<size_t>(ids.size(), size_t(long_prompt)));
      ingine::model::Sampling greedy;
      greedy.temperature = 0.0f;
      const ingine::model::GenerateStats st =
          gen->generate(ids, 1, greedy, {}, [](std::span<const int32_t>) { return true; });
      const double tok_s = double(ids.size()) / st.prefill_seconds;
      json += ", \"prefill\": {\"tokens\": " + std::to_string(ids.size()) + ", \"tok_s\": " + py_float(tok_s) + "}";
      std::printf("prefill %zu tokens: %.0f tok/s\n", ids.size(), tok_s);
      std::fflush(stdout);
    }
    std::printf("%s}\n", json.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-bench: %s\n", e.what());
    return 1;
  }
}
