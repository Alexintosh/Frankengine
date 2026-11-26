// Parity and microbenchmark harness for BatchDFlash2 vs solo DFlash2 on real target taps.
#include "core/arena.h"
#include "core/device.h"
#include "model/batch_dflash2.h"
#include "model/dflash2.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace m = ingine::model;
constexpr int H = m::HIDDEN;
constexpr int G = m::BatchDFlash2::GAMMA;
constexpr std::size_t TAP_ROW = std::size_t(m::NUM_TAPS) * H;

struct FetchOut {
  std::vector<std::int32_t> tokens;
  std::vector<float> conf;
};

FetchOut fetch_drafts(const std::int32_t* tok, const float* conf, int n, cudaStream_t s) {
  FetchOut out{std::vector<std::int32_t>(std::size_t(n) * G), std::vector<float>(std::size_t(n) * G)};
  CUDA_CHECK(cudaMemcpyAsync(out.tokens.data(), tok, out.tokens.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(out.conf.data(), conf, out.conf.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return out;
}

struct PromptTaps {
  std::vector<std::int32_t> ids;
  std::int32_t first_tok = 0;
  ingine::DeviceBuffer taps;  // [ids.size(), 4, H] bf16
};

PromptTaps run_prefill_taps(const m::Weights& w, const m::Ablation& abl, m::Prefiller& pf,
                            const std::vector<std::int32_t>& ids, int chunk, cudaStream_t s) {
  PromptTaps res;
  res.ids = ids;
  const int P = int(ids.size());
  res.taps = ingine::DeviceBuffer(std::size_t(P) * TAP_ROW * 2);
  m::State st = m::make_state(P + 16, /*kv_fp8=*/true);
  m::reset_state(st, s);
  ingine::DeviceBuffer dev_ids(std::size_t(chunk) * 4);
  ingine::DeviceBuffer logits(std::size_t(m::UNPADDED_VOCAB) * 4);
  for (int p = 0; p < P; p += chunk) {
    const int T = std::min(chunk, P - p);
    CUDA_CHECK(cudaMemcpyAsync(dev_ids.p, ids.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    auto* dst = static_cast<__nv_bfloat16*>(res.taps.p) + std::size_t(p) * TAP_ROW;
    ingine::Tensor taps_t = m::rm(dst, ingine::DType::BF16, {T, m::NUM_TAPS, H});
    pf.run(st, static_cast<const std::int32_t*>(dev_ids.p), T, p, static_cast<float*>(logits.p), &taps_t, {}, nullptr,
           p + T == P);
  }
  std::vector<float> h_logits(m::UNPADDED_VOCAB);
  CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), logits.p, h_logits.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  int best = 0;
  for (int v = 1; v < m::UNPADDED_VOCAB; ++v) {
    if (h_logits[v] > h_logits[best]) best = v;
  }
  res.first_tok = best;
  return res;
}

bool test_config(const char* label, const std::filesystem::path& dflash2_dir, const m::Weights& w,
                 const std::vector<PromptTaps>& prompts, const PromptTaps& long_prompt,
                 m::DFlash2& solo, m::BatchDFlash2::Precision prec, bool kv_fp8, cudaStream_t s) {
  std::printf("\n=== Testing BatchDFlash2 [%s] (prec=%s, kv_fp8=%d) ===\n", label,
              prec == m::BatchDFlash2::Precision::MXFP8 ? "MXFP8" : "BF16", int(kv_fp8));
  const int B = int(prompts.size());
  m::BatchDFlash2 batch(dflash2_dir, w, 32, 32, s, prec, kv_fp8);
  std::printf("  device_bytes (32 slots): %.3f GB\n", double(batch.device_bytes()) / 1e9);

  // 1. n=1 vs solo DFlash2 across all B prompts + long wrapped prompt
  int solo_matched_tokens = 0, solo_total_tokens = 0, solo_exact_rounds = 0;
  float max_solo_conf_diff = 0.f;
  std::vector<FetchOut> single_batch_outs(B);
  for (int b = 0; b < B; ++b) {
    solo.reset();
    solo.commit(static_cast<const __nv_bfloat16*>(prompts[b].taps.p), int(prompts[b].ids.size()), 0, false);
    const int* solo_tok_dev = solo.propose(prompts[b].first_tok);
    const FetchOut fs = fetch_drafts(solo_tok_dev, solo.confidence(), 1, s);

    batch.reset_slot(b);
    const std::int32_t sl = b, cnt = int(prompts[b].ids.size());
    const std::int64_t st0 = 0, pos = cnt;
    batch.commit(1, &sl, &st0, &cnt, static_cast<const __nv_bfloat16*>(prompts[b].taps.p));
    const auto d = batch.propose(1, &sl, &prompts[b].first_tok, &pos);
    const FetchOut fb = fetch_drafts(d.tokens, d.conf, 1, s);
    single_batch_outs[b] = fb;

    int match = 0;
    for (int j = 0; j < G; ++j) {
      if (fs.tokens[j] == fb.tokens[j]) ++match;
      max_solo_conf_diff = std::max(max_solo_conf_diff, std::fabs(fs.conf[j] - fb.conf[j]));
    }
    solo_matched_tokens += match;
    solo_total_tokens += G;
    if (match == G) ++solo_exact_rounds;
    if (match < G) {
      std::printf("  [diff] prompt %d: matched %d/7 (solo vs batch)\n", b, match);
      std::printf("    solo : ");
      for (int j = 0; j < G; ++j) std::printf("%d ", fs.tokens[j]);
      std::printf("\n    batch: ");
      for (int j = 0; j < G; ++j) std::printf("%d ", fb.tokens[j]);
      std::printf("\n");
    }
  }
  std::printf("  [1] n=1 vs solo DFlash2: %d/%d tokens matched (%d/%d exact 7/7 rounds), max |conf diff|=%.5f\n",
              solo_matched_tokens, solo_total_tokens, solo_exact_rounds, B, double(max_solo_conf_diff));

  // 2. n=4 and n=8 batched propose vs n=1 per-slot propose (MUST be bit-exact 7/7 per slot!)
  bool batch_parity_ok = true;
  for (int n : {4, 8}) {
    std::vector<std::int32_t> sls(n), ancs(n);
    std::vector<std::int64_t> poss(n);
    for (int i = 0; i < n; ++i) {
      sls[i] = i;
      ancs[i] = prompts[i].first_tok;
      poss[i] = int(prompts[i].ids.size());
    }
    const auto d = batch.propose(n, sls.data(), ancs.data(), poss.data());
    const FetchOut fb = fetch_drafts(d.tokens, d.conf, n, s);
    int matched = 0;
    float max_cd = 0.f;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < G; ++j) {
        if (fb.tokens[i * G + j] == single_batch_outs[i].tokens[j]) ++matched;
        max_cd = std::max(max_cd, std::fabs(fb.conf[i * G + j] - single_batch_outs[i].conf[j]));
      }
    }
    std::printf("  [2] n=%d batched vs n=1 per-slot: %d/%d matched, max |conf diff|=%.6g\n", n, matched, n * G,
                double(max_cd));
    if (matched != n * G) batch_parity_ok = false;
  }

  // 3. clone_slot parity (unwrapped slot 0 -> slot 9, and wrapped long_prompt > 2048 slot 10 -> slot 11)
  bool clone_ok = true;
  {
    const int len0 = int(prompts[0].ids.size());
    batch.reset_slot(9);
    batch.clone_slot(0, 9, len0);
    const std::int32_t sl9 = 9;
    const std::int64_t pos0 = len0;
    const auto d9 = batch.propose(1, &sl9, &prompts[0].first_tok, &pos0);
    const FetchOut f9 = fetch_drafts(d9.tokens, d9.conf, 1, s);
    int m0 = 0;
    float cd0 = 0.f;
    for (int j = 0; j < G; ++j) {
      if (f9.tokens[j] == single_batch_outs[0].tokens[j]) ++m0;
      cd0 = std::max(cd0, std::fabs(f9.conf[j] - single_batch_outs[0].conf[j]));
    }
    std::printf("  [3a] clone_slot(0 -> 9, unwrapped len=%d): %d/7 matched, max |conf diff|=%.6g\n", len0, m0,
                double(cd0));
    if (m0 != G) clone_ok = false;
  }
  {
    const int lenL = int(long_prompt.ids.size());
    solo.reset();
    solo.commit(static_cast<const __nv_bfloat16*>(long_prompt.taps.p), lenL, 0, false);
    const FetchOut fsolo = fetch_drafts(solo.propose(long_prompt.first_tok), solo.confidence(), 1, s);

    batch.reset_slot(10);
    const std::int32_t sl10 = 10, sl11 = 11, cntL = lenL;
    const std::int64_t st0 = 0, posL = lenL;
    batch.commit(1, &sl10, &st0, &cntL, static_cast<const __nv_bfloat16*>(long_prompt.taps.p));
    const auto d10 = batch.propose(1, &sl10, &long_prompt.first_tok, &posL);
    const FetchOut f10 = fetch_drafts(d10.tokens, d10.conf, 1, s);

    batch.reset_slot(11);
    batch.clone_slot(10, 11, lenL);
    const auto d11 = batch.propose(1, &sl11, &long_prompt.first_tok, &posL);
    const FetchOut f11 = fetch_drafts(d11.tokens, d11.conf, 1, s);

    int m_solo = 0, m_clone = 0;
    for (int j = 0; j < G; ++j) {
      if (f10.tokens[j] == fsolo.tokens[j]) ++m_solo;
      if (f11.tokens[j] == f10.tokens[j]) ++m_clone;
    }
    std::printf("  [3b] wrapped ring (len=%d > W=2048): vs solo=%d/7, clone_slot(10 -> 11)=%d/7\n", lenL, m_solo,
                m_clone);
    if (m_clone != G) clone_ok = false;
  }

  // 4. Microbench propose & commit latencies at c=1, 4, 8, 16, 32
  std::printf("  [4] Microbench (pos=512 and pos=2048):\n");
  for (int ctx_len : {512, 2048}) {
    ingine::DeviceBuffer dummy_taps(std::size_t(ctx_len) * TAP_ROW * 2);
    CUDA_CHECK(cudaMemsetAsync(dummy_taps.p, 0, dummy_taps.bytes, s));
    for (int sl = 0; sl < 32; ++sl) {
      batch.reset_slot(sl);
      const std::int32_t s32 = sl, c32 = ctx_len;
      const std::int64_t st0 = 0;
      batch.commit(1, &s32, &st0, &c32, static_cast<const __nv_bfloat16*>(dummy_taps.p));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::printf("    ctx=%4d:", ctx_len);
    for (int n : {1, 4, 8, 16, 32}) {
      std::vector<std::int32_t> sls(n), ancs(n, prompts[0].first_tok);
      std::vector<std::int64_t> poss(n, ctx_len);
      for (int i = 0; i < n; ++i) sls[i] = i;
      // warmup
      for (int w_it = 0; w_it < 3; ++w_it) batch.propose(n, sls.data(), ancs.data(), poss.data());
      CUDA_CHECK(cudaStreamSynchronize(s));
      cudaEvent_t e0, e1;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      constexpr int ITERS = 10;
      CUDA_CHECK(cudaEventRecord(e0, s));
      for (int it = 0; it < ITERS; ++it) batch.propose(n, sls.data(), ancs.data(), poss.data());
      CUDA_CHECK(cudaEventRecord(e1, s));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      std::printf("  c=%d: %.2f ms", n, double(ms) / ITERS);
    }
    std::printf("\n");
  }

  return batch_parity_ok && clone_ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
  std::string dflash2_dir = std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini-DFlash2";
  std::string prompts_file = std::string(INGINE_REPO_DIR) + "/bench/prompts-short.json";

  ingine::serve::ArgParser ap("ingine-batch-dflash2-check", "Parity and timing check for BatchDFlash2 vs solo DFlash2.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--dflash2-dir", "", "", &dflash2_dir);
  ap.option("--prompts", "", "", &prompts_file);
  ap.parse(argc, argv);

  try {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    std::vector<std::string> prompt_strs;
    {
      std::ifstream f(prompts_file);
      prompt_strs = nlohmann::json::parse(f).get<std::vector<std::string>>();
    }
    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    m::Weights w = m::load_weights(model_dir, /*dense_fp8=*/false, s, /*head_nvfp4=*/true);
    m::Ablation abl = m::load_ablation({});
    m::Prefiller pf(w, abl, 512, s);

    constexpr int B = 8;
    std::vector<PromptTaps> prompts;
    prompts.reserve(B);
    for (int b = 0; b < B; ++b) {
      ingine::text::ChatOptions chat;
      chat.reasoning_effort = "none";
      const nlohmann::ordered_json msgs =
          nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompt_strs[std::size_t(b)]}}});
      const auto ids = fmt.render(msgs, nlohmann::ordered_json(), chat).ids;
      prompts.push_back(run_prefill_taps(w, abl, pf, ids, 512, s));
    }

    // Build a >2048-token prompt to test sliding window wraparound (W=2048)
    std::string long_text;
    for (int rep = 0; rep < 170; ++rep) {
      long_text += prompt_strs[std::size_t(rep % prompt_strs.size())] + " ";
    }
    ingine::text::ChatOptions chat;
    chat.reasoning_effort = "none";
    const auto long_ids =
        fmt.render(nlohmann::ordered_json::array({{{"role", "user"}, {"content", long_text}}}), nlohmann::ordered_json(),
                   chat)
            .ids;
    std::printf("Prefilled %d short prompts + 1 long wrapped prompt (%zu tokens)\n", B, long_ids.size());
    PromptTaps long_prompt = run_prefill_taps(w, abl, pf, long_ids, 512, s);

    // Solo DFlash2 reference
    m::DFlash2 solo(dflash2_dir, s, /*nvfp4=*/false);
    solo.bind_head(w.unembed_nvfp4, static_cast<const __nv_bfloat16*>(w.embed.data),
                   static_cast<const float*>(w.embed_norm.data));

    bool ok = true;
    ok &= test_config("BF16 + KV_BF16", dflash2_dir, w, prompts, long_prompt, solo, m::BatchDFlash2::Precision::BF16,
                      false, s);
    ok &= test_config("BF16 + KV_FP8", dflash2_dir, w, prompts, long_prompt, solo, m::BatchDFlash2::Precision::BF16,
                      true, s);
    ok &= test_config("MXFP8 + KV_FP8", dflash2_dir, w, prompts, long_prompt, solo, m::BatchDFlash2::Precision::MXFP8,
                      true, s);

    CUDA_CHECK(cudaStreamDestroy(s));
    if (!ok) {
      std::fprintf(stderr, "PARITY CHECK FAILED\n");
      return 1;
    }
    std::printf("\nALL PARITY CHECKS PASSED!\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-batch-dflash2-check: %s\n", e.what());
    return 1;
  }
}
