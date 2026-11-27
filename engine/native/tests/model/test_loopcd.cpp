#include "common/test_util.h"
#include "model/batch.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
namespace dense = ingine::kernels::dense;

namespace ingine::model {

struct LoopCdTestAccess {
  // Run a separate model whose visit plan jumps directly from pass 1 (or 2) to the ordinary coda.
  static void shorter_decode(Decoder& decoder, Buffers& buffers, const std::int64_t* pos,
                             cudaStream_t stream, int passes) {
    decoder.decode_impl(buffers, pos, false, stream, TpReduction::Reconstruct, nullptr, false, passes);
  }
};

}  // namespace ingine::model

namespace {

template <class T>
std::vector<T> read_device(const T* ptr, std::size_t elements, cudaStream_t stream) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  std::vector<T> out(elements);
  INGINE_TEST_CUDA(cudaMemcpy(out.data(), ptr, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return out;
}

template <class T>
DeviceBuffer upload(std::span<const T> values) {
  DeviceBuffer buffer(values.size_bytes());
  buffer.copy_from_host(values.data(), values.size_bytes());
  return buffer;
}

int argmax(std::span<const float> row) {
  return int(std::max_element(row.begin(), row.end()) - row.begin());
}

void compare_exactly(std::string_view name, std::span<const float> actual, std::span<const float> expected,
                     double tolerance) {
  INGINE_CHECK(actual.size() == expected.size());
  double max_diff = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    INGINE_CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
    max_diff = std::max(max_diff, std::abs(double(actual[i]) - expected[i]));
  }
  std::printf("%.*s max diff %.8g (limit %.8g)\n", int(name.size()), name.data(), max_diff, tolerance);
  INGINE_CHECK(max_diff <= tolerance);
  if (tolerance == 0.) INGINE_CHECK(std::memcmp(actual.data(), expected.data(), actual.size_bytes()) == 0);
}

void compare_distributions(const char* name, std::span<const float> a, std::span<const float> b,
                           double max_kl = 1.5e-2, bool strict_top1 = true) {
  // The existing prefill and batch GEMMs have different BF16 rounding points than solo GEMV.
  // Check the same next token on decisive rows and bound the distribution change even on near ties.
  const double ma = a[argmax(a)], mb = b[argmax(b)];
  double za = 0., zb = 0.;
  for (std::size_t i = 0; i < a.size(); ++i) {
    za += std::exp(a[i] - ma);
    zb += std::exp(b[i] - mb);
  }
  double kl = 0.;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
    kl += std::exp(la) * (la - lb);
  }
  std::printf("%s KL %.8g (limit %.8g), top-1 %d / %d\n", name, kl, max_kl, argmax(a), argmax(b));
  INGINE_CHECK(kl <= max_kl);
  if (strict_top1) INGINE_CHECK(argmax(a) == argmax(b) || ma - a[argmax(b)] < 0.5);
}

std::vector<float> oracle(std::span<const float> main, std::span<const float> ref, LoopCdMode mode, float omega) {
  double weight = omega;
  if (mode == LoopCdMode::Adaptive) {
    auto sorted = std::vector<float>(main.begin(), main.end());
    std::nth_element(sorted.begin(), sorted.begin() + 1, sorted.end(), std::greater<>());
    const double first = sorted[0], second = sorted[1];
    double z = 0.;
    for (float x : main) z += std::exp(x - first);
    weight *= std::max(0., 1. - (1. - std::exp(second - first)) / z);
  }
  std::vector<float> out(main.size());
  for (std::size_t i = 0; i < main.size(); ++i) out[i] = float(main[i] + weight * (double(main[i]) - ref[i]));
  return out;
}

void test_options() {
  RuntimeOptions options;
  INGINE_CHECK(options.loopcd_mode == LoopCdMode::Off && options.loopcd_omega == 0.f && options.loopcd_ref == 1);
  set_loopcd_options(options, "hidden", std::nullopt, 2);
  INGINE_CHECK(loopcd_config(options).hidden() && options.loopcd_omega == 0.3f && options.loopcd_ref == 2);
  set_loopcd_options(options, "logits", std::nullopt, 1);
  INGINE_CHECK(loopcd_config(options).uses_ref_coda() && options.loopcd_omega == 0.3f);
  set_loopcd_options(options, "adaptive", std::nullopt, 1);
  INGINE_CHECK(loopcd_config(options).uses_ref_coda() && options.loopcd_omega == 0.5f);
  set_loopcd_options(options, "auto", std::nullopt, 1);
  const auto automatic = loopcd_config(options);
  INGINE_CHECK(automatic.auto_mode() && !automatic.uses_ref_coda() && automatic.ref_idx == 0);
  INGINE_CHECK(automatic.visits() == NUM_VISITS && automatic.full_visits() == NUM_FULL_VISITS &&
               automatic.ring_visits() == NUM_SLIDING_VISITS);
  INGINE_CHECK(automatic.effective_omega(Sampling{}.thinking_enabled) == 0.f);
  INGINE_CHECK(automatic.effective_omega(false) == 0.3f);
  validate_loopcd_tp(automatic, 2);
  set_loopcd_options(options, "auto", 0., 2);
  INGINE_CHECK(!loopcd_config(options).auto_mode());
  set_loopcd_options(options, "adaptive", 0., 1);
  INGINE_CHECK(loopcd_config(options).mode == LoopCdMode::Off);
  set_loopcd_options(options, "off", 4., 9);
  INGINE_CHECK(loopcd_config(options).mode == LoopCdMode::Off && options.loopcd_omega == 0.f);
  for (double bad : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    bool threw = false;
    try {
      set_loopcd_options(options, "logits", bad, 1);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    INGINE_CHECK(threw);
  }
  bool threw = false;
  try {
    set_loopcd_options(options, "hidden", 1., 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  INGINE_CHECK(threw);
  threw = false;
  try {
    validate_loopcd_tp({LoopCdMode::Logits, 0.3f, 0}, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  INGINE_CHECK(threw);
  validate_loopcd_tp({LoopCdMode::Hidden, 0.3f, 0}, 2);
  for (int v = 0; v < CODA_VISITS; ++v) {
    INGINE_CHECK(weight_visit(NUM_VISITS + v) == CODA_FIRST + v);
    INGINE_CHECK(ring_cache_index(NUM_VISITS + v) == (v < REF_RING_VISITS ? NUM_SLIDING_VISITS + v : -1));
    INGINE_CHECK(full_cache_index(NUM_VISITS + v) == (v < REF_RING_VISITS ? -1 : NUM_FULL_VISITS));
  }
}

void test_kernels(cudaStream_t stream) {
  constexpr int n = 3 * HIDDEN;
  constexpr float omega = 0.6f;
  Rng rng(226);
  auto random = [&](int count, double lo, double hi) {
    std::vector<float> values(count);
    for (float& x : values) x = float(rng.uniform(lo, hi));
    return values;
  };
  const auto h = random(n, -4., 4.), hr = random(n, -4., 4.), mlp = random(n, -4., 4.);
  const auto depth = random(n, -12., 12.), dr = random(n, -12., 12.);
  auto dh = upload<float>(h), dhr = upload<float>(hr), dd = upload<float>(depth), ddr = upload<float>(dr);
  const auto dmlp = upload<float>(mlp);
  dense::loopcd_hidden(static_cast<const float*>(dh.p), static_cast<const float*>(dmlp.p),
                       static_cast<const float*>(dhr.p), static_cast<float*>(dh.p), static_cast<float*>(dd.p),
                       static_cast<const float*>(ddr.p), n, omega, stream);
  const auto h_out = read_device(static_cast<float*>(dh.p), n, stream);
  const auto d_out = read_device(static_cast<float*>(dd.p), n, stream);
  for (int i = 0; i < n; ++i) {
    const float h_final = h[i] + mlp[i];
    INGINE_CHECK(h_out[i] == std::fma(omega, h_final - hr[i], h_final));
    INGINE_CHECK(d_out[i] == std::fma(omega, depth[i] - dr[i], depth[i]));
  }

  const std::array<float, 3> weights{0.f, omega, 0.f};
  auto device_weights = upload<float>(weights);
  dh = upload<float>(h);
  dd = upload<float>(depth);
  dense::loopcd_hidden_rows(static_cast<const float*>(dh.p), static_cast<const float*>(dmlp.p),
                            static_cast<const float*>(dhr.p), static_cast<float*>(dh.p),
                            static_cast<float*>(dd.p), static_cast<const float*>(ddr.p), 3, HIDDEN,
                            static_cast<const float*>(device_weights.p), stream);
  const auto rows_hidden = read_device(static_cast<const float*>(dh.p), n, stream);
  const auto rows_depth = read_device(static_cast<const float*>(dd.p), n, stream);
  for (int i = 0; i < n; ++i) {
    const bool on = weights[i / HIDDEN] != 0.f;
    INGINE_CHECK(rows_hidden[i] == (on ? h_out[i] : h[i] + mlp[i]));
    INGINE_CHECK(rows_depth[i] == (on ? d_out[i] : depth[i]));
  }

  constexpr int vocab = UNPADDED_VOCAB, ld = vocab + 6, rows = 6;
  auto main = random(rows * ld, -16., 6.);
  const auto ref = random(rows * ld, -7., 5.);
  std::fill_n(main.data() + ld, vocab, -30.f);
  main[ld + 13] = 30.f;  // peaked
  std::fill_n(main.data() + 2 * ld, vocab, 2.f);  // uniform
  std::fill_n(main.data() + 3 * ld, vocab, -30.f);
  main[3 * ld + 31] = main[3 * ld + 96] = 26.f;  // exact tie, across warps
  std::fill_n(main.data() + 4 * ld, vocab, -30.f);
  main[4 * ld + 1] = main[4 * ld + 257] = 26.f;  // exact tie, same lane
  std::fill_n(main.data() + 5 * ld, vocab, -30.f);
  main[5 * ld + 31] = 6.f;
  main[5 * ld + 80] = 5.2f;  // a nontrivial multiplier between zero and omega
  const auto dref = upload<float>(ref);

  for (const auto mode : {LoopCdMode::Logits, LoopCdMode::Adaptive}) {
    auto dmain = upload<float>(main);
    dense::loopcd_combine_logits(static_cast<float*>(dmain.p), static_cast<const float*>(dref.p),
                                rows, ld, vocab, mode, omega, stream);
    const auto actual = read_device(static_cast<float*>(dmain.p), main.size(), stream);
    for (int r = 0; r < rows; ++r) {
      const auto ref_row = std::span(ref).subspan(r * ld, vocab);
      const auto main_row = std::span(main).subspan(r * ld, vocab);
      compare_exactly("kernel combination", std::span(actual).subspan(r * ld, vocab),
                      oracle(main_row, ref_row, mode, omega), 1e-5);
      INGINE_CHECK(std::equal(actual.begin() + r * ld + vocab, actual.begin() + (r + 1) * ld,
                              main.begin() + r * ld + vocab));
    }
  }
  auto raw = random(2 * ld, -100.f, 100.f);
  auto draw = upload<float>(raw);
  dense::loopcd_softcap_logits(static_cast<float*>(draw.p), 2, ld, vocab, stream);
  const auto capped = read_device(static_cast<float*>(draw.p), raw.size(), stream);
  for (int r = 0; r < 2; ++r) {
    for (int v = 0; v < vocab; ++v) raw[r * ld + v] = 30.f * std::tanh(round_bf16(raw[r * ld + v]) / 30.f);
  }
  compare_exactly("batched head softcap including padding", capped, raw, 1e-5);
}

constexpr int kCtx = WINDOW + 16;
constexpr std::array<std::int32_t, 6> kTokens{1, 9906, 19570, 11, 198, 31373};
constexpr std::int32_t kOther = 1234;

std::vector<float> stepwise(const Weights& w, const Ablation& abl, RuntimeOptions options, cudaStream_t stream,
                            int passes = PASSES, std::vector<float>* reference = nullptr,
                            std::vector<std::uint16_t>* taps = nullptr) {
  const auto config = loopcd_config(options);
  auto st = make_state(kCtx, true, TpLayout{}, false, false, config);
  Decoder dec(w, abl, st, false);
  Buffers b = make_buffers(1, TpLayout{}, false, false, false, config.uses_ref_coda());
  DeviceBuffer pos(sizeof(std::int64_t));
  for (int p = 0; p < int(kTokens.size()); ++p) {
    set_step_inputs(kTokens.data() + p, 1, p, ptr<std::int64_t>(b.token), static_cast<std::int64_t*>(pos.p), stream);
    if (passes == PASSES) dec.decode(b, static_cast<std::int64_t*>(pos.p), false, stream);
    else LoopCdTestAccess::shorter_decode(dec, b, static_cast<std::int64_t*>(pos.p), stream, passes);
  }
  if (reference) *reference = read_device(ptr<float>(b.ref_logits), UNPADDED_VOCAB, stream);
  if (taps) *taps = read_device(ptr<std::uint16_t>(b.taps), NUM_TAPS * HIDDEN, stream);
  return read_device(ptr<float>(b.logits), UNPADDED_VOCAB, stream);
}

void test_paths(const Weights& w, const Ablation& abl, const LoopCdConfig config,
                std::span<const float> solo, cudaStream_t stream) {
  constexpr int length = int(kTokens.size());
  auto st = make_state(kCtx, true, TpLayout{}, false, false, config);
  Prefiller prefill(w, abl, length, stream, nullptr, config);
  auto tokens = upload<std::int32_t>(kTokens);
  DeviceBuffer logits(UNPADDED_VOCAB * sizeof(float));
  auto* out = static_cast<float*>(logits.p);
  const auto* ids = static_cast<const std::int32_t*>(tokens.p);
  prefill.run(st, ids, length, 0, out, nullptr);
  const auto whole = read_device(out, UNPADDED_VOCAB, stream);
  reset_state(st, stream);
  prefill.run(st, ids, 2, 0, nullptr, nullptr, {}, nullptr, false);
  prefill.run(st, ids + 2, length - 2, 2, out, nullptr);
  const auto chunked = read_device(out, UNPADDED_VOCAB, stream);
  // Prefiller T=6 vs T=2+4 already exhibits ~0.04 KL on baseline Off due to MoE MMA rounding at
  // tiny T; contrastive scaling amplifies that pre-existing prefill diff by ~(1+omega)^2 (~0.15).
  compare_distributions("whole vs chunked", whole, chunked, 0.20, false);
  compare_distributions("whole vs stepwise solo", whole, solo, 0.25, false);

  auto bstate = make_batch_state(2, kCtx, true, TpLayout{}, false, false, config);
  BatchDecoder batch(w, abl, bstate, 2, stream, 4);
  INGINE_CHECK(!config.uses_ref_coda() || !batch.reject_enabled());
  batch.keep_logits(true);
  std::array<std::int32_t, 2> slots{1, 0};
  std::array<std::int64_t, 2> positions{};
  auto get_batch_logits = [&](int row) {
    auto data = read_device(batch.logits() + std::size_t(row) * batch.kLogitsLd, UNPADDED_VOCAB, stream);
    if (batch.logits_raw()) {
      for (float& value : data) value = 30.f * std::tanh(round_bf16(value) / 30.f);
    }
    return data;
  };
  for (int p = 0; p < length - 2; ++p) {
    positions = {p, p};
    const std::int32_t rows[] = {kTokens[p], kTokens[p]};
    batch.step(2, rows, slots.data(), positions.data(), 0.f, 42);
  }
  // Reject two rows in the first slot and accept all rows in the other one, with the same committed prefix.
  const std::int32_t widths[] = {3, 2};
  const std::int32_t proposed[] = {kTokens[length - 2], kOther, kOther + 1,
                                   kTokens[length - 2], kTokens[length - 1]};
  positions = {length - 2, length - 2};
  const float temps[] = {0.f, 0.f};
  batch.step(2, widths, proposed, slots.data(), positions.data(), temps, 42);
  const auto verified = get_batch_logits(4);
  const std::int32_t kept[] = {1, 2};
  batch.accept(kept);
  positions = {length - 1, length - 1};
  const auto next = batch.step(1, &kTokens.back(), slots.data(), positions.data(), 0.f, 42);
  const auto after_accept = get_batch_logits(0);
  INGINE_CHECK(next[0] == argmax(after_accept));
  compare_distributions("batched verify vs stepwise solo", verified, solo);
  compare_distributions("batch accept and replay vs stepwise solo", after_accept, solo);

  // A verify crossing the physical ring boundary must retain wrapped-out rows still live after acceptance.
  const int ring = config.uses_ref_coda() ? NUM_SLIDING_VISITS : 0;
  const State slot = bstate.view(slots[0]);
  INGINE_CHECK(read_device(slot.ring_pos_at(ring) + 2, 1, stream)[0] == 2);
  positions = {WINDOW, WINDOW};
  batch.step(1, 3, proposed, slots.data(), positions.data(), 0.f, 43);
  batch.accept(kept);
  INGINE_CHECK(read_device(slot.ring_pos_at(ring) + 2, 1, stream)[0] == 2);
}

void test_runtime(Weights weights, Ablation abl, const RuntimeOptions& options, std::span<const float> expected) {
  Runtime rt(options, std::move(weights), std::move(abl));
  const auto s = rt.stream();
  INGINE_CHECK(!rt.has_cert_head());
  reset_state(rt.state(), s);
  const float* logits = nullptr;
  for (int p = 0; p < int(kTokens.size()); ++p) logits = rt.decode_logits(kTokens[p], p);
  auto decoded = read_device(logits, UNPADDED_VOCAB, s);
  compare_exactly("captured vs eager stepwise", decoded, expected, 1e-3);

  const int pos = int(kTokens.size());
  DeviceBuffer checkpoint(rt.checkpoint_bytes()), full_rows(rt.full_rows_bytes(pos));
  rt.save_checkpoint(checkpoint.p);
  rt.save_full_rows(full_rows.p, pos);
  const std::array<std::int32_t, 4> proposal{kOther, kOther + 1, kOther + 2, kOther + 3};
  rt.save_verify(pos, 4);
  rt.set_inputs(proposal, pos);
  rt.run_step(4);
  rt.rollback(pos, 1, 4);

  logits = rt.decode_logits(kOther + 4, pos + 1);
  const auto after_rollback = read_device(logits, UNPADDED_VOCAB, s);
  rt.restore_checkpoint(checkpoint.p);
  const auto ref_cache = rt.state().k_full_at(NUM_FULL_VISITS);
  const std::size_t bytes = pos * ref_cache.nb[2];
  const auto before = read_device(ptr<const std::uint8_t>(ref_cache), bytes, s);
  INGINE_TEST_CUDA(cudaMemsetAsync(ref_cache.data, 0, bytes, s));
  rt.restore_full_rows(full_rows.p, pos);
  INGINE_CHECK(read_device(ptr<const std::uint8_t>(ref_cache), bytes, s) == before);
  rt.decode_logits(kOther, pos);
  logits = rt.decode_logits(kOther + 4, pos + 1);
  compare_distributions("captured solo rollback vs replay", after_rollback, read_device(logits, UNPADDED_VOCAB, s));

  reset_state(rt.state_slot(0), s);
  reset_state(rt.state_slot(1), s);
  const float* batch = nullptr;
  for (int p = 0; p < int(kTokens.size()); ++p) {
    const std::array<Runtime::BatchInput, 2> rows{{{1, kTokens[p]}, {0, kTokens[p]}}};
    batch = ptr<float>(rt.run_batch(rows, true).logits);
  }
  const auto batched = read_device(batch, 2 * UNPADDED_VOCAB, s);
  for (int row = 0; row < 2; ++row) {
    compare_distributions("captured small batch vs solo",
                          std::span(batched).subspan(row * UNPADDED_VOCAB, UNPADDED_VOCAB), expected);
  }
}

// The outer test owns the GPU allocations shared by these independent runtimes.
Weights borrow_weights(const Weights& w) {
  return {w.layers, w.embed, w.embed_norm, w.final_query, w.final_norm, w.unembed, w.unembed_nvfp4,
          w.unembed_certified, w.message_queries, w.expert_formats, w.dense_fp8, w.dense_pico12, {}};
}

void test_auto_runtime(const Weights& weights, RuntimeOptions options) {
  options.max_batch = 2;
  const auto solo = [](Runtime& rt, bool thinking_enabled) {
    reset_state(rt.state(), rt.stream());
    rt.set_loopcd_omega(rt.state().loopcd.effective_omega(thinking_enabled));
    const float* logits = nullptr;
    for (int p = 0; p < int(kTokens.size()); ++p) logits = rt.decode_logits(kTokens[p], p);
    return read_device(logits, UNPADDED_VOCAB, rt.stream());
  };
  const auto batch = [](Runtime& rt, float first, float second) {
    for (int slot : {0, 1}) reset_state(rt.state_slot(slot), rt.stream());
    const float* logits = nullptr;
    for (int32_t token : kTokens) {
      const Runtime::BatchInput rows[]{{1, token, first}, {0, token, second}};
      logits = ptr<float>(rt.run_batch(rows, true).logits);
    }
    return read_device(logits, 2 * UNPADDED_VOCAB, rt.stream());
  };
  const auto fixed = [&](std::string_view mode) {
    set_loopcd_options(options, mode, std::nullopt, 1);
    Runtime rt(options, borrow_weights(weights), load_ablation({}));
    return std::pair(solo(rt, true), batch(rt, 0.f, 0.f));
  };
  const auto [off, batch_off] = fixed("off");
  const auto [hidden, batch_hidden] = fixed("hidden");
  INGINE_CHECK(off != hidden);

  set_loopcd_options(options, "auto", std::nullopt, 1);
  Runtime rt(options, borrow_weights(weights), load_ablation({}));
  for (bool thinking : {true, false, true, false}) {
    compare_exactly("Auto captured solo", solo(rt, thinking), thinking ? off : hidden, 0.);
  }
  const auto mixed = batch(rt, 0.f, options.loopcd_omega);
  compare_exactly("Auto captured plain batch, thinking on", std::span(mixed).first(UNPADDED_VOCAB),
                  std::span(batch_off).first(UNPADDED_VOCAB), 0.);
  compare_exactly("Auto captured plain batch, thinking off", std::span(mixed).subspan(UNPADDED_VOCAB),
                  std::span(batch_hidden).subspan(UNPADDED_VOCAB), 0.);
}

void test_auto_prefill_and_batch(const Weights& w, const Ablation& abl, cudaStream_t stream) {
  constexpr std::size_t vocab = UNPADDED_VOCAB;
  const auto tokens = upload<int32_t>(kTokens);
  const auto prefill = [&](LoopCdConfig config, float omega) {
    auto state = make_state(kCtx, true, TpLayout{}, false, false, config);
    Prefiller prefiller(w, abl, int(kTokens.size()), stream, nullptr, config);
    DeviceBuffer logits(vocab * sizeof(float));
    auto* out = static_cast<float*>(logits.p);
    prefiller.run(state, static_cast<const int32_t*>(tokens.p), int(kTokens.size()), 0, out, nullptr,
                  {}, nullptr, true, omega);
    return read_device(out, vocab, stream);
  };
  const LoopCdConfig off{}, hidden{LoopCdMode::Hidden, 0.3f, 0}, automatic{LoopCdMode::Auto, 0.3f, 0};
  compare_exactly("Auto prefill, thinking on", prefill(automatic, 0.f), prefill(off, -1.f), 0.);
  compare_exactly("Auto prefill, thinking off", prefill(automatic, hidden.omega), prefill(hidden, -1.f), 0.);

  const auto verify = [&](BatchDecoder& decoder, const float* omega) {
    const int32_t slots[] = {0, 1}, widths[] = {2, 1};
    const int32_t input[] = {kTokens[0], kTokens[1], kTokens[0]};
    const int64_t pos[] = {0, 0};
    const float temps[] = {0.f, 0.f};
    decoder.step(2, widths, input, slots, pos, temps, 42, nullptr, omega);
    return read_device(decoder.logits(), 3 * decoder.kLogitsLd, stream);
  };
  const auto fixed = [&](LoopCdConfig config) {
    auto state = make_batch_state(2, kCtx, true, TpLayout{}, false, false, config);
    BatchDecoder decoder(w, abl, state, 2, stream, 8);
    decoder.keep_logits(true);
    return verify(decoder, nullptr);
  };
  const auto expected_off = fixed(off);
  const auto expected_hidden = fixed(hidden);
  auto state = make_batch_state(2, kCtx, true, TpLayout{}, false, false, automatic);
  BatchDecoder decoder(w, abl, state, 2, stream, 8);
  decoder.keep_logits(true);
  for (bool first_on : {true, false, true}) {
    for (int slot : {0, 1}) reset_slot(state, slot, stream);
    const float omegas[] = {first_on ? 0.f : hidden.omega, first_on ? hidden.omega : 0.f};
    const auto mixed = verify(decoder, omegas);
    for (int row = 0; row < 3; ++row) {
      const auto& expected = (row < 2) == first_on ? expected_off : expected_hidden;
      const auto offset = std::size_t(row) * decoder.kLogitsLd;
      compare_exactly("Auto mixed ragged BatchDecoder", std::span(mixed).subspan(offset, vocab),
                      std::span(expected).subspan(offset, vocab), 0.);
    }
  }
}

void test_auto_prefix_cache(const Weights& weights, RuntimeOptions options) {
  set_loopcd_options(options, "auto", std::nullopt, 1);
  options.park_mb = 768;
  options.prefill_chunk = 256;
  options.max_batch = 2;
  auto generator = make_generator(options, borrow_weights(weights), load_ablation({}));
  Sampling sample;
  sample.temperature = 0;
  std::vector<int32_t> prompt(256, kTokens[1]);
  const auto run = [&](bool thinking_enabled) {
    sample.thinking_enabled = thinking_enabled;
    return generator->generate(prompt, 1, sample, {}, [](auto) { return true; });
  };
  INGINE_CHECK(run(true).reused_tokens == 0);
  INGINE_CHECK(run(false).reused_tokens == 0);
  const auto repeat = run(false);
  INGINE_CHECK(repeat.reuse == Reuse::Checkpoint && repeat.reused_tokens == int(prompt.size()));
  const auto parked = run(true);
  INGINE_CHECK(parked.reuse == Reuse::Parked && parked.reused_tokens == int(prompt.size()));

  BatchSampling batch_sample;
  batch_sample.params.temperature = 0;
  std::vector<int32_t> extended{1, 2, 3, 4};
  const auto result = generator->prefill_slot_chunk(1, std::span(extended).first(3), 0, batch_sample);
  INGINE_CHECK(result.offset == 3 && result.token.has_value());
  INGINE_CHECK(generator->reusable_prefix(1, extended) == 3);
  INGINE_CHECK(generator->reusable_prefix(1, extended, false) == 0);
  batch_sample.params.thinking_enabled = false;
  const auto flipped = generator->prefill_slot_chunk(1, extended, 0, batch_sample);
  INGINE_CHECK(flipped.reused_tokens == 0 && flipped.offset == 4);
  extended.push_back(kOther);
  INGINE_CHECK(generator->reusable_prefix(1, extended, false) == 4);
  generator->release_slot(1);
  INGINE_CHECK(generator->reusable_prefix(1, extended, false) == 0);
}

int test_real_model() {
  const char* env = std::getenv("INGINE_MODEL_DIR");
  const std::filesystem::path dir = env && *env ? env : INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "config.json")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return kSkip;
  }
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Weights weights = load_weights(dir, false, stream, true);
  Ablation abl = load_ablation({});
  RuntimeOptions options;
  options.max_ctx = kCtx;
  options.kv_fp8 = true;
  options.dense_fp8 = false;
  options.head_nvfp4 = true;
  std::vector<std::uint16_t> baseline_taps;
  const auto baseline = stepwise(weights, abl, options, stream, PASSES, nullptr, &baseline_taps);
  const auto one_pass = stepwise(weights, abl, options, stream, 1);
  const auto two_pass = stepwise(weights, abl, options, stream, 2);
  for (const auto mode : {LoopCdMode::Hidden, LoopCdMode::Logits, LoopCdMode::Adaptive}) {
    options.loopcd_mode = mode;
    options.loopcd_omega = 0.f;
    compare_exactly("zero omega vs off", stepwise(weights, abl, options, stream), baseline, 0.);
  }
  options.loopcd_mode = LoopCdMode::Off;
  options.loopcd_omega = 42.f;
  compare_exactly("off with custom omega vs default off", stepwise(weights, abl, options, stream), baseline, 0.);

  test_auto_runtime(weights, options);
  test_auto_prefill_and_batch(weights, abl, stream);
  test_auto_prefix_cache(weights, options);

  std::vector<float> adaptive;
  for (const auto mode : {LoopCdMode::Hidden, LoopCdMode::Logits, LoopCdMode::Adaptive}) {
    options.loopcd_mode = mode;
    options.loopcd_omega = mode == LoopCdMode::Adaptive ? 0.6f : 0.5f;
    for (int ref : {1, 2}) {
      std::printf("LoopCD mode %d, reference %d\n", int(mode), ref);
      options.loopcd_ref = ref;
      const auto config = loopcd_config(options);
      std::vector<float> reference;
      std::vector<std::uint16_t> taps;
      auto solo = stepwise(weights, abl, options, stream, PASSES, config.uses_ref_coda() ? &reference : nullptr, &taps);
      if (config.uses_ref_coda()) {
        const auto& independent = ref == 1 ? one_pass : two_pass;
        INGINE_CHECK(taps == baseline_taps);
        compare_exactly("reference coda vs separate shorter forward", reference, independent, 1e-3);
        compare_exactly("real logits vs closed form", solo, oracle(baseline, independent, mode, options.loopcd_omega), 1e-3);
      } else {
        INGINE_CHECK(solo != baseline);
      }
      if (ref == 1) test_paths(weights, abl, config, solo, stream);
      if (mode == LoopCdMode::Adaptive && ref == 1) adaptive = std::move(solo);
    }
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  options.loopcd_ref = 1;
  options.spec = true;
  options.max_batch = 2;
  test_runtime(std::move(weights), std::move(abl), options, adaptive);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  test_options();
  if (skip_without_gpu()) return kSkip;
  if (argc == 2 && std::string_view(argv[1]) == "--real-model") return test_real_model();
  test_kernels(nullptr);
  return 0;
}
