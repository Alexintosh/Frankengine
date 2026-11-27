#include "common/test_util.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "kernels/draft/dflash2_selector.h"
#include "kernels/draft/draft.h"
#include "kernels/head/head.h"
#include "model/batch_dflash2.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <numeric>
#include <span>

// Internal BatchDFlash2 selector entry points. Synthetic rows let this test
// inject missing candidates without loading the draft or target checkpoints.
namespace ingine::model::batch_dflash2_detail {
std::size_t selector_lattice_elements(int total_rows);
void launch_selector(int n, int total_rows, const std::int32_t* ids, const float* unary,
                     const __nv_bfloat16* projected, const __nv_bfloat16* pred, const __nv_bfloat16* succ,
                     const std::int32_t* anchors, const std::int32_t* row_map, const std::int32_t* prop_off,
                     const std::int32_t* prop_cnt, const float* temps, const std::uint64_t* seed, bool cumulative,
                     float* lattice, std::int32_t* tokens, float* confidence, kernels::draft::DFlash2Support support,
                     cudaStream_t stream);
}  // namespace ingine::model::batch_dflash2_detail

namespace {
namespace kd = ingine::kernels::draft;
namespace kh = ingine::kernels::head;
using namespace ingine::test;
using Bf = __nv_bfloat16;
constexpr int G = kd::GAMMA, K = kd::DFLASH2_SUPPORT;

template <class T>
struct DeviceArray {
  ingine::DeviceBuffer storage;
  T* const data;
  explicit DeviceArray(std::size_t n) : storage(n * sizeof(T)), data(static_cast<T*>(storage.p)) {}
  void upload(std::span<const T> values) { storage.copy_from_host(values.data(), values.size_bytes()); }
  auto download() const {
    std::vector<T> values(storage.bytes / sizeof(T));
    storage.copy_to_host(values.data(), storage.bytes);
    return values;
  }
};

void selector_parity() {
  constexpr int V = 61, R = 256;
  Rng rng(527);
  DeviceArray<int> ids(G * K), anchor(1), legacy_tokens(G), fused_tokens(G);
  DeviceArray<float> unary(G * K), lattice(G * K * K), legacy_conf(G), fused_conf(G), temp(1);
  DeviceArray<Bf> proj(G * R), pred(V * R), succ(V * R);
  DeviceArray<std::uint64_t> seed(1);
  DeviceArray<int> support_ids(G * K), picked(G), old_ids(G * K), old_picked(G);
  DeviceArray<float> support_probs(G * K), old_probs(G * K);
  const kd::DFlash2Support support{support_ids.data, support_probs.data, picked.data};
  const kd::DFlash2Support old_support{old_ids.data, old_probs.data, old_picked.data};

  for (int trial = 0; trial < 24; ++trial) {
    std::array<int, G * K> candidates;
    for (int edge = 0; edge < G; ++edge) {
      // Consecutive entries are distinct even across zero and the highest ID.
      const int start = int(rng.next() % V);
      for (int j = 0; j < K; ++j) candidates[edge * K + j] = (start + j * 7) % V;
    }
    std::array<float, G * K> logits;
    std::array<Bf, G * R> projected;
    std::vector<Bf> predecessors(V * R), successors(V * R);
    for (auto& x : logits) x = trial ? float(rng.uniform(-9, 9)) : 0.f;
    for (auto& x : projected) x = float_to_bf16(trial ? float(rng.uniform(-2, 2)) : 0.f);
    for (auto& x : predecessors) x = float_to_bf16(float(rng.uniform(-1, 1)));
    for (auto& x : successors) x = float_to_bf16(float(rng.uniform(-1, 1)));
    const int anchor_id = int(rng.next() % V);
    ids.upload(candidates);
    anchor.upload(std::span(&anchor_id, 1));
    unary.upload(logits);
    proj.upload(projected);
    pred.upload(predecessors);
    succ.upload(successors);

    kd::dflash2_greedy_selector(ids.data, unary.data, proj.data, pred.data, succ.data, anchor.data,
                                lattice.data, legacy_tokens.data, nullptr, legacy_conf.data);
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    const auto ref_tokens = legacy_tokens.download();
    const auto ref_conf = legacy_conf.download();
    if (trial == 0)
      for (int i = 0; i < G; ++i) INGINE_CHECK(ref_tokens[i] == candidates[i * K]);
    for (bool cumulative : {false, true}) {
      kd::dflash2_fused_selector(ids.data, unary.data, proj.data, pred.data, succ.data, anchor.data,
                                  fused_tokens.data, fused_conf.data, cumulative, nullptr, nullptr, {}, nullptr);
      INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
      INGINE_CHECK(fused_tokens.download() == ref_tokens);
      const auto conf = fused_conf.download();
      float product = 1.f;
      for (int i = 0; i < G; ++i) {
        product *= ref_conf[i];
        const float expected = cumulative ? product : ref_conf[i];
        INGINE_CHECK(std::bit_cast<std::uint32_t>(conf[i]) == std::bit_cast<std::uint32_t>(expected));
      }
    }

    // Duplicate one ID and ensure its combined probability is represented once.
    candidates[1] = candidates[0];
    ids.upload(candidates);
    const float temperature = trial % 2 ? 0.7f : 1.4f;
    const auto sample_seed = rng.next();
    temp.upload(std::span(&temperature, 1));
    seed.upload(std::span(&sample_seed, 1));
    kd::dflash2_lattice_selector(ids.data, unary.data, proj.data, pred.data, succ.data, anchor.data,
                                 lattice.data, legacy_tokens.data, legacy_conf.data, true,
                                 temp.data, seed.data, old_support, nullptr);
    kd::dflash2_fused_selector(ids.data, unary.data, proj.data, pred.data, succ.data, anchor.data,
                               fused_tokens.data, fused_conf.data, true, temp.data, seed.data, support, nullptr);
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    const auto sampled_ids = support_ids.download();
    const auto sampled_probs = support_probs.download();
    const auto chosen = picked.download();
    const auto tokens = fused_tokens.download();
    const auto confidence = fused_conf.download();
    INGINE_CHECK(sampled_ids == old_ids.download() && sampled_probs == old_probs.download());
    INGINE_CHECK(chosen == old_picked.download() && tokens == legacy_tokens.download());
    INGINE_CHECK(confidence == legacy_conf.download());
    const auto scores = lattice.download();
    float product = 1;
    int previous = 0;
    for (int edge = 0; edge < G; ++edge) {
      const float* row = scores.data() + (edge * K + previous) * K;
      const float largest = *std::max_element(row, row + K);
      std::array<double, K> weights{};
      for (int j = 0; j < K; ++j) {
        const auto* ids = candidates.data() + edge * K;
        const auto first = std::find(ids, ids + j, ids[j]) - ids;
        weights[first] += std::exp(double(row[j] - largest) / temperature);
      }
      const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
      for (int j = 0; j < K; ++j) {
        INGINE_CHECK(sampled_ids[edge * K + j] == candidates[edge * K + j]);
        INGINE_CHECK(std::abs(sampled_probs[edge * K + j] - weights[j] / total) < 2e-7);
      }
      previous = chosen[edge];
      INGINE_CHECK(previous >= 0 && previous < K);
      INGINE_CHECK(tokens[edge] == candidates[edge * K + previous]);
      INGINE_CHECK(sampled_probs[edge * K + previous] > 0.f);
      product *= sampled_probs[edge * K + previous];
      INGINE_CHECK(confidence[edge] == product);
    }
  }
}

void batch_selector_parity() {
  namespace batch = ingine::model::batch_dflash2_detail;
  constexpr int N = 8, P = ingine::model::BatchDFlash2::GAMMA, V = 61, R = 256;
  constexpr int vocabulary = 201024;
  constexpr char toggle[] = "INGINE_BATCH_DFLASH2_FUSED_SELECTOR";
  const std::array<int, N> max_drafts{P, 1, 3, 4, 2, P, 5, 6};
  const std::array<int, N> valid_drafts{P, 1, 0, 2, 2, P, 5, 6};
  const int rows = std::accumulate(max_drafts.begin(), max_drafts.end(), 0);
  std::array<int, N> anchors, offsets;
  std::vector<int> row_map(2 * rows), candidates(rows * K);
  std::vector<float> logits(rows * K);
  std::vector<Bf> projected(rows * R), predecessors(V * R), successors(V * R);
  Rng rng(775);

  int row = 0;
  for (int seq = 0; seq < N; ++seq) {
    offsets[seq] = row;
    anchors[seq] = int(rng.next() % V);
    for (int edge = 0; edge < max_drafts[seq]; ++edge, ++row) {
      row_map[2 * row] = seq;
      row_map[2 * row + 1] = edge;
      const int first = int(rng.next() % V);
      for (int next = 0; next < K; ++next) candidates[row * K + next] = (first + next * 7) % V;
    }
  }
  candidates[offsets[2] * K + 13] = -1;
  candidates[(offsets[3] + valid_drafts[3]) * K + K - 1] = vocabulary;
  for (auto& x : logits) x = float(rng.uniform(-5, 5));
  for (auto& x : projected) x = float_to_bf16(float(rng.uniform(-2, 2)));
  for (auto& x : predecessors) x = float_to_bf16(float(rng.uniform(-1, 1)));
  for (auto& x : successors) x = float_to_bf16(float(rng.uniform(-1, 1)));
  // The first row ties exactly, testing the greedy tie break and making seed
  // changes observable in the sampled graph replays.
  std::fill_n(logits.begin(), K, 0.f);
  std::fill_n(projected.begin(), R, float_to_bf16(0.f));

  DeviceArray<int> ids(rows * K), d_anchors(N), d_map(2 * rows), d_offsets(N), d_counts(N);
  DeviceArray<float> unary(rows * K), temps(N);
  DeviceArray<Bf> proj(rows * R), pred(V * R), succ(V * R);
  DeviceArray<std::uint64_t> seed(1);
  ids.upload(candidates);
  d_anchors.upload(anchors);
  d_map.upload(row_map);
  d_offsets.upload(offsets);
  d_counts.upload(max_drafts);
  unary.upload(logits);
  proj.upload(projected);
  pred.upload(predecessors);
  succ.upload(successors);

  struct Output {
    DeviceArray<int> tokens{N * P}, ids{N * P * K}, picked{N * P};
    DeviceArray<float> conf{N * P}, probs{N * P * K};

    void poison() {
      for (auto* buffer : {&tokens.storage, &ids.storage, &picked.storage, &conf.storage, &probs.storage})
        buffer->fill(0x5a);
    }
  } legacy, fused;
  auto lattice_size = [&](const char* value) {
    INGINE_CHECK((value ? setenv(toggle, value, 1) : unsetenv(toggle)) == 0);
    return batch::selector_lattice_elements(rows);
  };
  const char* original = std::getenv(toggle);
  const std::string saved = original ? original : "";
  const bool had_toggle = original != nullptr;
  DeviceArray<float> legacy_lattice(lattice_size("0")), fused_lattice(lattice_size("1"));
  INGINE_CHECK(legacy_lattice.storage.bytes == std::size_t(rows) * K * K * sizeof(float));
  INGINE_CHECK(fused_lattice.data == nullptr && lattice_size(nullptr) == 0);
  INGINE_CHECK((had_toggle ? setenv(toggle, saved.c_str(), 1) : unsetenv(toggle)) == 0);

  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto same_bits = [](const auto& a, const auto& b) {
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
  };
  for (bool cumulative : {false, true}) {
    auto capture = [&](DeviceArray<float>& lattice, Output& out, std::size_t nodes) {
      ingine::DecodeGraphDefinition definition;
      definition.capture(stream, [&] {
        batch::launch_selector(N, rows, ids.data, unary.data, proj.data, pred.data, succ.data, d_anchors.data,
                               d_map.data, d_offsets.data, d_counts.data, temps.data, seed.data, cumulative,
                               lattice.data, out.tokens.data, out.conf.data, {out.ids.data, out.probs.data, out.picked.data},
                               stream);
      });
      INGINE_CHECK(definition.topology_signature().first == nodes);
      ingine::DecodeGraphExecutable graph;
      graph.instantiate(definition);
      return graph;
    };
    auto legacy_graph = capture(legacy_lattice, legacy, 2);
    auto fused_graph = capture(fused_lattice, fused, 1);

    const std::array<int, 4> expected_first_pick{0, 14, 9, 14};
    for (int replay = 0; replay < int(expected_first_pick.size()); ++replay) {
      std::array<float, N> temperature;
      temperature.fill(replay == 0 ? 0.f : 0.8f);
      if (replay == 3)
        for (int seq = 1; seq < N; seq += 2) temperature[seq] = 0.f;
      const std::uint64_t seed_value = replay == 2 ? 1 : 0;
      temps.upload(temperature);
      seed.upload(std::span(&seed_value, 1));
      legacy.poison();
      fused.poison();
      legacy_graph.launch(stream);
      fused_graph.launch(stream);
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));

      const auto tokens = fused.tokens.download();
      const auto confidence = fused.conf.download();
      const auto support_ids = fused.ids.download();
      const auto probs = fused.probs.download();
      const auto picked = fused.picked.download();
      INGINE_CHECK(tokens == legacy.tokens.download());
      INGINE_CHECK(same_bits(confidence, legacy.conf.download()));
      INGINE_CHECK(support_ids == legacy.ids.download());
      INGINE_CHECK(same_bits(probs, legacy.probs.download()));
      INGINE_CHECK(picked == legacy.picked.download());
      INGINE_CHECK(picked[0] == expected_first_pick[replay]);
      INGINE_CHECK(confidence[0] == 1.f / K);
      for (int next = 0; next < K; ++next) {
        const float expected = replay > 0 ? 1.f / K : float(next == 0);
        INGINE_CHECK(probs[next] == expected);
      }
      for (int seq = 0; seq < N; ++seq) {
        for (int edge = 0; edge < P; ++edge) {
          const int out = seq * P + edge;
          if (edge < valid_drafts[seq]) {
            INGINE_CHECK(confidence[out] > 0.f && picked[out] >= 0 && picked[out] < K);
            INGINE_CHECK(tokens[out] == candidates[(offsets[seq] + edge) * K + picked[out]]);
            continue;
          }
          INGINE_CHECK(tokens[out] == 0 && std::bit_cast<std::uint32_t>(confidence[out]) == 0 && picked[out] == 0);
          for (int next = 0; next < K; ++next) {
            INGINE_CHECK(support_ids[out * K + next] == -1);
            INGINE_CHECK(std::bit_cast<std::uint32_t>(probs[out * K + next]) == 0);
          }
        }
      }
    }
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
}

void acceptance_edges() {
  constexpr int V = 1025;
  DeviceArray<float> p((G + 1) * V), q(G * K), u(G), ws(kd::dflash2_accept_workspace_floats(V)), resid(V);
  DeviceArray<int> ids(G * K), drafts(G), picked(G);
  DeviceArray<kd::AcceptResult> result(1);
  std::array<int, G * K> support;
  std::array<int, G> chosen, tokens;
  std::array<float, G * K> probs;
  std::array<float, G> uniforms;
  std::vector<float> target((G + 1) * V, 0.f);
  for (int i = 0; i < G; ++i) {
    std::iota(support.begin() + i * K, support.begin() + (i + 1) * K, V - K);
    chosen[i] = K - 1;
    tokens[i] = V - 1;
    uniforms[i] = i == 2 ? 0.f : 1.f;
    for (int j = 0; j < K; ++j) target[i * V + V - K + j] = probs[i * K + j] = 1.f / K;
  }
  target[G * V + 7] = 1.f;
  ids.upload(support);
  drafts.upload(tokens);
  u.upload(uniforms);
  auto run = [&](int eligible, int expected) {
    p.upload(target);
    q.upload(probs);
    picked.upload(chosen);
    kd::dflash2_exact_accept(p.data, V, drafts.data, eligible, ids.data, q.data, picked.data,
                             u.data, ws.data, resid.data, result.data, nullptr);
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    INGINE_CHECK(result.download()[0].accepted == expected);
    return resid.download();
  };
  INGINE_CHECK(run(0, 0)[V - 1] == 1.f / K);
  INGINE_CHECK(run(G, G)[7] == 1.f);

  // Reject after two matches. Target mass on and off the sparse support must
  // be normalized together across both residual tiles, including the tail.
  constexpr int row = 2;
  auto* reject = target.data() + row * V;
  std::fill(reject, reject + V, 0.f);
  reject[1] = 0.5f;
  reject[V - 2] = 0.5f;
  const auto r = run(G, row);
  INGINE_CHECK(std::abs(r[1] - 8.f / 15.f) < 1e-7);
  INGINE_CHECK(std::abs(r[V - 2] - 7.f / 15.f) < 1e-7 && r[V - 1] == 0.f);

  // Malformed picked index forces rejection. The zero residual must fall
  // back to the unchanged target row if p == q.
  chosen[0] = -1;
  const auto fallback = run(G, 0);
  INGINE_CHECK(std::equal(fallback.begin(), fallback.end(), target.begin()));
  chosen[0] = K - 1;
  probs[K - 1] = 0.f;
  INGINE_CHECK(run(G, 0)[V - 1] == 1.f);

  // A padded draft token has zero target mass. Its support entry stays out
  // of the residual without indexing beyond the unpadded vocabulary.
  probs[K - 1] = 1.f / K;
  support[K - 1] = tokens[0] = V;
  ids.upload(support);
  drafts.upload(tokens);
  INGINE_CHECK(run(G, 0)[V - 1] == 1.f);
}

void statistical_acceptance() {
  constexpr int V = 24, draws = 50000;
  const kh::SamplingParams params{0.85f, 20, 0.92f};
  std::array<float, 2 * V> logits;
  for (int i = 0; i < 2 * V; ++i) logits[i] = -0.057f * (i % V);
  DeviceArray<float> p(2 * V), input(2 * V), q(K), uniforms(draws),
      ws(kd::dflash2_accept_workspace_floats(V)), residual(draws * V);
  DeviceArray<int> ids(K), drafts(draws), picked(draws), next(draws);
  DeviceArray<kd::AcceptResult> results(draws);
  input.upload(logits);
  kh::sampling_probs(input.data, 2, V, V, params, p.data, nullptr);
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const auto actual = p.download();

  std::array<double, V> expected{};
  for (int i = 0; i < params.top_k; ++i) expected[i] = std::exp(double(logits[i]) / params.temperature);
  const double total = std::accumulate(expected.begin(), expected.end(), 0.0);
  double kept = 0;
  for (double& x : expected) {
    if (kept / total > params.top_p) x = 0;
    kept += x;
  }
  for (int i = 0; i < V; ++i) {
    expected[i] /= kept;
    INGINE_CHECK(std::abs(expected[i] - actual[i]) < 2e-7);
  }
  INGINE_CHECK(expected[0] > 0.0 && expected[V - K] > 0.0 &&
               expected[params.top_k - 1] == 0.0 && expected[V - 1] == 0.0);

  std::array<int, K> support;
  std::array<float, K> support_probs;
  std::array<double, K> cdf;
  std::iota(support.begin(), support.end(), V - K);
  double mass = 0;
  for (int j = 0; j < K; ++j) {
    support_probs[j] = float(j + 1) / (K * (K + 1) / 2);
    cdf[j] = mass += support_probs[j];
  }
  ids.upload(support);
  q.upload(support_probs);
  Rng host(642);
  std::vector<int> proposed(draws), indices(draws);
  for (int i = 0; i < draws; ++i) {
    indices[i] = int(std::upper_bound(cdf.begin(), cdf.end(), host.uniform() * mass) - cdf.begin());
    proposed[i] = support[indices[i]];
  }
  drafts.upload(proposed);
  picked.upload(indices);
  kh::SamplerRng rng{429, 0};
  kh::uniform(draws, rng, uniforms.data, nullptr);
  for (int i = 0; i < draws; ++i) {
    kd::dflash2_exact_accept(p.data, V, drafts.data + i, 1, ids.data, q.data, picked.data + i,
                             uniforms.data + i, ws.data, residual.data + i * V, results.data + i, nullptr);
  }
  kh::sample_from_probs(residual.data, draws, V, V, rng, next.data, nullptr);
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const auto acceptance = results.download();
  const auto resampled = next.download();
  std::array<int, V> histogram{};
  int accepted = 0;
  for (int i = 0; i < draws; ++i) {
    const int token = acceptance[i].accepted ? proposed[i] : resampled[i];
    INGINE_CHECK(token >= 0 && token < V);
    histogram[token]++;
    accepted += acceptance[i].accepted;
  }
  double overlap = 0;
  for (int j = 0; j < K; ++j) overlap += std::min(double(support_probs[j]), expected[support[j]]);
  const auto check_bin = [&](int count, double p) {
    const double mean = draws * p;
    INGINE_CHECK(p == 0.0 ? count == 0 : std::abs(count - mean) <= 5 * std::sqrt(mean * (1 - p)));
  };
  check_bin(accepted, overlap);
  for (int i = 0; i < V; ++i) check_bin(histogram[i], expected[i]);
}
}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  selector_parity();
  batch_selector_parity();
  acceptance_edges();
  statistical_acceptance();
  std::puts("PASS model_dflash2_selector");
}
