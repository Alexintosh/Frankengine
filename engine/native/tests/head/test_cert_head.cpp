#include "common/test_util.h"
#include "core/safetensors.h"
#include "kernels/common/pico12.cuh"
#include "kernels/head/cert_head.h"
#include "kernels/head/head.h"
#include "model/tp_layout.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <string_view>

using namespace ingine::test;
namespace dense = ingine::kernels::dense;
namespace head = ingine::kernels::head;
namespace pico12 = ingine::kernels::pico12;

namespace {

constexpr int H = ingine::model::HIDDEN;
constexpr int Width = 8;

struct Stream {
  cudaStream_t handle;
  Stream() { INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(handle); }
};

struct Graph {
  cudaGraphExec_t exec = nullptr;

  template <class Launch>
  Graph(cudaStream_t stream, Launch launch) {
    cudaGraph_t graph;
    INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
    INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    INGINE_TEST_CUDA(cudaGraphDestroy(graph));
  }
  ~Graph() { cudaGraphExecDestroy(exec); }

  void run(cudaStream_t stream) const { INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream)); }
};

template <class Launch>
float time_ms(cudaStream_t stream, Launch launch, int iterations = 1) {
  cudaEvent_t start, stop;
  INGINE_TEST_CUDA(cudaEventCreate(&start));
  INGINE_TEST_CUDA(cudaEventCreate(&stop));
  INGINE_TEST_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) launch();
  INGINE_TEST_CUDA(cudaEventRecord(stop, stream));
  INGINE_TEST_CUDA(cudaEventSynchronize(stop));
  float ms;
  INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, start, stop));
  INGINE_TEST_CUDA(cudaEventDestroy(start));
  INGINE_TEST_CUDA(cudaEventDestroy(stop));
  return ms / iterations;
}

std::vector<__nv_bfloat16> random_weights(int rows) {
  Rng rng(182);
  std::vector<__nv_bfloat16> w(size_t(rows) * H);
  for (auto& v : w) v = float_to_bf16(float(rng.normal() * 0.03));
  return w;
}

std::vector<__nv_bfloat16> normalized_states() {
  Rng rng(917);
  std::vector<__nv_bfloat16> x(Width * H);
  // Correlated, heavy-tailed states with the scale of an RMS-normalized hidden row.
  std::vector<double> basis(H);
  for (auto& v : basis) v = rng.normal();
  for (int m = 0; m < Width; ++m) {
    std::vector<double> row(H);
    double squares = 0;
    for (int k = 0; k < H; ++k) {
      row[k] = (basis[k] * 0.65 + rng.normal() * 0.35) * (k % 113 == 0 ? 7 : 1);
      squares += row[k] * row[k];
    }
    for (int k = 0; k < H; ++k)
      x[m * H + k] = float_to_bf16(float(row[k] * std::sqrt(H / squares)));
  }
  return x;
}

std::vector<__nv_bfloat16> read_weights(const char* dir) {
  const ingine::SafetensorsModel checkpoint(dir);
  const auto ref = checkpoint.at("model.llm.unembed.weight");
  const int rows = ingine::model::UNPADDED_VOCAB;
  INGINE_CHECK(ref->dtype == "BF16" && ref->shape.size() == 2 && ref->shape[0] >= rows && ref->shape[1] == H);
  std::vector<__nv_bfloat16> w(size_t(rows) * H);
  const auto req = ingine::upload_request(ref, w.data());
  ref.file->read_range(req.offset, w.size() * sizeof(w[0]), w.data());
  return w;
}

std::vector<__nv_bfloat16> read_hidden(const char* path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<__nv_bfloat16> x(Width * H);
  INGINE_CHECK(bool(input.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(x[0]))));
  INGINE_CHECK(input.peek() == std::char_traits<char>::eof());
  return x;
}

struct Harness {
  int rows;
  cudaStream_t stream;
  DeviceArray<__nv_bfloat16> exact_bits;
  pico12::Packed packed;
  DeviceArray<uint8_t> pico, codes, scales, workspace;
  DeviceArray<__nv_bfloat16> x{Width * H};
  DeviceArray<float> bounds, logits;
  DeviceArray<int32_t> reference{Width}, certified{Width};
  DeviceArray<unsigned> suspects{Width};
  head::CertifiedHead head;
  float calibration_ms;

  Harness(const std::vector<__nv_bfloat16>& weights, cudaStream_t s)
      : rows(int(weights.size() / H)), stream(s), exact_bits(weights),
        packed(pico12::pack_pico12_host(weights.data(), rows, H)), pico(packed.bytes),
        codes(weights.size() / 2), scales(weights.size() / 16),
        workspace(head::cert_head_workspace(rows)), bounds(rows), logits(size_t(Width) * rows) {
    INGINE_TEST_CUDA(cudaMemcpyAsync(pico.get(), packed.data.get(), packed.bytes, cudaMemcpyHostToDevice, stream));
    const float outer = head::nvfp4_outer_scale(exact_bits.get(), weights.size(), stream);
    head::nvfp4_pack_rows(exact_bits.get(), rows, outer, codes.get(), scales.get(), stream);
    head = {{codes.get(), scales.get(), outer, rows, 0}, bounds.get()};
    calibration_ms = time_ms(stream, [&] {
      head::cert_head_bounds(exact_bits.get(), head.proxy, bounds.get(), stream);
    });
  }

  dense::Matrix matrix(bool use_pico, bool bf16_order = false) const {
    const void* w = use_pico ? static_cast<const void*>(pico.get()) : exact_bits.get();
    return {w, nullptr, rows, H, use_pico ? &packed.layout : nullptr, bf16_order};
  }

  unsigned compare(int m, dense::Matrix weights, unsigned limit = 8192, int winner = -1) {
    const auto candidate = [&] {
      head::cert_head(x.get(), m, weights, head, logits.get(), workspace.get(), certified.get(),
                      stream, suspects.get(), limit);
    };
    const auto check = [&] {
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      const auto want = reference.download(), got = certified.download();
      for (int token = 0; token < m; ++token) {
        if (want[token] != got[token])
          std::fprintf(stderr, "M=%d token=%d PICO12=%d limit=%u expected=%d actual=%d\n",
                       m, token, bool(weights.pico12), limit, want[token], got[token]);
        INGINE_CHECK(want[token] == got[token]);
        if (winner >= 0) INGINE_CHECK(got[token] == winner);
      }
      return suspects.download();
    };
    dense::head(x.get(), m, weights, logits.get(), false, stream);
    head::argmax(logits.get(), m, rows, rows, reference.get(), stream);
    candidate();
    const auto expected_counts = check();
    const Graph graph(stream, candidate);
    graph.run(stream);
    graph.run(stream);
    INGINE_CHECK(check() == expected_counts);
    return expected_counts[0];
  }

  void bench(int m, dense::Matrix weights) {
    const Graph baseline(stream, [&] {
      dense::head(x.get(), m, weights, logits.get(), false, stream);
      head::argmax(logits.get(), m, rows, rows, reference.get(), stream);
    });
    const Graph candidate(stream, [&] {
      head::cert_head(x.get(), m, weights, head, logits.get(), workspace.get(), certified.get(),
                      stream, suspects.get());
    });
    for (int i = 0; i < 4; ++i) {
      baseline.run(stream);
      candidate.run(stream);
    }
    const float before = time_ms(stream, [&] { baseline.run(stream); }, 30);
    const float after = time_ms(stream, [&] { candidate.run(stream); }, 30);
    const float control = time_ms(stream, [&] { baseline.run(stream); }, 30);
    const auto want = reference.download(), got = certified.download();
    INGINE_CHECK(std::equal(want.begin(), want.begin() + m, got.begin()));
    const auto counts = suspects.download();
    std::printf("%s M=%d rows=%d full/cert/full=%.3f/%.3f/%.3f ms suspects=",
                weights.pico12 ? "PICO12" : "BF16", m, rows, before, after, control);
    for (int i = 0; i < m; ++i) std::printf("%s%u", i ? "," : "", counts[i]);
    std::printf("\n");
  }
};

void test_adversarial(cudaStream_t stream) {
  constexpr int rows = 277;
  constexpr int Winner = 109, Tie = rows - 1;
  std::vector<__nv_bfloat16> w(size_t(rows) * H, float_to_bf16(0));
  // Every proxy row is identical. The exact winner differs by a small weight that
  // NVFP4 rounds to zero. An equivalent higher-ID row is in another tile.
  for (int r = 0; r < rows; ++r) w[size_t(r) * H] = float_to_bf16(1);
  w[Winner * H + 1] = w[Tie * H + 1] = float_to_bf16(0x1p-7f);
  w[Tie * H + 2] = float_to_bf16(0x1p-6f);
  std::vector<__nv_bfloat16> x(Width * H, float_to_bf16(0));
  for (int m = 0; m < Width; ++m) {
    x[m * H] = float_to_bf16(1);
    x[m * H + 1] = float_to_bf16(1);
  }
  Harness h(w, stream);
  h.x.upload(x);
  for (bool packed : {false, true}) {
    const auto weight = h.matrix(packed);
    for (int m = 1; m <= Width; ++m) {
      h.compare(m, weight, 8192, Winner);
      INGINE_CHECK(h.compare(m, weight, 0, Winner) > 0);
    }

    // The last row of a partial tile can win.
    for (int m = 0; m < Width; ++m) x[m * H + 2] = float_to_bf16(1);
    h.x.upload(x);
    h.compare(Width, weight, 8192, Tie);

    // Lower-index rows win when tanh saturates, even though their raw dots are smaller.
    for (int m = 0; m < Width; ++m) {
      x[m * H] = float_to_bf16(512);
      x[m * H + 2] = float_to_bf16(0);
    }
    h.x.upload(x);
    h.compare(Width, weight, 8192, 0);

    // Nonfinite and exceptionally large states take the full reference path for those tokens.
    x[0] = float_to_bf16(std::numeric_limits<float>::quiet_NaN());
    x[H] = float_to_bf16(0x1p80f);
    h.x.upload(x);
    h.compare(Width, weight);
    for (int m = 0; m < Width; ++m) x[m * H] = float_to_bf16(1);
    h.x.upload(x);
  }
}

void test_states(cudaStream_t stream, const std::vector<__nv_bfloat16>& input) {
  auto weights = random_weights(4099);
  for (int exponent = 105; exponent < 127; ++exponent)
    weights[exponent - 105] = float_to_bf16(std::ldexp(1.f, exponent - 127));
  constexpr int Winner = 2048;
  // An aligned row separates the winner from randomly oriented rows. This exercises
  // pruning as well as the near-tie and fallback paths covered above.
  for (int k = 0; k < H; ++k)
    weights[Winner * H + k] = float_to_bf16(0.02f * bf16_to_float(input[k]));
  Harness h(weights, stream);
  INGINE_CHECK(h.packed.layout.escapes);
  h.x.upload(input);
  for (int m = 1; m <= Width; ++m) {
    h.compare(m, h.matrix(false));
    h.compare(m, h.matrix(true));
    h.compare(m, h.matrix(true, true));
  }
  INGINE_CHECK(h.compare(1, h.matrix(true), 8192, Winner) < unsigned(h.rows));
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false;
  const char* model_dir = nullptr;
  const char* hidden = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--bench") bench = true;
    else if (arg == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (arg == "--hidden" && i + 1 < argc) hidden = argv[++i];
    else {
      std::fprintf(stderr, "usage: %s [--bench] [--model-dir PATH] [--hidden BF16_8x2304_FILE]\n", argv[0]);
      return 2;
    }
  }
  if (skip_without_gpu()) return kSkip;
  Stream stream;
  const auto synthetic = normalized_states();
  test_adversarial(stream.handle);
  test_states(stream.handle, synthetic);
  if (bench || model_dir || hidden) {
    const auto input = hidden ? read_hidden(hidden) : synthetic;
    const auto w = model_dir ? read_weights(model_dir) : random_weights(ingine::model::UNPADDED_VOCAB);
    Harness h(w, stream.handle);
    h.x.upload(input);
    std::printf("NVFP4 bound calibration %.3f ms (%d rows)\n", h.calibration_ms, h.rows);
    for (bool packed : {false, true})
      for (int m = 1; m <= Width; ++m)
        if (bench) h.bench(m, h.matrix(packed));
        else h.compare(m, h.matrix(packed));
  }
  std::puts("certified head IDs match the full dense head");
}
