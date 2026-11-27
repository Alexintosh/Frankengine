#include "common/test_util.h"
#include "kernels/batch/dense.h"
#include "kernels/common/pico12.cuh"
#include "kernels/dense/dense.h"

#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <numeric>
#include <string_view>
#include <type_traits>

using namespace ingine::test;
namespace batch = ingine::kernels::batch;
namespace dense = ingine::kernels::dense;
namespace pico12 = ingine::kernels::pico12;

namespace {

constexpr int H = 2304;
constexpr std::array kTokens{1, 2, 4, 8};

std::vector<uint16_t> make_band_weights(int rows, int cols, uint64_t seed = 917) {
  Rng rng(seed);
  std::vector<uint16_t> weights(size_t(rows) * cols);
  // A triangular distribution over 14 exponents, grouped into seven upper fields.
  for (uint16_t& bits : weights) {
    const uint64_t sample = rng.next();
    const int e = 110 + ((sample >> 16) & 7) + ((sample >> 19) & 3) + ((sample >> 21) & 3);
    bits = uint16_t((sample & 0x807f) | (e << 7));
  }
  return weights;
}

std::vector<uint16_t> make_weights(int rows, int cols, bool escapes) {
  auto weights = make_band_weights(rows, cols);
  // Eight nonzero upper exponent fields exercise direct decoding with a nonzero code zero.
  weights[16] = 108 << 7;
  if (escapes) {
    for (int row = 0; row < rows; row += 7) {
      uint16_t* w = weights.data() + size_t(row) * cols;
      w[0] = 0x0000;
      w[1] = 0x8000;
      w[2] = 0x8001;
      w[3] = 0x007f;
      // Several escapes in one chunk and in distinct chunks of the same tile, then a row tail.
      w[7] = (124 << 7) | 7;
      w[8] = (125 << 7) | 0x8008;
      w[9] = (126 << 7) | 9;
      w[24] = (124 << 7) | 0x8018;
      w[31] = (125 << 7) | 0x801f;
      w[32] = (126 << 7) | 32;
      w[cols - 1] = (107 << 7) | 1;
    }
  }
  return weights;
}

void test_roundtrip() {
  {
    constexpr int rows = 951, cols = 69;
    std::vector<uint16_t> all(size_t(rows) * cols);
    for (size_t i = 0; i < all.size(); ++i) all[i] = uint16_t(i);
    const auto packed = pico12::pack_pico12_host(all.data(), rows, cols);
    INGINE_CHECK(packed.layout.escapes);
    const auto view = pico12::device_view(packed.layout, packed.data.get());
    INGINE_CHECK(reinterpret_cast<uintptr_t>(view.row_ptr) % alignof(uint32_t) == 0);
    INGINE_CHECK(view.row_ptr[0] == 0 && view.row_ptr[rows] > 0);
    INGINE_CHECK(pico12::unpack_pico12_host(packed) == all);
  }
  {
    const std::array<uint16_t, 8> edge{0x0000, 0x8000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7f81, 0xffc1};
    const auto packed = pico12::pack_pico12_host(edge.data(), 1, edge.size());
    INGINE_CHECK(!packed.layout.escapes);
    INGINE_CHECK(pico12::unpack_pico12_host(packed) == std::vector<uint16_t>(edge.begin(), edge.end()));
  }
  {
    std::vector<uint16_t> wide(32);
    for (int i = 0; i < 8; ++i) {
      const uint16_t high = uint16_t(i * 17) << 8;
      wide[4 * i] = high;
      wide[4 * i + 1] = high | 0x0035;
      wide[4 * i + 2] = high | 0x8080;
      wide[4 * i + 3] = high | 0x80b5;
    }
    const auto packed = pico12::pack_pico12_host(wide.data(), 1, wide.size());
    INGINE_CHECK(!packed.layout.escapes && packed.bytes == 48);
    INGINE_CHECK(pico12::unpack_pico12_host(packed) == wide);
  }
  {
    constexpr int cols = pico12::kCompactCols + 1;
    auto wide = make_band_weights(1, cols);
    wide[0] = 0x8000;
    wide[1] = 0x0001;
    wide.back() = 0xffc7;
    const auto packed = pico12::pack_pico12_host(wide.data(), 1, cols);
    const auto view = pico12::device_view(packed.layout, packed.data.get());
    INGINE_CHECK(packed.layout.escapes && view.row_ptr[1] == 3);
    INGINE_CHECK(view.entries[4] == cols - 1 && view.entries[5] == wide.back());
    INGINE_CHECK(view.num_outlier_rows == 1 && view.row_mask[0] == 1u && view.outlier_rows[0] == 0);
    INGINE_CHECK(pico12::unpack_pico12_host(packed) == wide);
  }
  for (const int cols : {1536, H, 3072, 9216}) {
    for (const bool escapes : {false, true}) {
      // Enough rows to exercise CSR offsets across partition boundaries.
      const int rows = (2 * (1 << 20) + cols - 1) / cols;
      const auto weights = make_weights(rows, cols, escapes);
      const auto packed = pico12::pack_pico12_host(weights.data(), rows, cols);
      INGINE_CHECK(packed.layout.escapes == escapes);
      INGINE_CHECK(pico12::unpack_pico12_host(packed) == weights);
      const size_t dense_bytes = size_t(rows) * cols * 3 / 2;
      if (escapes) {
        const auto view = pico12::device_view(packed.layout, packed.data.get());
        const size_t nnz = size_t((rows + 6) / 7) * 11 + 1;
        INGINE_CHECK(view.row_ptr[rows] == nnz && (view.lut[0] & 0xff) == 0);
        INGINE_CHECK(packed.layout.row_mask_offset == dense_bytes + pico12::row_ptr_bytes(rows) +
                     nnz * sizeof(uint32_t));
        INGINE_CHECK(packed.layout.outlier_rows_offset == packed.layout.row_mask_offset + pico12::row_mask_bytes(rows));
        INGINE_CHECK(view.num_outlier_rows == (rows + 6) / 7);
        INGINE_CHECK(packed.bytes == packed.layout.outlier_rows_offset + size_t(view.num_outlier_rows) * sizeof(int32_t));
        int outlier_row = 0;
        for (int row = 0; row < rows; ++row) {
          const size_t offset = size_t(row) * pico12::row_bytes(cols);
          const uint8_t* bf_low = packed.data.get() + offset;
          const uint8_t* hi4 = bf_low + cols;
          const uint32_t first = view.row_ptr[row], last = view.row_ptr[row + 1];
          INGINE_CHECK(last - first == (row % 7 == 0 ? (row == 0 ? 12u : 11u) : 0u));
          INGINE_CHECK(((view.row_mask[row >> 5] >> (row & 31)) & 1u) == (first < last));
          if (first < last) INGINE_CHECK(view.outlier_rows[outlier_row++] == row);
          for (uint32_t i = first; i < last; ++i) {
            const uint32_t entry = view.entries[i], col = entry & 0xffffu;
            INGINE_CHECK(col < unsigned(cols));
            INGINE_CHECK(i == first || (view.entries[i - 1] & 0xffffu) < col);
            INGINE_CHECK((entry >> 16) == weights[size_t(row) * cols + col]);
            INGINE_CHECK(bf_low[col] == 0 && ((hi4[col / 2] >> (4 * (col % 2))) & 15) == pico12::kEscape);
          }
        }
        INGINE_CHECK(outlier_row == view.num_outlier_rows);
        if (rows & 31) INGINE_CHECK((view.row_mask[rows >> 5] >> (rows & 31)) == 0);
      } else {
        INGINE_CHECK(packed.bytes == dense_bytes);
        const auto& layout = packed.layout;
        const auto view = pico12::device_view(layout, packed.data.get());
        INGINE_CHECK(layout.row_mask_offset == 0 && layout.outlier_rows_offset == 0 && layout.num_outlier_rows == 0);
        INGINE_CHECK(!view.row_mask && !view.outlier_rows && view.num_outlier_rows == 0);
        for (int col = 0; col < cols; ++col)
          INGINE_CHECK(packed.data[col] == uint8_t(weights[col]));
      }
    }
  }
  std::puts("PASS PICO12 host bit-exact round-trip");
}

struct Weights {
  pico12::Packed packed;
  DeviceArray<uint16_t> bf16;
  DeviceArray<uint8_t> blob;

  Weights(const std::vector<uint16_t>& weights, int rows, int cols)
      : packed(pico12::pack_pico12_host(weights.data(), rows, cols)), bf16(weights), blob(packed.bytes) {
    INGINE_TEST_CUDA(cudaMemcpy(blob.get(), packed.data.get(), packed.bytes, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }

  dense::Matrix matrix(bool pico, bool exact = false) const {
    return {pico ? static_cast<const void*>(blob.get()) : bf16.get(), nullptr, packed.layout.rows,
            packed.layout.cols, pico ? &packed.layout : nullptr, exact};
  }
};

pico12::Layout fused_layout(pico12::Layout layout) {
  // Older layouts carry CSR without the deferred epilogue metadata.
  layout.row_mask_offset = layout.outlier_rows_offset = 0;
  layout.num_outlier_rows = 0;
  return layout;
}

void test_device_roundtrip(cudaStream_t stream, bool escapes, int rows = 259, int cols = 288) {
  std::vector<uint16_t> bits(size_t(rows) * cols);
  for (size_t i = 0; i < bits.size(); ++i) {
    const uint16_t raw = uint16_t(i);
    bits[i] = escapes ? raw : uint16_t((raw & 0x807f) | ((106 + ((raw >> 7) & 15)) << 7));
  }
  if (cols > pico12::kCompactCols) {
    bits = make_band_weights(rows, cols);
    for (int row = 0; row < rows; ++row) {
      bits[size_t(row) * cols] = 0x8000;
      bits[size_t(row + 1) * cols - 1] = 0xffc7;
    }
  }
  const Weights w(bits, rows, cols);
  INGINE_CHECK(w.packed.layout.escapes == escapes);
  DeviceArray<__nv_bfloat16> out(bits.size());
  dense::unpack_pico12_rows(w.matrix(true), rows, out.get(), stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto actual = out.download();
  INGINE_CHECK(std::memcmp(actual.data(), bits.data(), bits.size() * sizeof(uint16_t)) == 0);

  constexpr int selected = 4;
  const std::vector<int32_t> ids{rows - 1, 0, rows / 2, 0};
  const DeviceArray<int32_t> device_ids(ids);
  DeviceArray<__nv_bfloat16> range(selected * cols), gathered(selected * cols);
  dense::unpack_pico12_rows(w.matrix(true), selected, range.get(), stream, rows - selected);
  dense::unpack_pico12_rows(w.matrix(true), selected, gathered.get(), stream, 0, device_ids.get());
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto actual_range = range.download(), actual_gathered = gathered.download();
  INGINE_CHECK(std::memcmp(actual_range.data(), bits.data() + (rows - selected) * cols,
                           selected * cols * sizeof(uint16_t)) == 0);
  for (int i = 0; i < selected; ++i)
    INGINE_CHECK(std::memcmp(actual_gathered.data() + i * cols, bits.data() + ids[i] * cols,
                             cols * sizeof(uint16_t)) == 0);
}

std::vector<__nv_bfloat16> activations(int M, int cols, int inputs = 1) {
  Rng rng(19);
  std::vector<__nv_bfloat16> x(size_t(inputs) * M * cols);
  for (auto& v : x) v = float_to_bf16(float(rng.uniform(-1.0, 1.0)));
  return x;
}

enum class Parity { exact, fp32_ulp, fp32, bf16, softcap };

template <class T>
void compare(const char* op, int M, const Weights& weights, const DeviceArray<T>& baseline,
             const DeviceArray<T>& packed, cudaStream_t stream, Parity parity = Parity::exact) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  const auto a = baseline.download(), b = packed.download();
  auto as_float = [](T v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else return bf16_to_float(v);
  };
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0) continue;
    const float ref = as_float(a[i]), got = as_float(b[i]);
    const float delta = std::abs(ref - got), mag = std::max(std::abs(ref), std::abs(got));
    bool matches = false;
    switch (parity) {
      case Parity::exact: break;
      case Parity::fp32_ulp: matches = std::nextafter(ref, got) == got; break;
      case Parity::fp32: matches = delta <= 1e-5f + mag * 4e-6f; break;
      case Parity::bf16: {
        const int ref_bits = int(std::bit_cast<uint32_t>(ref) >> 16);
        const int got_bits = int(std::bit_cast<uint32_t>(got) >> 16);
        // Near cancellation, one rounding step can be much smaller than the FP32 reduction error.
        matches = std::abs(ref_bits - got_bits) <= 1 || delta <= 2e-6f;
        break;
      }
      case Parity::softcap: matches = delta <= 2e-6f + mag / 128.f; break;
    }
    if (matches) continue;
    std::fprintf(stderr, "%s M=%d K=%d escapes=%d differs at output %zu (%.9g vs %.9g)\n",
                 op, M, weights.packed.layout.cols, weights.packed.layout.escapes, i, ref, got);
    std::exit(1);
  }
}

void test_fp32_ulp(cudaStream_t stream, bool escapes) {
  constexpr int cols = 1536;
  auto weights = make_weights(H, cols, escapes);
  // The active columns include both MMA halves and two CSR entries when sparse is enabled.
  for (int row = 0; row < H; ++row) {
    auto* w = weights.data() + size_t(row) * cols;
    const uint16_t sign = row % 2 ? 0x8000 : 0;
    w[7] = (121 << 7) | sign | 5;
    w[8] = ((escapes ? 124 : 122) << 7) | sign | 13;
    w[25] = ((escapes ? 125 : 123) << 7) | 0x8000 | 19;
  }
  const Weights w(weights, H, cols);
  INGINE_CHECK(w.packed.layout.escapes == escapes);
  for (const int M : kTokens) {
    std::vector<__nv_bfloat16> input(size_t(M) * cols, float_to_bf16(0.f));
    for (int m = 0; m < M; ++m) {
      input[size_t(m) * cols + 7] = float_to_bf16((m + 1) / 16.f);
      input[size_t(m) * cols + 8] = float_to_bf16(-0.5f);
      input[size_t(m) * cols + 25] = float_to_bf16(0.25f);
    }
    const DeviceArray<__nv_bfloat16> x(input);
    DeviceArray<float> a(size_t(M) * H), b(a.count);
    dense::proj_raw(x.get(), M, w.matrix(false), a.get(), nullptr, true, stream);
    dense::proj_raw(x.get(), M, w.matrix(true), b.get(), nullptr, true, stream);
    compare("controlled proj_raw FP32", M, w, a, b, stream, Parity::fp32_ulp);
  }
}

void test_gemv_all_patterns(cudaStream_t stream) {
  constexpr std::array columns{0, 7, 8, 15, 16, 31, 64, H - 1};
  constexpr int tokens = columns.size();
  constexpr int nonfinite_count = 2 * 128;
  constexpr int finite_rows = ((1 << 16) - nonfinite_count) / tokens;
  constexpr int rows = finite_rows + nonfinite_count;
  // One-hot activations isolate each finite operand. Give each infinity or NaN its own
  // row so it cannot obscure another nonfinite operand's result.
  std::vector<uint16_t> bits(size_t(rows) * H, 0x3f80);
  int finite = 0, nonfinite = 0;
  for (unsigned raw = 0; raw < (1u << 16); ++raw) {
    if ((raw & 0x7f80) == 0x7f80) {
      bits[size_t(finite_rows + nonfinite) * H + columns[nonfinite % tokens]] = uint16_t(raw);
      ++nonfinite;
    } else {
      bits[size_t(finite / tokens) * H + columns[finite % tokens]] = uint16_t(raw);
      ++finite;
    }
  }
  const Weights w(bits, rows, H);
  INGINE_CHECK(w.packed.layout.escapes);
  std::vector<__nv_bfloat16> one_hot(tokens * H, float_to_bf16(0.f));
  for (int m = 0; m < tokens; ++m) one_hot[m * H + columns[m]] = float_to_bf16(1.f);
  const DeviceArray<__nv_bfloat16> x(one_hot);
  for (const int M : kTokens) {
    DeviceArray<float> a(size_t(M) * rows), b(a.count);
    dense::gemv(x.get(), M, H, w.matrix(false), a.get(), true, stream);
    dense::gemv(x.get(), M, H, w.matrix(true, true), b.get(), true, stream);
    compare("all BF16 patterns in exact GEMV", M, w, a, b, stream);
    dense::gemv(x.get(), M, H, w.matrix(true), b.get(), true, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto ref = a.download(), got = b.download();
    for (size_t i = 0; i < ref.size(); ++i) {
      if (ref[i] == got[i] || (std::isnan(ref[i]) && std::isnan(got[i]))) continue;
      // The packed path handles CSR products on CUDA cores. MMA may flush subnormals.
      constexpr float normal_min = std::numeric_limits<float>::min();
      INGINE_CHECK(std::abs(ref[i]) < normal_min && std::abs(got[i]) < normal_min);
    }
  }
}

void test_batch_head(cudaStream_t stream, bool escapes) {
  constexpr int rows = 258, cols = 128;
  const Weights w(make_weights(rows, cols, escapes), rows, cols);
  for (int tokens : {3, 33}) {
    const DeviceArray<__nv_bfloat16> x(activations(tokens, cols));
    std::vector<float> temperatures(tokens, 0.7f);
    temperatures[0] = 0.f;
    const DeviceArray<float> temps(temperatures);
    const DeviceArray<std::uint64_t> seed(std::vector<std::uint64_t>{98});
    DeviceArray<float> baseline(tokens * rows), packed(tokens * rows);
    DeviceArray<int32_t> ref_token(tokens), packed_token(tokens);
    DeviceArray<int2> partials(batch::head_partials(tokens, rows));
    auto run = [&](dense::Matrix weights, float* logits, int32_t* output) {
      batch::head_sample(x.get(), cols, weights.w, cols, tokens, rows, cols, temps.get(), 0, logits, rows,
                         partials.get(), output, stream, 0, nullptr, seed.get(), weights.pico12);
    };
    run(w.matrix(false), baseline.get(), ref_token.get());
    run(w.matrix(true), packed.get(), packed_token.get());
    compare("batch head", tokens, w, baseline, packed, stream);
    INGINE_CHECK(ref_token.download() == packed_token.download());
  }
}

void test_batch_head_rejection(cudaStream_t stream) {
  constexpr int N = 259, max_rows = 33, K = batch::RejectSpec::kSupport, samples = 8192;
  constexpr float temperature = 0.9f;
  constexpr std::array peak_ids{128, 0, 1, 256, 258};
  constexpr std::array support{128, 0, 1, 2, 3, 7, 8, 15, 16, 121, 127, 129, 153, 191, 244, 255};
  constexpr std::array<float, K> proposal{0.25f, 0.625f, 0.03125f, 0.0078125f, 0.0078125f, 0.0078125f,
                                         0.0078125f, 0.0078125f, 0.0078125f, 0.0078125f, 0.0078125f, 0.0078125f,
                                         0.0078125f, 0.0078125f, 0.0078125f, 0.f};
  std::array<float, N> scores;
  std::array<double, N> target;
  std::array<int, N> bins;
  std::vector<uint16_t> bits(N * H);
  std::vector<uint8_t> codes(N * (H / 2));
  for (int v = 0; v < N; ++v) {
    bins[v] = int(peak_ids.size()) + v % 3;
    for (int i = 0; i < int(peak_ids.size()); ++i)
      if (v == peak_ids[i]) bins[v] = i;
    // One nonzero input makes the dot exactly representable for all three weight formats.
    constexpr std::array dots{4.f, 3.f, 2.f, 3.f, 2.f, -2.f, -3.f, -4.f};
    constexpr std::array<uint8_t, 8> fp4{6, 5, 4, 5, 4, 12, 13, 14};
    const float dot = dots[bins[v]];
    bits[v * H] = std::bit_cast<uint16_t>(float_to_bf16(dot));
    codes[v * (H / 2)] = fp4[bins[v]];
    scores[v] = 30.f * std::tanh(dot / 30.f);
    target[v] = std::exp(double(scores[v]) / temperature);
  }
  const double total = std::accumulate(target.begin(), target.end(), 0.0);
  std::array<double, peak_ids.size() + 3> bin_target{};
  for (int v = 0; v < N; ++v) {
    target[v] /= total;
    bin_target[bins[v]] += target[v];
  }
  INGINE_CHECK(target[support[0]] > proposal[0]);
  INGINE_CHECK(target[support[1]] < proposal[1]);
  INGINE_CHECK(target[support[2]] > proposal[2]);  // A support token has positive residual mass.

  const Weights weights(bits, N, H);
  const DeviceArray<uint8_t> nvfp4_codes(codes), nvfp4_scales(std::vector<uint8_t>(N * (H / 16), 0x38));
  const ingine::kernels::head::Nvfp4Matrix nvfp4{nvfp4_codes.get(), nvfp4_scales.get(), 1.f, N, 0};
  std::vector<__nv_bfloat16> input(max_rows * H, float_to_bf16(0.f));
  for (int row = 0; row < max_rows; ++row) input[row * H] = float_to_bf16(1.f);
  const DeviceArray<__nv_bfloat16> x(input);
  auto check = [&](const char* name, bool use_nvfp4, bool packed, int M, int runs) {
    std::vector<float> temperatures(M, temperature);
    temperatures[1] = 0.85f;
    temperatures[3] = 0.f;
    std::vector<uint8_t> active(M);
    active[0] = active[1] = active[3] = active[4] = 1;
    std::vector<int32_t> support_ids(M * K, -1), picked(M, -1);
    std::vector<float> probabilities(M * K, 0.f);
    std::copy(support.begin(), support.end(), support_ids.begin());
    std::copy(proposal.begin(), proposal.end(), probabilities.begin());
    std::copy_n(support.begin(), 3, support_ids.begin() + K);
    probabilities[K] = 0.25f;
    probabilities[K + 1] = 0.625f;
    probabilities[K + 2] = 0.125f;
    // The final vocab tile is entirely inside this row's support.
    std::iota(support_ids.begin() + 4 * K, support_ids.begin() + 5 * K - 3, 0);
    std::iota(support_ids.begin() + 5 * K - 3, support_ids.begin() + 5 * K, N - 3);
    probabilities[4 * K] = 1.f;
    picked[0] = picked[1] = picked[4] = 0;
    const DeviceArray<float> d_temps(temperatures), d_probs(probabilities);
    const DeviceArray<uint8_t> d_active(active);
    const DeviceArray<int32_t> d_ids(support_ids), d_picked(picked);
    const DeviceArray<uint64_t> d_seed(std::vector<uint64_t>{0});
    const size_t tiled_partials = batch::head_partials(M, N);
    const size_t n_partials = use_nvfp4 ? batch::head_partials_nvfp4(M, N) : tiled_partials;
    const size_t ntiles = (use_nvfp4 && M <= 8 ? n_partials : tiled_partials) / M;
    constexpr float sentinel = -123.f;
    DeviceArray<float> target_scores(std::vector<float>(M * K, sentinel)), partial_lse(2 * n_partials);
    DeviceArray<int2> partials(n_partials), candidates(M);
    DeviceArray<int32_t> out(M);
    const batch::RejectSpec reject{d_active.get(), d_ids.get(), d_probs.get(), d_picked.get(),
                                   target_scores.get(), partial_lse.get()};
    const auto w = weights.matrix(packed);
    auto launch = [&](const batch::RejectSpec* spec, void* cand = nullptr) {
      if (use_nvfp4) {
        batch::head_sample_nvfp4(x.get(), H, nvfp4, M, d_temps.get(), 0, nullptr, 0, partials.get(), out.get(),
                                 stream, cand, d_seed.get(), spec);
      } else {
        batch::head_sample(x.get(), H, w.w, H, M, N, H, d_temps.get(), 0, nullptr, 0, partials.get(), out.get(),
                           stream, 0, cand, d_seed.get(), w.pico12, spec);
      }
    };
    launch(nullptr);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto legacy = out.download();
    const batch::RejectSpec unused{};
    launch(&unused, candidates.get());
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    INGINE_CHECK(out.download() == legacy);
    launch(&reject);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto initial = out.download();
    INGINE_CHECK(initial[0] == support[0] && initial[1] == support[0]);
    INGINE_CHECK(initial[2] == legacy[2] && initial[3] == legacy[3]);
    const auto captured_scores = target_scores.download();
    for (int row = 0; row < M; ++row)
      for (int j = 0; j < K; ++j) {
        const int idx = row * K + j, v = support_ids[idx];
        if (active[row] && temperatures[row] > 0.f && v >= 0)
          INGINE_CHECK(std::abs(captured_scores[idx] - scores[v]) < 1e-6f);
        else
          INGINE_CHECK(captured_scores[idx] == sentinel);
      }
    const auto outside = partial_lse.download();
    INGINE_CHECK(outside[2 * (5 * ntiles - 1)] == -INFINITY && outside[2 * (5 * ntiles - 1) + 1] == 0.f);

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(&reject);
    INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 0;
    INGINE_TEST_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    INGINE_CHECK(nodes == 2);  // One head pass and one row reduction.
    INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    Rng seeds(817), drafts(9103);
    std::vector<uint64_t> seed_values(runs);
    std::vector<int32_t> draft_indices(runs);
    for (int i = 0; i < runs; ++i) {
      seed_values[i] = seeds.next();
      double u = drafts.uniform();
      int j = 0;
      while (u >= proposal[j]) u -= proposal[j++];
      draft_indices[i] = j;
    }
    const DeviceArray<uint64_t> replay_seeds(seed_values);
    const DeviceArray<int32_t> replay_drafts(draft_indices);
    DeviceArray<int32_t> outputs(2 * runs);
    for (int i = 0; i < runs; ++i) {
      INGINE_TEST_CUDA(cudaMemcpyAsync(d_seed.get(), replay_seeds.get() + i, sizeof(uint64_t),
                                       cudaMemcpyDeviceToDevice, stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(d_picked.get(), replay_drafts.get() + i, sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
      INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(outputs.get() + 2 * i, out.get(), 2 * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
    }
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto draws = outputs.download();
    // Replay a prior seed and proposal to check that the graph reads current seed contents deterministically.
    INGINE_TEST_CUDA(cudaMemcpyAsync(d_seed.get(), replay_seeds.get(), sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    INGINE_TEST_CUDA(cudaMemcpyAsync(d_picked.get(), replay_drafts.get(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    INGINE_CHECK(out.download()[0] == draws[0]);
    INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
    INGINE_TEST_CUDA(cudaGraphDestroy(graph));

    std::array<int, N> counts{};
    std::array<int, bin_target.size()> bin_counts{};
    int support_residual_draws = 0;
    for (int i = 0; i < runs; ++i) {
      const int v = draws[2 * i], j = draft_indices[i];
      INGINE_CHECK(v >= 0 && v < N);
      if (target[support[j]] >= proposal[j]) INGINE_CHECK(v == support[j]);
      Rng row_key_rng(seed_values[i] ^ 0x9E3779B97F4A7C15ull);
      Rng acceptance_rng(row_key_rng.next());  // Row zero, draw zero.
      const double u1 = (acceptance_rng.next() >> 40) * 0x1p-24;
      const double decision = u1 * proposal[j] - target[support[j]];
      if (std::abs(decision) > 1e-6) INGINE_CHECK((v == support[j]) == (decision < 0.0));
      if (decision > 1e-6 && v == support[2]) ++support_residual_draws;
      INGINE_CHECK(draws[2 * i + 1] == support[0]);
      ++counts[v];
      ++bin_counts[bins[v]];
    }
    INGINE_CHECK(counts[256] > 0 && counts[0] < runs);
    if (runs == samples) {
      INGINE_CHECK(support_residual_draws > 0);
      auto within_four_sigma = [&](const char* kind, int index, int observed, double p) {
        const double expected = runs * p, sigma = std::sqrt(expected * (1.0 - p));
        if (std::abs(observed - expected) > 4.0 * sigma)
          std::fprintf(stderr, "%s rejection %s %d: observed %d, expected %.2f (sigma %.2f)\n",
                       name, kind, index, observed, expected, sigma);
        INGINE_CHECK(std::abs(observed - expected) <= 4.0 * sigma);
      };
      for (int v = 0; v < N; ++v) within_four_sigma("token", v, counts[v], target[v]);
      for (int b = 0; b < int(bin_target.size()); ++b) within_four_sigma("bin", b, bin_counts[b], bin_target[b]);
    }
    std::printf("PASS %s rejection sampling (M=%d, %d graph replays)\n", name, M, runs);
  };
  check("BF16", false, false, 5, 64);
  check("BF16", false, false, max_rows, samples);
  check("PICO12", false, true, 5, 64);
  check("PICO12", false, true, max_rows, samples);
  check("NVFP4 small", true, false, 8, 64);
  check("NVFP4 tiled", true, false, 9, 64);
  check("NVFP4 tiled", true, false, max_rows, samples);
}

void test_gemv_family(cudaStream_t stream, bool escapes, bool exact) {
  constexpr int rows = 53;
  // Uniform tiles, a mixed tile with equal endpoint inputs and an interior hole, and a partial tail.
  const DeviceArray<int> segments(std::vector<int>{0, 16, 0, 16, 17, 1, 18, 31, 2, 31, 49, 1, 49, rows, 2});
  for (const int cols : {H, 3072, 9216}) {
    auto weights = make_weights(rows, cols, escapes);
    if (escapes) weights[17 * cols + 10] = 125 << 7;  // The projection must skip this outlier row.
    const Weights w(weights, rows, cols);
    const auto packed = w.matrix(true, exact);
    const auto layout = fused_layout(w.packed.layout);
    auto fused = packed;
    fused.pico12 = &layout;
    for (int M = 1; M <= 8; ++M) {
      const DeviceArray<__nv_bfloat16> x(activations(M, cols, cols == H ? 3 : 1));
      DeviceArray<float> a(size_t(M) * rows), b(size_t(M) * rows);
      dense::gemv(x.get(), M, cols, w.matrix(false), a.get(), true, stream);
      dense::gemv(x.get(), M, cols, packed, b.get(), true, stream);
      const bool same_order = exact || (!escapes && M == 1 && cols != 9216);
      compare("gemv", M, w, a, b, stream, same_order ? Parity::exact : Parity::bf16);
      if (escapes && !exact) {
        dense::gemv(x.get(), M, cols, fused, a.get(), true, stream);
        compare("deferred gemv", M, w, a, b, stream);
      }
      if (cols == H) {
        dense::head(x.get(), M, w.matrix(false), a.get(), true, stream);
        dense::head(x.get(), M, packed, b.get(), true, stream);
        compare("head", M, w, a, b, stream, same_order ? Parity::exact : Parity::softcap);
        if (escapes && !exact) {
          dense::head(x.get(), M, fused, a.get(), true, stream);
          compare("deferred head", M, w, a, b, stream);
        }
        constexpr float sentinel = 1.f;
        const std::vector<float> initial(a.count, sentinel);
        a.upload(initial);
        b.upload(initial);
        dense::proj(x.get(), M, w.matrix(false), segments.get(), int(segments.count / 3), a.get(), true, stream);
        dense::proj(x.get(), M, packed, segments.get(), int(segments.count / 3), b.get(), true, stream);
        compare("proj", M, w, a, b, stream, same_order ? Parity::exact : Parity::bf16);
        const auto output = b.download();
        for (int m = 0; m < M; ++m) INGINE_CHECK(output[size_t(m) * rows + 17] == sentinel);
        if (escapes && !exact) {
          a.upload(initial);
          dense::proj(x.get(), M, fused, segments.get(), int(segments.count / 3), a.get(), true, stream);
          compare("deferred proj", M, w, a, b, stream);
        }
      }
    }
  }
}

void test_convolution(cudaStream_t stream, bool escapes, bool exact) {
  Rng rng(103);
  std::vector<float> conv_w(H * 4), state(H * 3);
  for (float& v : conv_w) v = float(rng.uniform(-1.0, 1.0));
  for (float& v : state) v = ingine::test::round_bf16(float(rng.uniform(-1.0, 1.0)));
  const DeviceArray<float> taps(conv_w);
  for (const int cols : {1536, 3072}) {
    const Weights w(make_weights(H, cols, escapes), H, cols);
    const auto packed = w.matrix(true, exact);
    const auto layout = fused_layout(w.packed.layout);
    auto fused = packed;
    fused.pico12 = &layout;
    for (int M = 1; M <= 8; ++M) {
      const DeviceArray<__nv_bfloat16> x(activations(M, cols));
      DeviceArray<float> a(size_t(M) * H), b(size_t(M) * H);
      DeviceArray<__nv_bfloat16> ba(a.count), bb(a.count);
      dense::proj_raw(x.get(), M, w.matrix(false), a.get(), ba.get(), true, stream);
      dense::proj_raw(x.get(), M, packed, b.get(), bb.get(), true, stream);
      const bool same_order = exact || (!escapes && M == 1);
      compare("proj_raw FP32", M, w, a, b, stream, same_order ? Parity::exact : Parity::fp32);
      compare("proj_raw BF16", M, w, ba, bb, stream, same_order ? Parity::exact : Parity::bf16);
      if (escapes && !exact) {
        dense::proj_raw(x.get(), M, fused, a.get(), ba.get(), true, stream);
        compare("deferred proj_raw FP32", M, w, a, b, stream);
        compare("deferred proj_raw BF16", M, w, ba, bb, stream);
        INGINE_TEST_CUDA(cudaMemsetAsync(ba.get(), 0xff, ba.count * sizeof(__nv_bfloat16), stream));
        dense::proj_raw(x.get(), M, packed, nullptr, ba.get(), true, stream);
        compare("proj_raw BF16-only output", M, w, ba, bb, stream);
      }
      if (cols == 3072) {
        DeviceArray<float> sa(state), sb(state), ha(a.count), hb(a.count);
        dense::proj_conv(x.get(), M, w.matrix(false), taps.get(), sa.get(), a.get(), ha.get(), true, stream);
        dense::proj_conv(x.get(), M, packed, taps.get(), sb.get(), b.get(), hb.get(), true, stream);
        compare("proj_conv history", M, w, ha, hb, stream, same_order ? Parity::exact : Parity::bf16);
        // Check convolution and state against the BF16-rounded inputs of each run. A one-step
        // difference in those inputs can be amplified by cancellation in the convolution.
        auto check_conv = [&](const DeviceArray<float>& got, const DeviceArray<float>& history,
                              const DeviceArray<float>& next_state) {
          const auto output = got.download(), hist = history.download(), next = next_state.download();
          for (int row = 0; row < H; ++row) {
            auto value = [&](int frame) {
              if (frame < 3) return state[size_t(frame) * H + row];
              return hist[size_t(frame - 3) * H + row];
            };
            for (int m = 0; m < M; ++m) {
              double expected = value(m + 3);
              for (int j = 0; j < 4; ++j) expected += double(conv_w[4 * row + j]) * value(m + j);
              expected /= std::sqrt(5.0);
              INGINE_CHECK(std::abs(output[size_t(m) * H + row] - expected) <= 2e-6 * (1 + std::abs(expected)));
            }
            for (int j = 0; j < 3; ++j) INGINE_CHECK(next[size_t(j) * H + row] == value(M + j));
          }
        };
        check_conv(a, ha, sa);
        check_conv(b, hb, sb);
      }
    }
  }
}

void test_sparse_graph(cudaStream_t stream) {
  constexpr int rows = 959;
  const Weights w(make_weights(rows, H, true), rows, H);
  const auto layout = fused_layout(w.packed.layout);
  auto fused = w.matrix(true);
  fused.pico12 = &layout;
  const DeviceArray<int> segments(std::vector<int>{0, 512, 0, 512, 768, 1, 768, rows, 2});
  INGINE_CHECK(w.packed.layout.num_outlier_rows > 128);
  for (int M = 1; M <= 8; ++M) {
    auto input = activations(M, H, 3);
    DeviceArray<__nv_bfloat16> source(input);
    DeviceArray<__nv_bfloat16> x(source.count);
    DeviceArray<float> ref(size_t(M) * rows), eager(ref.count), stage(ref.count), replayed(ref.count);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    INGINE_TEST_CUDA(cudaMemcpyAsync(x.get(), source.get(), x.count * sizeof(*x.get()), cudaMemcpyDeviceToDevice, stream));
    dense::proj(x.get(), M, w.matrix(true), segments.get(), 3, stage.get(), true, stream);
    INGINE_TEST_CUDA(cudaMemcpyAsync(replayed.get(), stage.get(), stage.count * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 0;
    INGINE_TEST_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    const char* defer_env = std::getenv("INGINE_PICO12_DEFER_CSR");
    const size_t expected_nodes = (defer_env && std::atoi(defer_env) != 0) ? 4 : 3;
    INGINE_CHECK(nodes == expected_nodes);  // Input copy, GEMV (+ optional outlier kernel), output copy.
    INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 3; ++replay) {
      dense::proj(source.get(), M, fused, segments.get(), 3, ref.get(), false, stream);
      dense::proj(source.get(), M, w.matrix(true), segments.get(), 3, eager.get(), false, stream);
      compare("deferred proj without PDL", M, w, ref, eager, stream);
      INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
      compare("deferred proj PDL graph replay", M, w, ref, replayed, stream);
      for (auto& v : input) v = float_to_bf16(-bf16_to_float(v));
      source.upload(input);
    }
    INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
    INGINE_TEST_CUDA(cudaGraphDestroy(graph));
  }
}

enum class BenchOp { head, wproj, wo };

constexpr size_t kBenchOutlierStride = 5000;

std::vector<uint16_t> make_bench_weights(int rows, int cols, bool escapes, unsigned matrix_index = 0) {
  auto input = make_band_weights(rows, cols, 917 + matrix_index);
  const size_t outlier_count = escapes ? input.size() / kBenchOutlierStride : 0;
  if (outlier_count) {
    // Distribute 0.02% outliers across 30% of rows, with distinct columns in each affected row.
    Rng rng(223 + matrix_index);
    std::vector<int> row_order(rows);
    std::iota(row_order.begin(), row_order.end(), 0);
    const size_t active = std::min(size_t(rows) * 3 / 10, outlier_count);
    for (size_t i = 0; i < active; ++i) {
      std::swap(row_order[i], row_order[i + rng.next() % (rows - i)]);
      const size_t count = outlier_count / active + (i < outlier_count % active);
      uint16_t* row = input.data() + size_t(row_order[i]) * cols;
      for (size_t j = 0; j < count; ++j) {
        const uint64_t sample = rng.next();
        const size_t begin = j * cols / count, end = (j + 1) * cols / count;
        const size_t col = begin + sample % (end - begin);
        const uint16_t exponent = sample >> 63 ? 108 : 124;
        row[col] = (row[col] & 0x807f) | (exponent << 7);
      }
    }
  }
  return input;
}

void test_batch_pico12_gemm(cudaStream_t stream, bool escapes, int rows, int K) {
  const Weights w(escapes ? make_bench_weights(rows, K, true) : make_weights(rows, K, false), rows, K);
  INGINE_CHECK(w.packed.layout.escapes == escapes);
  if (escapes)
    INGINE_CHECK(pico12::device_view(w.packed.layout, w.packed.data.get()).row_ptr[rows] ==
                 size_t(rows) * K / kBenchOutlierStride);
  DeviceArray<__nv_bfloat16> unpacked(size_t(rows) * K);
  dense::unpack_pico12_rows(w.matrix(true), rows, unpacked.get(), stream);
  batch::TunedGemm gemm;
  gemm.set_capturing(true);  // Use the heuristic's algorithm for the numerical reference.
  const int lda = K + 8;
  const DeviceArray<__nv_bfloat16> a(activations(192, lda));
  auto check = [&]<class T>(int M, int first, int N) {
    constexpr bool fp32 = std::is_same_v<T, float>;
    const int ldc = N + 8;
    const std::vector<T> initial(size_t(M) * ldc, T(-29.f));
    DeviceArray<T> ref(initial), got(initial);
    gemm.run(a.get(), lda, unpacked.get() + size_t(first) * K, K, ref.get(), ldc, fp32, M, N, K, stream);
    batch::pico12_gemm(a.get(), lda, w.blob.get(), w.packed.layout, first, got.get(), ldc, fp32, M, N, K, stream);
    compare(fp32 ? "batch PICO12 GEMM FP32" : "batch PICO12 GEMM BF16", M, w, ref, got, stream,
            fp32 ? Parity::fp32 : Parity::bf16);
  };
  for (const int M : {1, 16, 64, 96, 128, 192}) {
    auto check_slice = [&](int first, int count) {
      check.template operator()<float>(M, first, count);
      check.template operator()<__nv_bfloat16>(M, first, count);
    };
    check_slice(0, rows);
    if (K == H) {
      check_slice(6553, 1024);  // Offset by one from the K segment to exercise a crossing row-mask lookup.
      check_slice(rows - 32, 32);
    }
  }
}

void test_batch_pico12_exact_graph(cudaStream_t stream, int K) {
  constexpr int rows = 38, first = 3, N = 34, M = 17, ldc = N + 5;
  const int lda = K + 1;
  auto bits = make_band_weights(rows, K);
  for (int row = 0; row < rows; row += 3) {
    bits[size_t(row) * K + 7] = (108 << 7) | 5;
    bits[size_t(row) * K + K - 1] = (124 << 7) | 0x801d;
  }
  const Weights w(bits, rows, K);
  INGINE_CHECK(w.packed.layout.escapes);
  const std::array columns{0, 7, 8, 15, 16, 31, 32, K - 1};
  std::vector<__nv_bfloat16> input(size_t(M) * lda + 1, float_to_bf16(0.f));
  for (int m = 0; m < M; ++m) input[1 + size_t(m) * lda + columns[m % columns.size()]] = float_to_bf16(1.f);
  const DeviceArray<__nv_bfloat16> a(input);
  auto check = [&]<class T>(T sentinel, const pico12::Layout& layout) {
    constexpr bool fp32 = std::is_same_v<T, float>;
    DeviceArray<T> out(std::vector<T>(size_t(M) * ldc + 1, sentinel));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    batch::pico12_gemm(a.get() + 1, lda, w.blob.get(), layout, first, out.get() + 1, ldc, fp32, M, N, K, stream);
    INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 0;
    INGINE_TEST_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    INGINE_CHECK(nodes == 1);
    INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    const auto actual = out.download();
    INGINE_CHECK(std::memcmp(actual.data(), &sentinel, sizeof(T)) == 0);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < ldc; ++n) {
        T expected = sentinel;
        if (n < N) {
          const uint16_t raw = bits[size_t(first + n) * K + columns[m % columns.size()]];
          if constexpr (fp32) expected = std::bit_cast<float>(uint32_t(raw) << 16);
          else std::memcpy(&expected, &raw, sizeof(raw));
        }
        INGINE_CHECK(std::memcmp(&actual[1 + size_t(m) * ldc + n], &expected, sizeof(T)) == 0);
      }
    }
    INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
    INGINE_TEST_CUDA(cudaGraphDestroy(graph));
  };
  check(-29.f, w.packed.layout);
  check(float_to_bf16(-29.f), fused_layout(w.packed.layout));  // CSR without a row mask.
}

void benchmark_batch_pico12(cudaStream_t stream, const char* name, int N, int K) {
  constexpr int copies = 8, rounds = 12;
  const auto weights = make_bench_weights(N, K, true);
  const auto packed = pico12::pack_pico12_host(weights.data(), N, K);
  const size_t stride = (packed.bytes + 15) / 16 * 16;
  DeviceArray<uint8_t> blobs(copies * stride);
  DeviceArray<__nv_bfloat16> unpacked(size_t(N) * K);
  INGINE_TEST_CUDA(cudaMemcpyAsync(blobs.get(), packed.data.get(), packed.bytes, cudaMemcpyHostToDevice, stream));
  for (int i = 1; i < copies; ++i)
    INGINE_TEST_CUDA(cudaMemcpyAsync(blobs.get() + i * stride, blobs.get(), packed.bytes, cudaMemcpyDeviceToDevice,
                                   stream));
  batch::TunedGemm gemm;
  cudaEvent_t first, last;
  INGINE_TEST_CUDA(cudaEventCreate(&first));
  INGINE_TEST_CUDA(cudaEventCreate(&last));
  std::printf("batch %s %d x %d, %d rotating PICO12 blobs, BF16 output\n", name, N, K, copies);
  for (const int M : {64, 96, 128, 192}) {
    const DeviceArray<__nv_bfloat16> a(activations(M, K));
    DeviceArray<__nv_bfloat16> c(size_t(M) * N);
    auto fused = [&](int i) {
      batch::pico12_gemm(a.get(), K, blobs.get() + i * stride, packed.layout, 0, c.get(), N, false, M, N, K, stream);
    };
    auto baseline = [&](int i) {
      dense::unpack_pico12_rows({blobs.get() + i * stride, nullptr, N, K, &packed.layout}, N, unpacked.get(), stream);
      gemm.run(a.get(), K, unpacked.get(), K, c.get(), N, false, M, N, K, stream);
    };
    gemm.set_capturing(false);
    for (int step = 0; step < batch::TunedGemm::kWarmSteps + batch::TunedGemm::kSamples * (batch::TunedGemm::kCandidates + 4);
         ++step) {
      baseline(step % copies);
      gemm.collect();
    }
    gemm.set_capturing(true);
    auto time = [&](auto run) {
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      INGINE_TEST_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < copies; ++i) run(i);
      INGINE_TEST_CUDA(cudaStreamEndCapture(stream, &graph));
      INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, graph, 0));
      INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
      INGINE_TEST_CUDA(cudaEventRecord(first, stream));
      for (int i = 0; i < rounds; ++i) INGINE_TEST_CUDA(cudaGraphLaunch(exec, stream));
      INGINE_TEST_CUDA(cudaEventRecord(last, stream));
      INGINE_TEST_CUDA(cudaEventSynchronize(last));
      float ms = 0;
      INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, first, last));
      INGINE_TEST_CUDA(cudaGraphExecDestroy(exec));
      INGINE_TEST_CUDA(cudaGraphDestroy(graph));
      return ms * 1000.f / (rounds * copies);
    };
    const float baseline_us = time(baseline), fused_us = time(fused);
    std::printf("M=%d  unpack + tuned cuBLASLt %.2f us  fused %.2f us  %.2fx\n",
                M, baseline_us, fused_us, baseline_us / fused_us);
  }
  INGINE_TEST_CUDA(cudaEventDestroy(first));
  INGINE_TEST_CUDA(cudaEventDestroy(last));
}

void benchmark(cudaStream_t stream, bool escapes, BenchOp op, const char* name, int rows, int cols) {
  constexpr int iterations = 24;
  const auto input = make_bench_weights(rows, cols, escapes);
  const size_t outlier_count = escapes ? input.size() / kBenchOutlierStride : 0;
  const auto start = std::chrono::steady_clock::now();
  const auto packed = pico12::pack_pico12_host(input.data(), rows, cols);
  const double pack_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  INGINE_CHECK(packed.layout.escapes == escapes);
  if (escapes)
    INGINE_CHECK(pico12::device_view(packed.layout, packed.data.get()).row_ptr[rows] == outlier_count);
  const DeviceArray<uint16_t> bf16(input);
  DeviceArray<uint8_t> blob(packed.bytes);
  INGINE_TEST_CUDA(cudaMemcpy(blob.get(), packed.data.get(), packed.bytes, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  const dense::Matrix baseline{bf16.get(), nullptr, rows, cols}, pico{blob.get(), nullptr, rows, cols, &packed.layout};
  // Full-layer q, gate, rel, sink, k, v, k_dyn and v_dyn projection segments.
  const DeviceArray<int> segments(std::vector<int>{0, 3072, 0, 3072, 6144, 0, 6144, 6528, 0, 6528, 6552, 0,
                                                  6552, 7576, 1, 7576, 8600, 2, 8600, 8632, 1, 8632, 8664, 2});
  const DeviceArray<float> taps(std::vector<float>(op == BenchOp::wo ? H * 4 : 1, 0.f));
  const DeviceArray<float> state(std::vector<float>(op == BenchOp::wo ? H * 3 : 1, 0.f));
  cudaEvent_t first, last;
  INGINE_TEST_CUDA(cudaEventCreate(&first));
  INGINE_TEST_CUDA(cudaEventCreate(&last));
  const double baseline_bytes = double(input.size()) * sizeof(uint16_t);
  std::printf("%s %d x %d, %s (%.4f%% outliers), PICO12 host pack %.1f ms, blob %.2f%% of BF16\n",
              name, rows, cols, escapes ? "sparse" : "direct", 100.0 * outlier_count / input.size(), pack_ms,
              100.0 * double(packed.bytes) / baseline_bytes);
  for (const int M : kTokens) {
    const DeviceArray<__nv_bfloat16> x(activations(M, cols, op == BenchOp::wproj ? 3 : 1));
    const DeviceArray<float> y(size_t(M) * rows);
    auto run = [&](dense::Matrix w) {
      switch (op) {
        case BenchOp::head: dense::head(x.get(), M, w, y.get(), false, stream); break;
        case BenchOp::wproj:
          dense::proj(x.get(), M, w, segments.get(), int(segments.count / 3), y.get(), false, stream);
          break;
        case BenchOp::wo: dense::proj_conv(x.get(), M, w, taps.get(), state.get(), y.get(), nullptr, false, stream); break;
      }
    };
    auto time = [&](dense::Matrix w) {
      for (int i = 0; i < 3; ++i) run(w);
      INGINE_TEST_CUDA(cudaEventRecord(first, stream));
      for (int i = 0; i < iterations; ++i) run(w);
      INGINE_TEST_CUDA(cudaEventRecord(last, stream));
      INGINE_TEST_CUDA(cudaEventSynchronize(last));
      float elapsed = 0;
      INGINE_TEST_CUDA(cudaEventElapsedTime(&elapsed, first, last));
      return elapsed / iterations;
    };
    const float bf16_ms = time(baseline), pico_ms = time(pico);
    std::printf("M=%d  BF16 %.3f ms (%.1f GB/s)  PICO12 %.3f ms (%.1f real, %.1f BF16-equivalent GB/s)  %.2fx\n",
                M, bf16_ms, baseline_bytes / (bf16_ms * 1e6), pico_ms, double(packed.bytes) / (pico_ms * 1e6),
                baseline_bytes / (pico_ms * 1e6), bf16_ms / pico_ms);
  }
  INGINE_TEST_CUDA(cudaEventDestroy(first));
  INGINE_TEST_CUDA(cudaEventDestroy(last));
}

void benchmark_cold_wproj(cudaStream_t stream, bool escapes) {
  constexpr int rows = 8664, copies = 24, rounds = 4;
  constexpr size_t elements = size_t(rows) * H;
  constexpr size_t bf16_bytes = elements * sizeof(uint16_t);
  constexpr size_t fp8_bytes = elements + size_t(rows) * sizeof(float);
  const size_t outlier_count = escapes ? elements / kBenchOutlierStride : 0;
  const size_t outlier_rows = std::min(size_t(rows) * 3 / 10, outlier_count);
  const size_t pico_bytes = size_t(rows) * pico12::row_bytes(H) +
                            (escapes ? pico12::row_ptr_bytes(rows) + outlier_count * sizeof(uint32_t) +
                                       pico12::row_mask_bytes(rows) + outlier_rows * sizeof(int32_t) : 0);
  const size_t pico_stride = (pico_bytes + 15) / 16 * 16;
  const DeviceArray<int> segments(std::vector<int>{0, 6144, 0, 6144, 7424, 1, 7424, rows, 2});
  const DeviceArray<uint16_t> bf16(copies * elements);
  const DeviceArray<uint8_t> fp8(copies * elements), pico(copies * pico_stride);
  const DeviceArray<float> scales(copies * rows);
  std::array<pico12::Layout, copies> layouts;
  static_assert(bf16_bytes % 16 == 0 && elements % 16 == 0);
  for (int i = 0; i < copies; ++i) {
    const auto input = make_bench_weights(rows, H, escapes, i);
    const auto packed = pico12::pack_pico12_host(input.data(), rows, H);
    INGINE_CHECK(packed.layout.escapes == escapes && packed.bytes == pico_bytes);
    INGINE_CHECK(size_t(packed.layout.num_outlier_rows) == outlier_rows);
    auto* w = bf16.get() + i * elements;
    auto* q = fp8.get() + i * elements;
    auto* s = scales.get() + i * rows;
    INGINE_TEST_CUDA(cudaMemcpyAsync(w, input.data(), bf16_bytes, cudaMemcpyHostToDevice, stream));
    INGINE_TEST_CUDA(cudaMemcpyAsync(pico.get() + i * pico_stride, packed.data.get(), pico_bytes,
                                   cudaMemcpyHostToDevice, stream));
    dense::fp8_rows(reinterpret_cast<const __nv_bfloat16*>(w), rows, H, q, s, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    layouts[i] = packed.layout;
  }

  std::printf("cold wproj %d x %d, %d matrices per format, PICO12 %s (%.4f%% outliers)\n",
              rows, H, copies, escapes ? "sparse (deferred)" : "direct", 100.0 * outlier_count / elements);
  cudaEvent_t first, last;
  INGINE_TEST_CUDA(cudaEventCreate(&first));
  INGINE_TEST_CUDA(cudaEventCreate(&last));
  for (const int M : kTokens) {
    const DeviceArray<__nv_bfloat16> x(activations(M, H, 3));
    const DeviceArray<float> y(size_t(M) * rows);
    auto time = [&](auto matrix) {
      auto rotate = [&] {
        for (int i = 0; i < copies; ++i)
          dense::proj(x.get(), M, matrix(i), segments.get(), int(segments.count / 3), y.get(), true, stream);
      };
      rotate();
      INGINE_TEST_CUDA(cudaEventRecord(first, stream));
      for (int i = 0; i < rounds; ++i) rotate();
      INGINE_TEST_CUDA(cudaEventRecord(last, stream));
      INGINE_TEST_CUDA(cudaEventSynchronize(last));
      float elapsed_ms = 0;
      INGINE_TEST_CUDA(cudaEventElapsedTime(&elapsed_ms, first, last));
      return 1000.0 * elapsed_ms / (rounds * copies);
    };
    const double bf16_us = time([&](int i) -> dense::Matrix { return {bf16.get() + i * elements, nullptr, rows, H}; });
    const double fp8_us = time([&](int i) -> dense::Matrix {
      return {fp8.get() + i * elements, scales.get() + i * rows, rows, H};
    });
    const double pico_us = time([&](int i) -> dense::Matrix {
      return {pico.get() + i * pico_stride, nullptr, rows, H, &layouts[i]};
    });
    std::printf("M=%d  BF16 %.2f us/call (%.1f GB/s)  FP8 %.2f us/call (%.1f GB/s)"
                "  PICO12 %.2f us/call (%.1f GB/s)  PICO12/FP8 %.2fx\n",
                M, bf16_us, bf16_bytes / (bf16_us * 1e3), fp8_us, fp8_bytes / (fp8_us * 1e3),
                pico_us, pico_bytes / (pico_us * 1e3), pico_us / fp8_us);
  }
  INGINE_TEST_CUDA(cudaEventDestroy(first));
  INGINE_TEST_CUDA(cudaEventDestroy(last));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc == 2 ? argv[1] : "";
  if (argc > 2 || (argc == 2 && mode != "--host-only" && mode != "--bench" && mode != "--bench-cold-wproj" &&
                   mode != "--bench-batch-pico12")) {
    std::fprintf(stderr, "usage: %s [--host-only | --bench | --bench-cold-wproj | --bench-batch-pico12]\n", argv[0]);
    return 1;
  }
  if (mode == "--host-only") {
    test_roundtrip();
    return 0;
  }
  if (skip_without_gpu()) return kSkip;
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (mode == "--bench-batch-pico12") {
    benchmark_batch_pico12(stream, "wproj", 8664, H);
    benchmark_batch_pico12(stream, "wo", H, 3072);
    INGINE_TEST_CUDA(cudaStreamDestroy(stream));
    return 0;
  }
  for (const bool escapes : {false, true}) {
    if (mode == "--bench") {
      benchmark(stream, escapes, BenchOp::head, "head", 201024, H);
      benchmark(stream, escapes, BenchOp::wproj, "wproj", 8664, H);
      benchmark(stream, escapes, BenchOp::wo, "wo", H, 3072);
    } else if (mode == "--bench-cold-wproj") {
      benchmark_cold_wproj(stream, escapes);
    } else {
      test_device_roundtrip(stream, escapes);
      if (escapes) test_device_roundtrip(stream, true, 4, pico12::kCompactCols + pico12::kTile);
      test_fp32_ulp(stream, escapes);
      test_batch_head(stream, escapes);
      test_batch_pico12_gemm(stream, escapes, 8664, H);
      test_batch_pico12_gemm(stream, escapes, H, 3072);
      for (const bool exact : {false, true}) {
        test_gemv_family(stream, escapes, exact);
        test_convolution(stream, escapes, exact);
      }
    }
  }
  if (mode.empty()) {
    test_batch_head_rejection(stream);
    test_gemv_all_patterns(stream);
    test_sparse_graph(stream);
    for (const int K : {64, 1536, pico12::kCompactCols + 64}) test_batch_pico12_exact_graph(stream, K);
    std::puts("PASS PICO12 GPU bit-exact BF16 reconstruction, GEMV parity and batch GEMM parity");
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
}
