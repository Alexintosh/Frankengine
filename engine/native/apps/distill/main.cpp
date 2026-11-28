#include "core/arena.h"
#include "core/device.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace m = ingine::model;
namespace chat = ingine::text;
using Json = nlohmann::ordered_json;
constexpr std::size_t tap_bytes = m::NUM_TAPS * m::HIDDEN * sizeof(std::uint16_t);
static_assert(std::endian::native == std::endian::little);

struct Sequence {
  std::vector<std::int32_t> tokens;
  std::size_t prompt_len;
};

std::size_t checked_index(const Json& value, std::size_t limit, const char* field) {
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<std::int64_t>() < 0) ||
      value.get<std::uint64_t>() >= limit)
    throw std::invalid_argument(std::string("invalid ") + field);
  return value.get<std::size_t>();
}

Sequence read_sequence(const Json& row, const fs::path& model_dir, std::optional<chat::ChatFormat>& format) {
  if (row.contains("tokens")) {
    const auto& tokens = row.at("tokens").get_ref<const Json::array_t&>();
    Sequence seq{{}, checked_index(row.at("prompt_len"), tokens.size() + 1, "prompt_len")};
    seq.tokens.reserve(tokens.size());
    for (const auto& id : tokens)
      seq.tokens.push_back(static_cast<std::int32_t>(checked_index(id, m::UNPADDED_VOCAB, "token id")));
    return seq;
  }

  if (!format) format.emplace(chat::ChatFormat::load(model_dir));
  chat::ChatOptions options;
  const bool thinking = row.value("thinking", true);
  options.enable_thinking = thinking;
  if (const auto effort = row.find("reasoning_effort"); effort != row.end()) options.reasoning_effort = *effort;
  auto prompt = format->render(Json::array({{{"role", "user"}, {"content", row.at("prompt").get<std::string>()}}}),
                               nullptr, options);
  const auto prompt_len = prompt.ids.size();
  Sequence seq{std::move(prompt.ids), prompt_len};
  const auto& tokenizer = format->tokenizer();
  const auto special = [&](std::string_view token) {
    seq.tokens.push_back(tokenizer.token_id(token).value());
  };
  const auto body = [&](const std::string& text) {
    auto ids = tokenizer.encode(text, /*special=*/false);
    seq.tokens.insert(seq.tokens.end(), ids.begin(), ids.end());
  };

  const auto optional_string = [&](const char* key) {
    const auto it = row.find(key);
    return it == row.end() || it->is_null() ? std::string{} : it->get<std::string>();
  };
  const auto reasoning = optional_string("reasoning");
  const auto response = optional_string("response");
  const bool truncated = optional_string("finish_reason") == "length";
  if (!reasoning.empty()) {
    if (!thinking) throw std::invalid_argument("reasoning supplied with thinking disabled");
    special(chat::MESSAGE_MODEL);
    special(chat::CONTENT_THINKING);
    body(reasoning);
    if (!response.empty() || !truncated) special(chat::END_MESSAGE);
  }
  if (!thinking || !response.empty() || (reasoning.empty() && !truncated)) {
    if (thinking) {
      special(chat::MESSAGE_MODEL);
      special(chat::CONTENT_TEXT);
    }
    body(response);
    if (!truncated) special(chat::END_MESSAGE);
  }
  if (!truncated) special(chat::END_SAMPLING);

  for (auto id : seq.tokens)
    if (id < 0 || id >= m::UNPADDED_VOCAB) throw std::invalid_argument("invalid rendered token id");
  return seq;
}

class Shards {
 public:
  Shards(fs::path dir, std::size_t limit) : dir_(std::move(dir)), limit_(limit) {
    fs::create_directories(dir_);
    for (const auto& entry : fs::directory_iterator(dir_))
      if (entry.path().filename().string().starts_with("shard_"))
        throw std::runtime_error("output directory already contains shard files");
    for (auto* stream : {&tokens_, &taps_}) stream->exceptions(std::ios::failbit | std::ios::badbit);
  }

  void begin(const Sequence& seq) {
    if (rows_ && rows_ + seq.tokens.size() > limit_) finish();
    if (rows_ == 0) {
      char stem[32];
      std::snprintf(stem, sizeof(stem), "shard_%04zu", count_);
      stem_ = dir_ / stem;
      tokens_.open(path(".tokens.bin"), std::ios::binary);
      taps_.open(path(".taps.bin"), std::ios::binary);
    }
    tokens_.write(reinterpret_cast<const char*>(seq.tokens.data()), seq.tokens.size() * sizeof(std::int32_t));
    seqs_.push_back({{"offset", rows_}, {"length", seq.tokens.size()}, {"prompt_len", seq.prompt_len}});
    rows_ += seq.tokens.size();
  }

  void write_taps(const void* data, std::size_t bytes) {
    taps_.write(static_cast<const char*>(data), bytes);
  }

  void finish() {
    if (!rows_) return;
    tokens_.close();
    taps_.close();
    const Json meta = {{"shard", count_}, {"num_tokens", rows_}, {"num_seqs", seqs_.size()},
                       {"hidden", m::HIDDEN}, {"num_taps", m::NUM_TAPS}, {"seqs", seqs_}};
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path(".meta.part"));
    out << meta.dump() << '\n';
    out.close();
    fs::rename(path(".meta.part"), path(".meta.json"));
    ++count_;
    rows_ = 0;
    seqs_ = Json::array();
  }

  std::size_t count() const { return count_; }

 private:
  fs::path path(const char* suffix) const { return stem_.string() + suffix; }

  fs::path dir_, stem_;
  std::size_t limit_, rows_ = 0, count_ = 0;
  std::ofstream tokens_, taps_;
  Json seqs_ = Json::array();
};

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, input_path, out_dir;
  int shard_tokens = 65536, max_ctx = 4096, prefill_chunk = 2048, max_seqs = 0;
  bool dense_fp8 = false, kv_fp8 = false;
  ingine::serve::ArgParser ap("ingine-dump-taps", "Extract native target BF16 taps from JSONL rollouts.");
  ap.option("--model-dir", "PATH", "target checkpoint", &model_dir, {}, true);
  ap.option("--input", "JSONL", "pre-tokenized or chat rollouts", &input_path, {}, true);
  ap.option("--out-dir", "PATH", "directory for binary shards and metadata", &out_dir, {}, true);
  ap.option("--shard-tokens", "N", "target tokens per shard (65536)", &shard_tokens);
  ap.option("--max-ctx", "N", "maximum sequence length (4096)", &max_ctx);
  ap.option("--prefill-chunk", "N", "tokens per prefill call (2048)", &prefill_chunk);
  ap.option("--max-seqs", "N", "maximum accepted sequences (0 = all)", &max_seqs);
  ap.flag("--dense-fp8", "Use native FP8 dense prefill", &dense_fp8);
  ap.flag("--kv-fp8", "FP8 full-attention KV caches", &kv_fp8);
  ap.parse(argc, argv);

  try {
    if (shard_tokens < 1 || max_ctx < 8 || prefill_chunk < 1 || max_seqs < 0)
      throw std::invalid_argument("expected positive shard/chunk sizes, max context >= 8 and max sequences >= 0");
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("cannot open " + input_path);
    Shards shards(out_dir, shard_tokens);
    ingine::DeviceContext device;
    const auto stream = device.stream;
    const auto weights = m::load_weights(model_dir, dense_fp8, stream);
    const auto ablation = m::load_ablation({});
    auto state = m::make_state(max_ctx, kv_fp8);
    const int chunk = std::min(prefill_chunk, max_ctx);
    m::Prefiller prefill(weights, ablation, chunk, stream);
    ingine::DeviceBuffer tokens(std::size_t(chunk) * sizeof(std::int32_t)), taps(std::size_t(chunk) * tap_bytes);
    ingine::PinnedHostBuffer host_taps(taps.bytes);
    std::optional<chat::ChatFormat> format;
    std::size_t accepted = 0, total_tokens = 0, short_seqs = 0, long_seqs = 0, line_no = 0;
    const auto start = std::chrono::steady_clock::now();
    std::string line;
    while ((!max_seqs || accepted < std::size_t(max_seqs)) && std::getline(input, line)) {
      ++line_no;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      Sequence seq;
      try {
        seq = read_sequence(Json::parse(line), model_dir, format);
      } catch (const std::exception& e) {
        throw std::runtime_error("line " + std::to_string(line_no) + ": " + e.what());
      }
      if (seq.tokens.size() - seq.prompt_len < 8) {
        ++short_seqs;
        continue;
      }
      if (seq.tokens.size() > std::size_t(max_ctx)) {
        ++long_seqs;
        continue;
      }
      shards.begin(seq);
      m::reset_state(state, stream);
      const int length = static_cast<int>(seq.tokens.size());
      for (int pos = 0; pos < length;) {
        const int count = std::min(chunk, length - pos);
        CUDA_CHECK(cudaMemcpyAsync(tokens.p, seq.tokens.data() + pos, std::size_t(count) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        auto view = m::rm(taps.p, ingine::DType::BF16, {count, m::NUM_TAPS, m::HIDDEN});
        prefill.run(state, static_cast<const std::int32_t*>(tokens.p), count, pos, nullptr, &view, {}, nullptr,
                    /*emit_head=*/false);
        const auto bytes = std::size_t(count) * tap_bytes;
        CUDA_CHECK(cudaMemcpyAsync(host_taps.data(), taps.p, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        shards.write_taps(host_taps.data(), bytes);
        pos += count;
      }
      ++accepted;
      total_tokens += length;
    }
    if (input.bad()) throw std::runtime_error("error reading " + input_path);
    shards.finish();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (short_seqs || long_seqs)
      std::fprintf(stderr, "warning: skipped %zu short and %zu over-context sequences\n", short_seqs, long_seqs);
    std::printf("seqs=%zu tokens=%zu shards=%zu elapsed_s=%.3f tok/s=%.1f\n", accepted, total_tokens,
                shards.count(), elapsed, elapsed > 0 ? total_tokens / elapsed : 0.0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-dump-taps: %s\n", e.what());
    return 1;
  }
}
