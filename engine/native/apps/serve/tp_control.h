#pragma once

#include "ingine/generator.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::serve {

class TpControl {
 public:
  explicit TpControl(int fd) : fd_(fd), output_(fd) {
    if (fd_ < 0) throw std::runtime_error("TP control socket failed");
    const int on = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
  }
  TpControl(int input, int output) : fd_(dup(input)), output_(dup(output)) {
    if (fd_ < 0 || output_ < 0) {
      if (fd_ >= 0) close(fd_);
      if (output_ >= 0) close(output_);
      throw std::runtime_error("cannot duplicate TP control descriptors");
    }
  }
  ~TpControl() {
    close(fd_);
    if (output_ != fd_) close(output_);
  }
  TpControl(const TpControl&) = delete;
  int fd() const { return fd_; }

  void send_frame(const std::string& body) {
    if (body.size() > kLimit) throw std::invalid_argument("TP control request too large");
    uint32_t size = htonl(body.size());
    send_bytes(&size, sizeof(size));
    send_bytes(body.data(), body.size());
  }

  std::string receive_frame() {
    uint32_t size;
    read_bytes(&size, sizeof(size));
    size = ntohl(size);
    if (size > kLimit) throw std::runtime_error("TP control frame too large");
    std::string body(size, '\0');
    read_bytes(body.data(), size);
    return body;
  }

 private:
  static constexpr size_t kLimit = 16 << 20;
  void send_bytes(const void* data, size_t size) {
    auto* bytes = static_cast<const char*>(data);
    while (size) {
      const auto n = write(output_, bytes, size);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("TP control write failed");
      bytes += n;
      size -= n;
    }
  }
  void read_bytes(void* data, size_t size) {
    auto* bytes = static_cast<char*>(data);
    while (size) {
      const auto n = read(fd_, bytes, size);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("TP control disconnected");
      bytes += n;
      size -= n;
    }
  }
  int fd_, output_;
};

inline std::unique_ptr<TpControl> connect_tp_control(int rank) {
  const auto* ip = std::getenv("INKLING_SERVE_CTRL_HOST");
  const auto* port_env = std::getenv("INKLING_SERVE_CTRL_PORT");
  const int port = port_env ? std::stoi(port_env) : 29621;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  if (port < 1 || port > 65535 || inet_pton(AF_INET, ip ? ip : "10.42.101.1", &address.sin_addr) != 1)
    throw std::invalid_argument("invalid TP control address");
  address.sin_port = htons(port);
  auto socket = std::make_unique<TpControl>(::socket(AF_INET, SOCK_STREAM, 0));
  const auto* endpoint = reinterpret_cast<const sockaddr*>(&address);
  if (rank == 0) {
    const int on = 1;
    setsockopt(socket->fd(), SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    if (bind(socket->fd(), endpoint, sizeof(address)) || listen(socket->fd(), 1))
      throw std::runtime_error("cannot listen on TP control address");
    return std::make_unique<TpControl>(accept(socket->fd(), nullptr, nullptr));
  }
  for (int retry = 0; retry < 600; ++retry) {
    if (connect(socket->fd(), endpoint, sizeof(address)) == 0) return socket;
    if (errno != ECONNREFUSED && errno != EINTR) break;
    usleep(100000);
  }
  throw std::runtime_error("cannot connect to TP control address");
}

class TpGenerator final : public model::Generator {
 public:
  TpGenerator(std::unique_ptr<model::Generator> inner, std::unique_ptr<TpControl> socket)
      : inner_(std::move(inner)), socket_(std::move(socket)) {}
  ~TpGenerator() override {
    try { socket_->send_frame(""); } catch (const std::exception&) {}
  }
  std::string model_id() const override { return inner_->model_id(); }
  int max_model_len() const override { return inner_->max_model_len(); }
  model::GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const model::Sampling& sample,
                                std::span<const int32_t> stops,
                                const std::function<bool(std::span<const int32_t>)>& emit) override {
    nlohmann::json request = {{"prompt", std::vector<int32_t>(prompt.begin(), prompt.end())},
                              {"limit", max_new_tokens}, {"temperature", sample.temperature},
                              {"top_p", sample.top_p}, {"top_k", sample.top_k},
                              {"stops", std::vector<int32_t>(stops.begin(), stops.end())}};
    if (sample.seed) request["seed"] = *sample.seed;
    socket_->send_frame(request.dump());
    return inner_->generate(prompt, max_new_tokens, sample, stops, emit);
  }
  int max_batch() const override { return inner_->max_batch(); }
  int batch_max_ctx() const override { return inner_->batch_max_ctx(); }
  int batch_spec() const override { return inner_->batch_spec(); }
  int reusable_prefix(int slot, std::span<const int32_t> prompt) const override {
    return inner_->reusable_prefix(slot, prompt);
  }

  model::PrefillChunkResult prefill_slot_chunk(int slot, std::span<const int32_t> prompt, int offset,
                                              model::BatchSampling& sampling, int stop_at = 0) override {
    socket_->send_frame(nlohmann::json{
        {"op", "prefill"}, {"slot", slot}, {"offset", offset}, {"stop_at", stop_at},
        {"prompt", std::vector<int32_t>(prompt.begin(), prompt.end())},
        {"temperature", sampling.params.temperature}}.dump());
    return inner_->prefill_slot_chunk(slot, prompt, offset, sampling, stop_at);
  }

  bool clone_slot(int src_slot, int dst_slot) override {
    socket_->send_frame(nlohmann::json{{"op", "clone"}, {"src", src_slot}, {"dst", dst_slot}}.dump());
    return inner_->clone_slot(src_slot, dst_slot);
  }

  std::vector<int32_t> decode_batch_step(std::span<const model::BatchSlotStep> rows) override {
    auto packed = nlohmann::json::array();
    for (const auto& row : rows)
      packed.push_back({row.slot, row.token, row.sampling->params.temperature});
    socket_->send_frame(nlohmann::json{{"op", "decode"}, {"rows", packed}}.dump());
    return inner_->decode_batch_step(rows);
  }

  std::vector<std::vector<int32_t>> decode_batch_step_multi(std::span<const model::BatchSlotStep> rows) override {
    auto packed = nlohmann::json::array();
    for (const auto& row : rows)
      packed.push_back({row.slot, row.token, row.sampling->params.temperature});
    socket_->send_frame(nlohmann::json{{"op", "decode_multi"}, {"rows", packed}}.dump());
    return inner_->decode_batch_step_multi(rows);
  }

  bool promote_solo_to_batch(int slot, int32_t pending_token) override {
    socket_->send_frame(nlohmann::json{{"op", "promote"}, {"slot", slot}, {"token", pending_token}}.dump());
    return inner_->promote_solo_to_batch(slot, pending_token);
  }

  void release_slot(int slot) override {
    socket_->send_frame(nlohmann::json{{"op", "release"}, {"slot", slot}}.dump());
    inner_->release_slot(slot);
  }

  model::GenerateStats generate_interruptible(
      const model::Prompt& prompt, int limit, model::BatchSampling& sampling,
      std::span<const int32_t> stops,
      const std::function<bool(std::span<const int32_t>)>& emit) override {
    const auto& sample = sampling.params;
    socket_->send_frame(nlohmann::json{
        {"op", "solo"}, {"prompt", std::vector<int32_t>(prompt.ids.begin(), prompt.ids.end())},
        {"limit", limit}, {"temperature", sample.temperature}, {"top_p", sample.top_p},
        {"top_k", sample.top_k}, {"seed", sampling.seed},
        {"stops", std::vector<int32_t>(stops.begin(), stops.end())}}.dump());
    return inner_->generate_interruptible(prompt, limit, sampling, stops, emit);
  }

 private:
  std::unique_ptr<model::Generator> inner_;
  std::unique_ptr<TpControl> socket_;
};

inline void run_tp_worker(model::Generator& generator, TpControl& socket) {
  for (;;) {
    const auto body = socket.receive_frame();
    if (body.empty()) return;
    const auto request = nlohmann::json::parse(body);
    const auto op = request.value("op", "");
    if (op == "release") {
      generator.release_slot(request.at("slot").get<int>());
      continue;
    }
    if (op == "promote") {
      generator.promote_solo_to_batch(request.at("slot").get<int>(), request.at("token").get<int32_t>());
      continue;
    }
    if (op == "clone") {
      generator.clone_slot(request.at("src").get<int>(), request.at("dst").get<int>());
      continue;
    }
    if (op == "decode_multi") {
      const auto& packed = request.at("rows");
      std::vector<model::BatchSampling> samples(packed.size());
      std::vector<model::BatchSlotStep> rows;
      for (std::size_t i = 0; i < packed.size(); ++i) {
        const auto& row = packed.at(i);
        samples[i].params.temperature = row.at(2).get<float>();
        rows.push_back({row.at(0).get<int>(), row.at(1).get<int32_t>(), &samples[i]});
      }
      generator.decode_batch_step_multi(rows);
      continue;
    }
    if (op == "decode") {
      const auto& packed = request.at("rows");
      std::vector<model::BatchSampling> samples(packed.size());
      std::vector<model::BatchSlotStep> rows;
      for (std::size_t i = 0; i < packed.size(); ++i) {
        const auto& row = packed.at(i);
        samples[i].params.temperature = row.at(2).get<float>();
        rows.push_back({row.at(0).get<int>(), row.at(1).get<int32_t>(), &samples[i]});
      }
      generator.decode_batch_step(rows);
      continue;
    }
    if (op == "prefill") {
      model::BatchSampling sample;
      sample.params.temperature = request.at("temperature").get<float>();
      try {
        generator.prefill_slot_chunk(request.at("slot").get<int>(),
                                     request.at("prompt").get<std::vector<int32_t>>(),
                                     request.at("offset").get<int>(), sample,
                                     request.value("stop_at", 0));
      } catch (const std::invalid_argument&) {
        // Rank 0 rejects the same invalid prompt before entering a collective.
      }
      continue;
    }
    const auto prompt = request.at("prompt").get<std::vector<int32_t>>();
    const auto stops = request.at("stops").get<std::vector<int32_t>>();
    model::Sampling sampling;
    sampling.temperature = request.at("temperature").get<float>();
    sampling.top_p = request.at("top_p").get<float>();
    sampling.top_k = request.at("top_k").get<int>();
    if (request.contains("seed")) sampling.seed = request.at("seed").get<uint64_t>();
    try {
      const int limit = request.at("limit").get<int>();
      if (op == "solo") {
        model::BatchSampling state{sampling, *sampling.seed};
        generator.generate_interruptible({prompt, {}}, limit, state, stops, [](auto) { return true; });
      } else {
        generator.generate(prompt, limit, sampling, stops, [](auto) { return true; });
      }
    } catch (const std::invalid_argument&) {
      // Both generators apply the same context-limit checks before entering collectives.
    }
  }
}

}  // namespace ingine::serve
