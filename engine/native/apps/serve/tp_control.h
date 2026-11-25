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
 private:
  std::unique_ptr<model::Generator> inner_;
  std::unique_ptr<TpControl> socket_;
};

inline void run_tp_worker(model::Generator& generator, TpControl& socket) {
  for (;;) {
    const auto body = socket.receive_frame();
    if (body.empty()) return;
    const auto request = nlohmann::json::parse(body);
    const auto prompt = request.at("prompt").get<std::vector<int32_t>>();
    const auto stops = request.at("stops").get<std::vector<int32_t>>();
    model::Sampling sampling;
    sampling.temperature = request.at("temperature").get<float>();
    sampling.top_p = request.at("top_p").get<float>();
    sampling.top_k = request.at("top_k").get<int>();
    if (request.contains("seed")) sampling.seed = request.at("seed").get<uint64_t>();
    try {
      generator.generate(prompt, request.at("limit").get<int>(), sampling, stops, [](auto) { return true; });
    } catch (const std::invalid_argument&) {
      // Both generators apply the same context-limit checks before entering collectives.
    }
  }
}

}  // namespace ingine::serve
