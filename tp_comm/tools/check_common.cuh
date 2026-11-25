#pragma once

#include "inkling_comm/comm_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

namespace inkling::comm::tools {
using Clock = std::chrono::steady_clock;

inline void cuda_check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) ::inkling::comm::tools::cuda_check((call), #call)

inline void check(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

inline void check_health(const CommBackend& comm) {
    const auto status = comm.status();
    if (!status.ok) {
        throw std::runtime_error("epoch " + std::to_string(status.epoch) + ", schedule " +
            std::to_string(status.site.schedule_id) + ", op " + std::to_string(status.site.op_index) +
            ": " + status.message);
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), size * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return ptr_; }
    DeviceSpan<T> span() const { return {ptr_, size_}; }
    DeviceSpan<const T> input() const { return {ptr_, size_}; }
    MutableDeviceBytes bytes() const { return {reinterpret_cast<std::byte*>(ptr_), size_ * sizeof(T)}; }
    DeviceBytes input_bytes() const { return {reinterpret_cast<const std::byte*>(ptr_), size_ * sizeof(T)}; }

    void fill_zero(cudaStream_t stream) { CUDA(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream)); }

    std::vector<T> read() const {
        std::vector<T> data(size_);
        CUDA(cudaMemcpy(data.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }

    void write(const std::vector<T>& data) {
        check(data.size() == size_, "invalid upload size");
        CUDA(cudaMemcpy(ptr_, data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_;
};

struct Stream {
    Stream() { CUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator cudaStream_t() const { return value; }
    cudaStream_t value{};
};

class Event {
public:
    explicit Event(unsigned flags = cudaEventDisableTiming) { CUDA(cudaEventCreateWithFlags(&event_, flags)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream) { CUDA(cudaEventRecord(event_, stream)); }
    void record_external(cudaStream_t stream) { CUDA(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal)); }
    void wait() { CUDA(cudaEventSynchronize(event_)); }
    float us_since(const Event& start) const {
        float ms = 0;
        CUDA(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms * 1000.f;
    }

private:
    cudaEvent_t event_{};
};

class Graph {
public:
    template <class Record>
    Graph(cudaStream_t stream, Record record) {
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        try {
            record();
        } catch (...) {
            cudaStreamEndCapture(stream, &graph_);
            if (graph_) cudaGraphDestroy(graph_);
            throw;
        }
        CUDA(cudaStreamEndCapture(stream, &graph_));
        const auto status = cudaGraphInstantiate(&exec_, graph_, 0);
        if (status != cudaSuccess) cudaGraphDestroy(graph_);
        CUDA(status);
    }
    ~Graph() {
        cudaGraphExecDestroy(exec_);
        cudaGraphDestroy(graph_);
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    void launch(cudaStream_t stream) { CUDA(cudaGraphLaunch(exec_, stream)); }

private:
    cudaGraph_t graph_{};
    cudaGraphExec_t exec_{};
};

struct Mismatch {
    unsigned long long count = 0;
    std::uint64_t ordinal = 0;
    std::size_t element = 0;
    std::uint32_t got = 0;
    std::uint32_t expected = 0;
};

__device__ inline void mark(Mismatch* error, std::uint64_t ordinal, std::size_t index,
                            std::uint32_t got, std::uint32_t expected) {
    if (atomicAdd(&error->count, 1ull) == 0) {
        error->ordinal = ordinal;
        error->element = index;
        error->got = got;
        error->expected = expected;
    }
}

inline void verify(const DeviceBuffer<Mismatch>& errors) {
    const auto error = errors.read().front();
    if (error.count) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "%llu element mismatches. First at sequence %llu element %zu: got 0x%04x expected 0x%04x",
                      error.count, static_cast<unsigned long long>(error.ordinal), error.element,
                      error.got, error.expected);
        throw std::runtime_error(message);
    }
}

__host__ __device__ inline std::uint32_t pattern(std::uint64_t sequence, unsigned index, int rank) {
    auto value = std::uint32_t(sequence) * 0x9e3779b9u + (index ^ (rank * 0xd1b54a35u));
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__device__ inline unsigned short finite_bf16_bits(std::uint64_t sequence, unsigned index, int rank) {
    const auto value = pattern(sequence, index, rank);
    return (value & 0x807fu) | ((112u + ((value >> 16) & 15u)) << 7);
}

inline const char* visibility_name(VisibilityMode mode) {
    switch (mode) {
    case VisibilityMode::direct: return "direct";
    case VisibilityMode::cache_bypass: return "cache_bypass";
    case VisibilityMode::host_bounce: return "host_bounce";
    }
    return "unknown";
}

inline std::uint64_t real_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

inline void print_stats(const CommStats& first, const CommStats& last, std::uint64_t interval_ns, int rank) {
    std::fprintf(stderr,
        "rank=%d proxy_cpu=%d proxy_cpu_pct=%.2f completed=%llu tx_wire_bytes=%llu/%llu "
        "rx_wire_bytes=%llu/%llu dual_first_rx=%llu/%llu\n",
        rank, last.proxy_cpu, 100.0 * (last.proxy_cpu_ns - first.proxy_cpu_ns) / interval_ns,
        static_cast<unsigned long long>(last.completed - first.completed),
        static_cast<unsigned long long>(last.tx_wire_bytes[0] - first.tx_wire_bytes[0]),
        static_cast<unsigned long long>(last.tx_wire_bytes[1] - first.tx_wire_bytes[1]),
        static_cast<unsigned long long>(last.rx_wire_bytes[0] - first.rx_wire_bytes[0]),
        static_cast<unsigned long long>(last.rx_wire_bytes[1] - first.rx_wire_bytes[1]),
        static_cast<unsigned long long>(last.dual_first_rx[0] - first.dual_first_rx[0]),
        static_cast<unsigned long long>(last.dual_first_rx[1] - first.dual_first_rx[1]));
}

inline void print_trace(const CommTrace& trace, int rank) {
    std::fprintf(stderr,
        "rank=%d trace_epoch=%llu schedule=%u op=%u gpu_publish_ns=%llu gpu_ready_ns=%llu "
        "host_rx_cq_ns=%llu/%llu host_tx_cq_ns=%llu/%llu host_ready_ns=%llu\n",
        rank, static_cast<unsigned long long>(trace.epoch), trace.site.schedule_id, trace.site.op_index,
        static_cast<unsigned long long>(trace.gpu_publish_ns),
        static_cast<unsigned long long>(trace.gpu_ready_ns),
        static_cast<unsigned long long>(trace.host_rx_cq_ns[0]),
        static_cast<unsigned long long>(trace.host_rx_cq_ns[1]),
        static_cast<unsigned long long>(trace.host_tx_cq_ns[0]),
        static_cast<unsigned long long>(trace.host_tx_cq_ns[1]),
        static_cast<unsigned long long>(trace.host_ready_ns));
}
}  // namespace inkling::comm::tools
