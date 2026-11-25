#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace inkling::comm {

using Bf16 = __nv_bfloat16;

template <class T>
struct DeviceSpan {
    T* data = nullptr;
    std::size_t size = 0;  // Elements, including for byte spans.
};

using DeviceBytes = DeviceSpan<const std::byte>;
using MutableDeviceBytes = DeviceSpan<std::byte>;

struct CollectiveSite {
    std::uint32_t schedule_id;
    std::uint32_t op_index;
};

struct CommPlan {
    std::size_t max_payload_bytes = 0;  // Maximum per rank, across all sites.
    std::uint64_t layout_hash = 0;
    std::uint64_t checkpoint_hash = 0;
    std::uint64_t schedule_hash = 0;
};

struct GatherLayout {
    std::size_t rows;
    std::array<std::size_t, 2> columns_per_rank;
    // Root receives rows in rank/column order. Zero selects a packed root row.
    std::size_t root_row_stride = 0;  // Floats.
};

enum class VisibilityMode { direct, cache_bypass, host_bounce };

struct RoceRail {
    std::string device;
    std::array<std::string, 2> ip;
    int gid_index = 3;  // -1 selects a RoCEv2 GID by local IP.
    int port = 1;
};

struct CommOptions {
    int rank = 0;
    int world_size = 1;
    int cuda_device = 0;
    std::string rendezvous_ip = "10.42.101.1";
    std::uint16_t rendezvous_port = 29620;
    std::array<RoceRail, 2> rails{{
        {"rocep1s0f1", {"10.42.101.1", "10.42.101.2"}},
        {"roceP2p1s0f1", {"10.42.102.1", "10.42.102.2"}},
    }};
    int rail_count = 2;
    int proxy_cpu = -1;  // Highest CPU in the calling thread's affinity mask.
    std::chrono::milliseconds bootstrap_timeout{10000};
    std::chrono::milliseconds timeout{5000};  // Data-path timeout per collective.
    VisibilityMode visibility = VisibilityMode::direct;
    bool trace = false;
};

struct CommStatus {
    bool ok = true;
    std::uint64_t epoch = 0;
    CollectiveSite site{};
    std::string message;
};

struct CommStats {
    std::array<std::uint64_t, 2> tx_wire_bytes{};
    std::array<std::uint64_t, 2> rx_wire_bytes{};
    std::array<std::uint64_t, 2> dual_first_rx{};
    std::uint64_t completed = 0;
    std::uint64_t proxy_cpu_ns = 0;
    int proxy_cpu = -1;
};

struct CommTrace {
    std::uint64_t epoch = 0;
    CollectiveSite site{};
    std::uint64_t gpu_publish_ns = 0;
    std::uint64_t gpu_ready_ns = 0;
    std::array<std::uint64_t, 2> host_rx_cq_ns{};
    std::array<std::uint64_t, 2> host_tx_cq_ns{};
    std::uint64_t host_ready_ns = 0;
};

class CommBackend {
public:
    virtual ~CommBackend() = default;
    static std::unique_ptr<CommBackend> init_from_env(const CommPlan& plan);

    virtual int rank() const noexcept = 0;
    virtual int world_size() const noexcept = 0;

    // Callers serialize all work on a single stream/graph chain per rank.
    // Retire submitted work before destroying the backend. All spans are device
    // memory. AllReduce allows exact src == dst aliasing.
    virtual void allreduce_bf16(DeviceSpan<const Bf16> src, DeviceSpan<Bf16> dst,
                                CollectiveSite site, cudaStream_t stream) = 0;
    virtual void allreduce_f32(DeviceSpan<const float> src, DeviceSpan<float> dst,
                               CollectiveSite site, cudaStream_t stream) = 0;
    virtual void allgather_small(DeviceBytes src, MutableDeviceBytes rank_order_dst,
                                 CollectiveSite site, cudaStream_t stream) = 0;
    virtual void broadcast_tokens(MutableDeviceBytes record, int root,
                                  CollectiveSite site, cudaStream_t stream) = 0;
    virtual void gather_rows_to_root(DeviceSpan<const float> local, MutableDeviceBytes root_dst,
                                     const GatherLayout& layout, int root,
                                     CollectiveSite site, cudaStream_t stream) = 0;

    // Health, statistics and tracing are host queries outside graph capture.
    // Test status after the replay-completion event, before committing results.
    virtual CommStatus status() const = 0;
    virtual CommStats stats() const = 0;
    virtual CommTrace last_trace() const = 0;
    // Two disjoint words so the CPU and GPU each own a flag. Device code
    // acquire-loads either flag with a system-scope atomic; nullptr for TP=1.
    struct DeviceHealth {
        const std::uint64_t* host_failed;
        const std::uint64_t* gpu_failed_epoch;
    };
    virtual DeviceHealth device_health() const noexcept = 0;
};

CommOptions comm_options_from_env();
std::unique_ptr<CommBackend> make_comm_backend(const CommPlan& plan, const CommOptions& options);

}  // namespace inkling::comm
