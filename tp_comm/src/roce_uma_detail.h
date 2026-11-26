#pragma once

#include "inkling_comm/comm_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inkling::comm::detail {

using Clock = std::chrono::steady_clock;
constexpr unsigned kSlots = 2;
constexpr unsigned kRails = 2;
constexpr unsigned kHeaderBytes = 64;
constexpr unsigned kTinyBytes = 256;
constexpr unsigned kControlAlignment = 128;
constexpr std::uint32_t kLargeThresholdBytesF32 = 1u << 20;    // 1 MiB: empirically optimal 1-CTA->48-CTA crossover on 2-node Dual-Spark FP32
constexpr std::uint32_t kLargeThresholdBytesBf16 = 512u << 10; // 512 KiB: empirically optimal 1-CTA->48-CTA crossover on 2-node Dual-Spark BF16
constexpr std::uint32_t kBulkPipeThresholdBytes = 8u << 20;    // 8 MiB: pipelined K=4 bulk RDMA wins cleanly on 9.44 MiB BF16 & 18.87 MiB FP32 prefill
constexpr unsigned kMaxBulkSlices = 16;

inline void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define INKLING_CUDA(operation) ::inkling::comm::detail::cuda_check((operation), #operation)

inline void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}

inline std::size_t product(std::size_t a, std::size_t b) {
    require(!b || a <= std::numeric_limits<std::size_t>::max() / b, "collective size overflow");
    return a * b;
}

inline std::size_t sum(std::size_t a, std::size_t b) {
    require(a <= std::numeric_limits<std::size_t>::max() - b, "collective size overflow");
    return a + b;
}

template <class T>
void valid_span(DeviceSpan<T> span) {
    require(!span.size || span.data, "collective requires a device buffer");
}

struct RowSizes {
    std::array<std::size_t, 2> rank_bytes;
    std::size_t stride;
    std::size_t root_bytes;
};

inline RowSizes row_sizes(const GatherLayout& layout, int ranks) {
    const auto columns = sum(layout.columns_per_rank[0], ranks == 2 ? layout.columns_per_rank[1] : 0);
    const auto stride = layout.root_row_stride ? layout.root_row_stride : columns;
    require(stride >= columns, "gather root stride is smaller than the row");
    const auto elements = layout.rows && columns ? sum(product(layout.rows - 1, stride), columns) : 0;
    return {{
        product(product(layout.rows, layout.columns_per_rank[0]), sizeof(float)),
        ranks == 2 ? product(product(layout.rows, layout.columns_per_rank[1]), sizeof(float)) : 0,
    }, stride, product(elements, sizeof(float))};
}

inline void validate_gather(DeviceSpan<const float> local, MutableDeviceBytes dst,
                            const RowSizes& sizes, int rank, int root, int ranks) {
    require(root >= 0 && root < ranks, "invalid collective root");
    valid_span(local);
    require(product(local.size, sizeof(float)) == sizes.rank_bytes[rank], "gather input size mismatch");
    if (rank == root) {
        valid_span(dst);
        require(dst.size >= sizes.root_bytes, "gather root output is too small");
    }
}

enum class Kind : std::uint8_t { bf16, f32, allgather, broadcast, gather_rows };

struct Operation {
    CollectiveSite site;
    std::uint32_t bytes[2];  // Payload staged by each rank; a zero-byte send still carries a credit.
    std::uint32_t columns[2]{};
    std::uint64_t stride = 0;
    Kind kind;
    std::uint8_t root = 0;
    std::uint8_t slices = 1;
};

struct WireHeader {
    std::uint64_t session;
    std::uint64_t epoch;
    std::uint64_t stride;
    CollectiveSite site;
    std::uint32_t bytes[2];
    std::uint32_t columns[2];
    std::uint32_t stripe_bytes;
    Kind kind;
    std::uint8_t root;
    std::uint8_t src_rank;
    std::uint8_t rail;
    std::uint8_t slot;
    std::uint8_t slices;
    std::uint8_t reserved[6];
};
static_assert(sizeof(WireHeader) == kHeaderBytes);

struct alignas(kControlAlignment) EpochWord {
    std::uint64_t value = 0;
};

struct alignas(kControlAlignment) TxNotice {
    EpochWord epoch;
    Operation operation;
    std::uint64_t publish_ns;
};

struct alignas(kControlAlignment) RxNotice {
    EpochWord epoch;
    std::uint64_t trace_ready_ns[kSlots];
};

struct SharedControl {
    TxNotice tx;
    EpochWord ready;
    RxNotice consumed;
    EpochWord host_failed;
    EpochWord gpu_failed;
    EpochWord bulk_tx_slice;  // (epoch << 8) | packed_slices  (GPU -> host proxy)
    EpochWord bulk_rx_slice;  // (epoch << 8) | arrived_slices (host proxy -> GPU)
};

struct Buffers {
    std::byte* tx[kSlots][kRails]{};
    std::byte* rx[kSlots][kRails]{};
};

struct Stripe {
    std::uint32_t offset;
    std::uint32_t bytes;
};

#ifdef __CUDACC__
#define INKLING_HOST_DEVICE __host__ __device__
#else
#define INKLING_HOST_DEVICE
#endif

INKLING_HOST_DEVICE inline std::uint32_t wire_bytes(const Operation& op, unsigned rank) {
    return op.kind == Kind::gather_rows && rank == op.root ? 0 : op.bytes[rank];
}

INKLING_HOST_DEVICE inline unsigned rails_for(std::uint32_t bytes, int rails) {
    return rails == 2 && bytes > kTinyBytes ? 2 : 1;
}

INKLING_HOST_DEVICE inline Stripe stripe(std::uint32_t bytes, unsigned rail, int rails) {
    if (rails_for(bytes, rails) == 1) return {0, rail == 0 ? bytes : 0};
    // Keep the midpoint aligned for vectorized traffic and typed reductions.
    const std::uint32_t midpoint = (bytes / 2) & ~15u;
    return rail == 0 ? Stripe{0, midpoint} : Stripe{midpoint, bytes - midpoint};
}

INKLING_HOST_DEVICE inline unsigned slot_for(std::uint64_t epoch) {
    return static_cast<unsigned>(epoch % kSlots);
}

INKLING_HOST_DEVICE inline Stripe slice_of(std::uint32_t rail_bytes, unsigned slice, unsigned slices) {
    if (slices <= 1) return {0, rail_bytes};
    const std::uint32_t chunk = ((rail_bytes / slices) + 15u) & ~15u;
    const std::uint32_t offset = slice * chunk;
    if (offset >= rail_bytes) return {rail_bytes, 0};
    const std::uint32_t end =
        (slice + 1 == slices || offset + chunk > rail_bytes) ? rail_bytes : (offset + chunk);
    return {offset, end - offset};
}

INKLING_HOST_DEVICE inline std::uint64_t pack_slice_word(std::uint64_t epoch, unsigned count) {
    return (epoch << 8) | static_cast<std::uint64_t>(count & 0xffu);
}

INKLING_HOST_DEVICE inline WireHeader make_header(std::uint64_t session, const Operation& op,
                                                  int rank, unsigned rail, std::uint64_t epoch,
                                                  unsigned slot, std::uint32_t bytes) {
    return {session, epoch, op.stride, op.site, {op.bytes[0], op.bytes[1]},
            {op.columns[0], op.columns[1]}, bytes, op.kind, op.root, static_cast<std::uint8_t>(rank),
            static_cast<std::uint8_t>(rail), static_cast<std::uint8_t>(slot), op.slices, {}};
}

#undef INKLING_HOST_DEVICE

class RoceProxy {
public:
    RoceProxy(const CommPlan& plan, const CommOptions& options, SharedControl* control,
              const Buffers& dma, const Buffers& visible, std::byte* registered_base,
              std::size_t registered_bytes);
    ~RoceProxy();
    RoceProxy(const RoceProxy&) = delete;
    RoceProxy& operator=(const RoceProxy&) = delete;

    std::uint64_t session() const noexcept;
    CommStatus status() const;
    CommStats stats() const;
    CommTrace last_trace() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<CommBackend> make_roce_backend(const CommPlan& plan, const CommOptions& options);

}  // namespace inkling::comm::detail
