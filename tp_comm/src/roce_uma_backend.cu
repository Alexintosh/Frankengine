#include "roce_uma_detail.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdlib>
#include <new>

namespace inkling::comm::detail {
namespace {
constexpr unsigned kThreads = 256;

__device__ std::uint64_t gpu_ns() {
    std::uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

__device__ std::uint64_t load(EpochWord& word) {
    return cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(word.value).load(cuda::memory_order_acquire);
}

__device__ void store(EpochWord& word, std::uint64_t epoch) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(word.value).store(epoch, cuda::memory_order_release);
}

template <bool bypass, class T>
__device__ T payload_load(const T* ptr) {
    if constexpr (bypass) return __ldcv(ptr);
    return *ptr;
}

__device__ bool vector_aligned(const void* src, const void* dst) {
    return !((reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)) & 15);
}

template <bool bypass>
__device__ void copy_bytes(const std::byte* src, std::byte* dst, std::uint32_t bytes) {
    std::uint32_t copied = 0;
    if (vector_aligned(src, dst)) {
        const auto* input = reinterpret_cast<const uint4*>(src);
        auto* output = reinterpret_cast<uint4*>(dst);
        const auto vectors = bytes / sizeof(uint4);
        for (std::uint32_t i = threadIdx.x; i < vectors; i += blockDim.x)
            output[i] = payload_load<bypass>(input + i);
        copied = vectors * sizeof(uint4);
    }
    const auto* input = reinterpret_cast<const unsigned char*>(src);
    auto* output = reinterpret_cast<unsigned char*>(dst);
    for (std::uint32_t i = copied + threadIdx.x; i < bytes; i += blockDim.x)
        output[i] = payload_load<bypass>(input + i);
}

__device__ void pack_bytes(const std::byte* src, std::byte* const* tx,
                           std::uint32_t bytes, int rails) {
    if (bytes % sizeof(uint4) == 0 && vector_aligned(src, tx[0])) {
        const auto* input = reinterpret_cast<const uint4*>(src);
        auto* first = reinterpret_cast<uint4*>(tx[0] + kHeaderBytes);
        const auto count = bytes / sizeof(uint4);
        const auto split = stripe(bytes, 0, rails).bytes / sizeof(uint4);
        auto* second = count > split ? reinterpret_cast<uint4*>(tx[1] + kHeaderBytes) : nullptr;
        for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            auto* output = i < split ? first + i : second + (i - split);
            *output = input[i];
        }
        return;
    }
    for (unsigned rail = 0; rail < rails_for(bytes, rails); ++rail) {
        const auto part = stripe(bytes, rail, rails);
        copy_bytes<false>(src + part.offset, tx[rail] + kHeaderBytes, part.bytes);
    }
}

__device__ std::uint64_t pack(SharedControl* control, std::uint64_t* device_epoch,
                              const Buffers& buffers, const std::byte* src,
                              const Operation& op, int rank, int rails,
                              std::uint64_t session, bool trace) {
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = load(control->host_failed) ? 0 : *device_epoch;
        if (epoch) *device_epoch = epoch + 1;
    }
    __syncthreads();
    if (!epoch) return 0;

    const unsigned slot = slot_for(epoch);
    const auto bytes = op.bytes[rank];
    if (bytes) pack_bytes(src, buffers.tx[slot], bytes, rails);
    const auto wire = wire_bytes(op, rank);
    if (threadIdx.x < rails_for(wire, rails)) {
        const auto rail = threadIdx.x;
        const auto part = stripe(wire, rail, rails);
        *reinterpret_cast<WireHeader*>(buffers.tx[slot][rail]) =
            make_header(session, op, rank, rail, epoch, slot, part.bytes);
    }
    if (threadIdx.x == 0) control->tx.operation = op;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (trace) control->tx.publish_ns = gpu_ns();
        store(control->tx.epoch, epoch);
    }
    return epoch;
}

__device__ unsigned short sum_bf16(unsigned short local, unsigned short peer, int rank) {
    const float a = __bfloat162float(__ushort_as_bfloat16(rank == 0 ? local : peer));
    const float b = __bfloat162float(__ushort_as_bfloat16(rank == 0 ? peer : local));
    return __bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(a, b)));
}

__device__ unsigned sum_bf16_pair(unsigned local, unsigned peer, int rank) {
    const auto lo = sum_bf16(local, peer, rank);
    const auto hi = sum_bf16(local >> 16, peer >> 16, rank);
    return static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
}

__device__ float sum_f32(float local, float peer, int rank) {
    return rank == 0 ? __fadd_rn(local, peer) : __fadd_rn(peer, local);
}

template <bool bf16>
__device__ unsigned sum_word(unsigned local, unsigned peer, int rank) {
    if constexpr (bf16) return sum_bf16_pair(local, peer, rank);
    return __float_as_uint(sum_f32(__uint_as_float(local), __uint_as_float(peer), rank));
}

template <bool bypass, bool bf16>
__device__ void reduce_bytes(const std::byte* local, const std::byte* peer, std::byte* output,
                             std::uint32_t bytes, int rank) {
    std::uint32_t done = 0;
    if (vector_aligned(local, output)) {
        const auto* a = reinterpret_cast<const uint4*>(local);
        const auto* b = reinterpret_cast<const uint4*>(peer);
        auto* dst = reinterpret_cast<uint4*>(output);
        const auto vectors = bytes / sizeof(uint4);
        for (std::uint32_t i = threadIdx.x; i < vectors; i += blockDim.x) {
            const auto x = a[i];
            const auto y = payload_load<bypass>(b + i);
            dst[i] = make_uint4(sum_word<bf16>(x.x, y.x, rank), sum_word<bf16>(x.y, y.y, rank),
                                sum_word<bf16>(x.z, y.z, rank), sum_word<bf16>(x.w, y.w, rank));
        }
        done = vectors * sizeof(uint4);
    }
    if constexpr (bf16) {
        const auto* a = reinterpret_cast<const unsigned short*>(local);
        const auto* b = reinterpret_cast<const unsigned short*>(peer);
        auto* dst = reinterpret_cast<unsigned short*>(output);
        for (std::uint32_t i = done / 2 + threadIdx.x; i < bytes / 2; i += blockDim.x)
            dst[i] = sum_bf16(a[i], payload_load<bypass>(b + i), rank);
    } else {
        const auto* a = reinterpret_cast<const float*>(local);
        const auto* b = reinterpret_cast<const float*>(peer);
        auto* dst = reinterpret_cast<float*>(output);
        for (std::uint32_t i = done / 4 + threadIdx.x; i < bytes / 4; i += blockDim.x)
            dst[i] = sum_f32(a[i], payload_load<bypass>(b + i), rank);
    }
}

template <bool bypass>
__device__ void reduce_bf16_vectors(const std::byte* src, std::byte* dst, std::byte* const* rx,
                                    std::uint32_t bytes, int rank, int rails) {
    const auto* local = reinterpret_cast<const uint4*>(src);
    auto* output = reinterpret_cast<uint4*>(dst);
    const auto* first = reinterpret_cast<const uint4*>(rx[0] + kHeaderBytes);
    const auto count = bytes / sizeof(uint4);
    const auto split = stripe(bytes, 0, rails).bytes / sizeof(uint4);
    const auto* second = count > split ? reinterpret_cast<const uint4*>(rx[1] + kHeaderBytes) : nullptr;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const auto a = local[i];
        const auto b = payload_load<bypass>(i < split ? first + i : second + (i - split));
        output[i] = make_uint4(sum_bf16_pair(a.x, b.x, rank), sum_bf16_pair(a.y, b.y, rank),
                               sum_bf16_pair(a.z, b.z, rank), sum_bf16_pair(a.w, b.w, rank));
    }
}

constexpr unsigned kGatherBlocks = 32;

// Grid boundaries order staging before publication and unpack before returning
// slot credit. All launches stay on the collective stream during graph replay.
__global__ void stage_gather(SharedControl* control, const std::uint64_t* device_epoch,
                             __grid_constant__ const Buffers buffers, const std::byte* src,
                             std::uint32_t bytes, int rails) {
    const auto epoch = *device_epoch;
    if (!epoch || load(control->host_failed)) return;
    const auto slot = slot_for(epoch);
    const unsigned worker = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned workers = gridDim.x * blockDim.x;
    for (unsigned rail = 0; rail < rails_for(bytes, rails); ++rail) {
        const auto part = stripe(bytes, rail, rails);
        const auto* input = src + part.offset;
        auto* output = buffers.tx[slot][rail] + kHeaderBytes;
        unsigned copied = 0;
        if (vector_aligned(input, output)) {
            const auto vectors = part.bytes / sizeof(uint4);
            for (unsigned i = worker; i < vectors; i += workers)
                reinterpret_cast<uint4*>(output)[i] = reinterpret_cast<const uint4*>(input)[i];
            copied = vectors * sizeof(uint4);
        }
        for (std::size_t i = std::size_t{copied} + worker; i < part.bytes; i += workers)
            output[i] = input[i];
    }
    __threadfence_system();
}

template <bool bypass>
__global__ void unpack_gather(const std::uint64_t* completed_epoch,
                              __grid_constant__ const Buffers buffers, std::byte* dst,
                              __grid_constant__ const Operation op, int rank, int rails) {
    const auto epoch = *completed_epoch;
    if (!epoch) return;
    const auto slot = slot_for(epoch);
    const unsigned worker = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned workers = gridDim.x * blockDim.x;
    auto* output = reinterpret_cast<unsigned*>(dst);
    for (unsigned source = 0; source < 2; ++source) {
        const auto bytes = op.bytes[source];
        const auto columns = op.columns[source];
        const auto column_offset = source == 0 ? 0 : op.columns[0];
        for (unsigned rail = 0; bytes && rail < rails_for(bytes, rails); ++rail) {
            const auto part = stripe(bytes, rail, rails);
            const auto* payload = (source == rank ? buffers.tx[slot][rail] : buffers.rx[slot][rail]) + kHeaderBytes;
            const auto count = part.bytes / sizeof(unsigned);
            for (unsigned i = worker * 4; i < count; i += workers * 4) {
                const auto element = part.offset / sizeof(unsigned) + i;
                const auto row = element / columns;
                const auto column = element % columns;
                auto* target = output + row * op.stride + column_offset + column;
                if (count - i >= 4 && columns - column >= 4) {
                    const auto* input = reinterpret_cast<const uint4*>(payload) + i / 4;
                    const auto value = source == rank ? *input : payload_load<bypass>(input);
                    if (vector_aligned(input, target)) {
                        *reinterpret_cast<uint4*>(target) = value;
                    } else {
                        target[0] = value.x;
                        target[1] = value.y;
                        target[2] = value.z;
                        target[3] = value.w;
                    }
                } else {
                    const auto* input = reinterpret_cast<const unsigned*>(payload);
                    for (unsigned j = 0; j < 4 && i + j < count; ++j) {
                        const auto index = element + j;
                        output[(index / columns) * op.stride + column_offset + index % columns] =
                            source == rank ? input[i + j] : payload_load<bypass>(input + i + j);
                    }
                }
            }
        }
    }
}

__global__ void consume_gather(SharedControl* control, const std::uint64_t* completed_epoch) {
    if (*completed_epoch) store(control->consumed.epoch, *completed_epoch);
}

__device__ bool await_peer(SharedControl* control, std::uint64_t* device_epoch,
                           std::uint64_t epoch, std::uint64_t timeout_ns, bool trace) {
    __shared__ bool valid;
    const auto slot = slot_for(epoch);
    if (threadIdx.x == 0) {
        const auto start = gpu_ns();
        unsigned spins = 0;
        valid = false;
        for (;;) {
            if (load(control->ready) == epoch) {
                if (trace) control->consumed.trace_ready_ns[slot] = gpu_ns();
                valid = true;
                break;
            }
            if ((++spins & 7u) == 0) {
                if (load(control->host_failed)) break;
                if (gpu_ns() - start >= timeout_ns) {
                    *device_epoch = 0;
                    store(control->gpu_failed, epoch);
                    break;
                }
            }
        }
    }
    __syncthreads();
    return valid;
}

template <bool bypass>
__device__ void place_rows(const std::byte* payload, std::byte* dst, const Operation& op,
                           unsigned source_rank, Stripe part) {
    const auto* input = reinterpret_cast<const float*>(payload);
    auto* output = reinterpret_cast<float*>(dst);
    const auto columns = op.columns[source_rank];
    const auto column_offset = source_rank == 0 ? 0 : op.columns[0];
    for (std::uint32_t i = threadIdx.x; i < part.bytes / sizeof(float); i += blockDim.x) {
        const auto element = part.offset / sizeof(float) + i;
        const auto row = element / columns;
        const auto column = element % columns;
        output[row * op.stride + column_offset + column] = payload_load<bypass>(input + i);
    }
}

template <bool bypass>
__device__ void finish(SharedControl* control, std::uint64_t* device_epoch, std::uint64_t epoch,
                       const Buffers& buffers,
                       const std::byte* local, std::byte* dst, const Operation& op,
                       int rank, int rails, std::uint64_t timeout_ns, bool trace) {
    __shared__ bool valid;
    const auto slot = slot_for(epoch);
    if (threadIdx.x == 0) {
        const auto start = gpu_ns();
        unsigned spins = 0;
        valid = false;
        for (;;) {
            if (load(control->ready) == epoch) {
                if (trace) control->consumed.trace_ready_ns[slot] = gpu_ns();
                valid = true;
                break;
            }
            if ((++spins & 7u) == 0) {
                if (load(control->host_failed)) break;
                if (gpu_ns() - start >= timeout_ns) {
                    *device_epoch = 0;
                    store(control->gpu_failed, epoch);
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (!valid) return;

    const unsigned peer = 1 - rank;
    const auto peer_bytes = wire_bytes(op, peer);
    const auto local_bytes = op.bytes[rank];
    if (op.kind == Kind::allgather || (op.kind == Kind::gather_rows && rank == op.root)) {
        for (unsigned rail = 0; local_bytes && rail < rails_for(local_bytes, rails); ++rail) {
            const auto part = stripe(local_bytes, rail, rails);
            const auto* tx = buffers.tx[slot][rail] + kHeaderBytes;
            if (op.kind == Kind::allgather)
                copy_bytes<false>(tx, dst + (rank == 0 ? 0 : op.bytes[0]) + part.offset, part.bytes);
            else
                place_rows<false>(tx, dst, op, rank, part);
        }
    }

    if (op.kind == Kind::bf16 && peer_bytes && peer_bytes % sizeof(uint4) == 0 &&
        vector_aligned(local, dst)) {
        reduce_bf16_vectors<bypass>(local, dst, buffers.rx[slot], peer_bytes, rank, rails);
    } else {
        for (unsigned rail = 0; peer_bytes && rail < rails_for(peer_bytes, rails); ++rail) {
            const auto part = stripe(peer_bytes, rail, rails);
            const auto* rx = buffers.rx[slot][rail] + kHeaderBytes;
            switch (op.kind) {
            case Kind::bf16:
                reduce_bytes<bypass, true>(local + part.offset, rx, dst + part.offset, part.bytes, rank);
                break;
            case Kind::f32:
                reduce_bytes<bypass, false>(local + part.offset, rx, dst + part.offset, part.bytes, rank);
                break;
            case Kind::allgather:
                copy_bytes<bypass>(rx, dst + (peer == 0 ? 0 : op.bytes[0]) + part.offset, part.bytes);
                break;
            case Kind::broadcast:
                copy_bytes<bypass>(rx, dst + part.offset, part.bytes);
                break;
            case Kind::gather_rows:
                if (rank == op.root) place_rows<bypass>(rx, dst, op, peer, part);
                break;
            }
        }
    }
    __syncthreads();
    // Slot credit is carried by the next publish, which is ordered after this
    // kernel on the same stream. The host observes this for diagnostics too.
    if (threadIdx.x == 0) store(control->consumed.epoch, epoch);
}

template <bool bypass>
__global__ void collective(SharedControl* control, std::uint64_t* device_epoch,
                           __grid_constant__ const Buffers buffers,
                           const std::byte* src, std::byte* dst,
                           __grid_constant__ const Operation op, int rank, int rails,
                           std::uint64_t timeout_ns, std::uint64_t session, bool trace) {
    const auto epoch = pack(control, device_epoch, buffers, src, op, rank, rails, session, trace);
    if (epoch)
        finish<bypass>(control, device_epoch, epoch, buffers, src, dst, op, rank, rails, timeout_ns, trace);
}

__device__ std::uint64_t publish_staged(SharedControl* control, std::uint64_t* device_epoch,
                              const Buffers& buffers,
                              const Operation& op, int rank, int rails,
                              std::uint64_t session, bool trace) {
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = load(control->host_failed) ? 0 : *device_epoch;
        if (epoch) *device_epoch = epoch + 1;
    }
    __syncthreads();
    if (!epoch) return 0;

    const unsigned slot = slot_for(epoch);
    const auto wire = wire_bytes(op, rank);
    if (threadIdx.x < rails_for(wire, rails)) {
        const auto rail = threadIdx.x;
        const auto part = stripe(wire, rail, rails);
        *reinterpret_cast<WireHeader*>(buffers.tx[slot][rail]) =
            make_header(session, op, rank, rail, epoch, slot, part.bytes);
    }
    if (threadIdx.x == 0) control->tx.operation = op;
    // Every writer orders its stores to the mapped TX area before publication.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (trace) control->tx.publish_ns = gpu_ns();
        store(control->tx.epoch, epoch);
    }
    return epoch;
}

__global__ void publish_gather(SharedControl* control, std::uint64_t* device_epoch,
                               __grid_constant__ const Buffers buffers,
                               __grid_constant__ const Operation op,
                               int rank, int rails, std::uint64_t timeout_ns,
                               std::uint64_t session, bool trace) {
    if (rank == op.root && threadIdx.x == 0) device_epoch[1] = 0;
    const auto epoch = publish_staged(control, device_epoch, buffers, op, rank, rails, session, trace);
    if (!epoch || !await_peer(control, device_epoch, epoch, timeout_ns, trace)) return;
    if (threadIdx.x == 0) {
        if (rank == op.root)
            device_epoch[1] = epoch;
        else
            store(control->consumed.epoch, epoch);
    }
}

// The large FP32/BF16 path preserves wire headers and per-element summation order.
__global__ void pack_large(SharedControl* control, const std::uint64_t* device_epoch,
                           __grid_constant__ const Buffers buffers,
                           const std::byte* src, std::uint32_t bytes, int rails) {
    const auto epoch = *device_epoch;
    if (!epoch || load(control->host_failed)) return;
    const auto slot = slot_for(epoch);
    const auto part = stripe(bytes, blockIdx.y, rails);
    const auto* input = reinterpret_cast<const uint4*>(src + part.offset);
    auto* output = reinterpret_cast<uint4*>(buffers.tx[slot][blockIdx.y] + kHeaderBytes);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < part.bytes / sizeof(uint4);
         i += gridDim.x * blockDim.x) {
        output[i] = input[i];
    }
    __threadfence_system();
}

__global__ void publish_large(SharedControl* control, std::uint64_t* device_epoch,
                              __grid_constant__ const Buffers buffers,
                              __grid_constant__ const Operation op,
                              int rank, int rails, std::uint64_t timeout_ns,
                              std::uint64_t session, bool trace) {
    const auto epoch = publish_staged(control, device_epoch, buffers, op, rank, rails, session, trace);
    if (epoch) await_peer(control, device_epoch, epoch, timeout_ns, trace);
}

template <bool bypass, bool bf16>
__global__ void reduce_large(SharedControl* control, const std::uint64_t* device_epoch,
                             __grid_constant__ const Buffers buffers,
                             const std::byte* src, std::byte* dst, std::uint32_t bytes, int rank, int rails) {
    __shared__ bool valid;
    const auto next = *device_epoch;
    if (!next || next == 1) return;
    const auto epoch = next - 1;
    if (threadIdx.x == 0) valid = !load(control->host_failed) && load(control->ready) == epoch;
    __syncthreads();
    if (!valid) return;
    const auto slot = slot_for(epoch);
    const auto part = stripe(bytes, blockIdx.y, rails);
    const auto* local = reinterpret_cast<const uint4*>(src + part.offset);
    const auto* peer = reinterpret_cast<const uint4*>(buffers.rx[slot][blockIdx.y] + kHeaderBytes);
    auto* output = reinterpret_cast<uint4*>(dst + part.offset);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < part.bytes / sizeof(uint4);
         i += gridDim.x * blockDim.x) {
        const auto x = local[i], y = payload_load<bypass>(peer + i);
        output[i] = make_uint4(sum_word<bf16>(x.x, y.x, rank), sum_word<bf16>(x.y, y.y, rank),
                               sum_word<bf16>(x.z, y.z, rank), sum_word<bf16>(x.w, y.w, rank));
    }
}

__global__ void consume_large(SharedControl* control, const std::uint64_t* device_epoch) {
    const auto next = *device_epoch;
    if (next > 1 && !load(control->host_failed) && load(control->ready) == next - 1)
        store(control->consumed.epoch, next - 1);
}

// Pipelined multi-slice bulk AllReduce (>= 2 MiB): overlaps GPU pack(s+1),
// dual-rail RDMA(s), and GPU reduce(s-1) within a single epoch and slot on one
// stream (CUDA Graph safe, zero extra DMA memory).
__global__ void pack_bulk_slice(SharedControl* control, const std::uint64_t* epoch_ptr,
                                __grid_constant__ const Buffers buffers,
                                const std::byte* src, std::uint32_t bytes, int rails,
                                unsigned slice, unsigned slices) {
    const auto epoch = *epoch_ptr;
    if (!epoch || load(control->host_failed)) return;
    const auto slot = slot_for(epoch);
    const auto part = stripe(bytes, blockIdx.y, rails);
    const auto sl = slice_of(part.bytes, slice, slices);
    const auto* input = reinterpret_cast<const uint4*>(src + part.offset + sl.offset);
    auto* output = reinterpret_cast<uint4*>(buffers.tx[slot][blockIdx.y] + kHeaderBytes + sl.offset);
    const unsigned vecs = sl.bytes / sizeof(uint4);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < vecs; i += gridDim.x * blockDim.x) {
        output[i] = input[i];
    }
    __threadfence_system();
}

__global__ void publish_bulk_start(SharedControl* control, std::uint64_t* device_epoch,
                                   __grid_constant__ const Buffers buffers,
                                   __grid_constant__ const Operation op,
                                   int rank, int rails, std::uint64_t session, bool trace) {
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = load(control->host_failed) ? 0 : device_epoch[0];
        device_epoch[1] = epoch;
        if (epoch) device_epoch[0] = epoch + 1;
    }
    __syncthreads();
    if (!epoch) return;

    const unsigned slot = slot_for(epoch);
    const auto wire = wire_bytes(op, rank);
    if (threadIdx.x < rails_for(wire, rails)) {
        const auto rail = threadIdx.x;
        const auto part = stripe(wire, rail, rails);
        *reinterpret_cast<WireHeader*>(buffers.tx[slot][rail]) =
            make_header(session, op, rank, rail, epoch, slot, part.bytes);
    }
    if (threadIdx.x == 0) control->tx.operation = op;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        store(control->bulk_tx_slice, pack_slice_word(epoch, 1));
        if (trace) control->tx.publish_ns = gpu_ns();
        store(control->tx.epoch, epoch);
    }
}

__global__ void signal_bulk_tx(SharedControl* control, const std::uint64_t* active_epoch, unsigned ready_slices) {
    const auto epoch = *active_epoch;
    if (epoch && !load(control->host_failed)) {
        __threadfence_system();
        store(control->bulk_tx_slice, pack_slice_word(epoch, ready_slices));
    }
}

__global__ void await_bulk_rx(SharedControl* control, std::uint64_t* device_epoch,
                              unsigned needed_slices, std::uint64_t timeout_ns) {
    const auto epoch = device_epoch[1];
    if (!epoch) return;
    const auto target = pack_slice_word(epoch, needed_slices);
    const auto start = gpu_ns();
    unsigned spins = 0;
    for (;;) {
        if (load(control->bulk_rx_slice) >= target) return;
        if ((++spins & 7u) == 0) {
            if (load(control->host_failed)) {
                device_epoch[1] = 0;
                return;
            }
            if (gpu_ns() - start >= timeout_ns) {
                device_epoch[0] = 0;
                device_epoch[1] = 0;
                store(control->gpu_failed, epoch);
                return;
            }
        }
    }
}

template <bool bypass, bool bf16>
__global__ void reduce_bulk_slice(SharedControl* control, const std::uint64_t* active_epoch,
                                  __grid_constant__ const Buffers buffers,
                                  const std::byte* src, std::byte* dst, std::uint32_t bytes,
                                  int rank, int rails, unsigned slice, unsigned slices) {
    const auto epoch = *active_epoch;
    if (!epoch || load(control->host_failed)) return;
    const auto slot = slot_for(epoch);
    const auto part = stripe(bytes, blockIdx.y, rails);
    const auto sl = slice_of(part.bytes, slice, slices);
    const auto* local = reinterpret_cast<const uint4*>(src + part.offset + sl.offset);
    const auto* peer = reinterpret_cast<const uint4*>(buffers.rx[slot][blockIdx.y] + kHeaderBytes + sl.offset);
    auto* output = reinterpret_cast<uint4*>(dst + part.offset + sl.offset);
    const unsigned vecs = sl.bytes / sizeof(uint4);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < vecs; i += gridDim.x * blockDim.x) {
        const auto x = local[i], y = payload_load<bypass>(peer + i);
        output[i] = make_uint4(sum_word<bf16>(x.x, y.x, rank), sum_word<bf16>(x.y, y.y, rank),
                               sum_word<bf16>(x.z, y.z, rank), sum_word<bf16>(x.w, y.w, rank));
    }
}

__global__ void consume_bulk(SharedControl* control, std::uint64_t* device_epoch,
                             std::uint64_t timeout_ns, bool trace) {
    const auto epoch = device_epoch[1];
    if (!epoch) return;
    if (!await_peer(control, device_epoch, epoch, timeout_ns, trace)) {
        if (threadIdx.x == 0) device_epoch[1] = 0;
        return;
    }
    if (threadIdx.x == 0) store(control->consumed.epoch, epoch);
}

class MappedAllocation {
public:
    ~MappedAllocation() { if (host) cudaFreeHost(host); }
    void allocate(std::size_t bytes) {
        INKLING_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped));
        INKLING_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&gpu), host, 0));
    }
    std::byte* host = nullptr;
    std::byte* gpu = nullptr;
};

class DeviceEpoch {
    // Word 0 is the next collective epoch. Word 1 is the ready gather awaiting unpack.
public:
    ~DeviceEpoch() { if (value) cudaFree(value); }
    void allocate() {
        INKLING_CUDA(cudaMalloc(reinterpret_cast<void**>(&value), 2 * sizeof(*value)));
        constexpr std::uint64_t first = 1;
        INKLING_CUDA(cudaMemcpy(value, &first, sizeof(first), cudaMemcpyHostToDevice));
    }
    std::uint64_t* value = nullptr;
};

std::size_t slot_stride(std::size_t payload) {
    return (payload + kHeaderBytes + kControlAlignment - 1) & ~(std::size_t(kControlAlignment) - 1);
}

class RoceUmaBackend final : public CommBackend {
public:
    RoceUmaBackend(const CommPlan& plan, const CommOptions& options)
        : rank_(options.rank), rails_(options.rail_count), capacity_(plan.max_payload_bytes),
          // Let the proxy report the CQ/rail failure first when it is responsive.
          timeout_ns_(std::chrono::duration_cast<std::chrono::nanoseconds>(
              options.timeout + std::chrono::milliseconds(20)).count()),
          visibility_(options.visibility), trace_(options.trace),
          bulk_pipe_(read_env_bool("INKLING_COMM_BULK_PIPE", true)),
          bulk_slices_override_(read_env_u32("INKLING_COMM_BULK_SLICES", 0)),
          large_min_bytes_(read_env_u32("INKLING_COMM_LARGE_MIN_BYTES", 0)) {
        int mapped = 0;
        INKLING_CUDA(cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, options.cuda_device));
        require(mapped, "CUDA device cannot access mapped host memory");
        // Only naturally aligned system-scope loads/stores touch mapped
        // control words. The epoch counter lives solely in device memory.
        control_.allocate(sizeof(SharedControl));
        auto* control = new (control_.host) SharedControl{};
        gpu_control_ = reinterpret_cast<SharedControl*>(control_.gpu);
        epoch_.allocate();

        const auto cap = static_cast<std::uint32_t>(capacity_);
        const auto second = stripe(cap, 1, rails_).bytes;
        const auto first = std::max(std::min(cap, kTinyBytes), stripe(cap, 0, rails_).bytes);
        const std::size_t stride[kRails]{slot_stride(first), rails_ == 2 ? slot_stride(second + 15) : 0};
        const auto pair_bytes = stride[0] + stride[1];
        const auto registered_bytes = 2 * kSlots * pair_bytes;
        dma_.allocate(registered_bytes);
        if (visibility_ == VisibilityMode::host_bounce) bounce_.allocate(kSlots * pair_bytes);

        Buffers dma_host{};
        Buffers visible_host{};
        for (unsigned slot = 0; slot < kSlots; ++slot) {
            for (unsigned rail = 0; rail < kRails && stride[rail]; ++rail) {
                const auto offset = slot * pair_bytes + (rail ? stride[0] : 0);
                dma_host.tx[slot][rail] = dma_.host + offset;
                dma_host.rx[slot][rail] = dma_.host + kSlots * pair_bytes + offset;
                gpu_buffers_.tx[slot][rail] = dma_.gpu + offset;
                if (visibility_ == VisibilityMode::host_bounce) {
                    visible_host.rx[slot][rail] = bounce_.host + offset;
                    gpu_buffers_.rx[slot][rail] = bounce_.gpu + offset;
                } else {
                    visible_host.rx[slot][rail] = dma_host.rx[slot][rail];
                    gpu_buffers_.rx[slot][rail] = dma_.gpu + kSlots * pair_bytes + offset;
                }
            }
        }
        proxy_ = std::make_unique<RoceProxy>(plan, options, control, dma_host, visible_host,
                                             dma_.host, registered_bytes);
    }

    int rank() const noexcept override { return rank_; }
    int world_size() const noexcept override { return 2; }

    void allreduce_bf16(DeviceSpan<const Bf16> src, DeviceSpan<Bf16> dst,
                        CollectiveSite site, cudaStream_t stream) override {
        reduce(Kind::bf16, src, dst, site, stream);
    }

    void allreduce_f32(DeviceSpan<const float> src, DeviceSpan<float> dst,
                       CollectiveSite site, cudaStream_t stream) override {
        reduce(Kind::f32, src, dst, site, stream);
    }

    void allgather_small(DeviceBytes src, MutableDeviceBytes dst,
                         CollectiveSite site, cudaStream_t stream) override {
        valid_span(src);
        valid_span(dst);
        require(dst.size >= product(src.size, 2), "allgather requires two rank records");
        const auto n = payload_count(src.size);
        Operation op{site, {n, n}};
        op.kind = Kind::allgather;
        enqueue(src.data, dst.data, op, stream);
    }

    void broadcast_tokens(MutableDeviceBytes record, int root,
                          CollectiveSite site, cudaStream_t stream) override {
        require(root == 0 || root == 1, "invalid collective root");
        valid_span(record);
        Operation op{site, {}};
        op.bytes[root] = payload_count(record.size);
        op.kind = Kind::broadcast;
        op.root = root;
        enqueue(record.data, record.data, op, stream);
    }

    void gather_rows_to_root(DeviceSpan<const float> local, MutableDeviceBytes dst,
                             const GatherLayout& layout, int root,
                             CollectiveSite site, cudaStream_t stream) override {
        const auto sizes = row_sizes(layout, 2);
        validate_gather(local, dst, sizes, rank_, root, 2);
        if (rank_ == root && sizes.root_bytes)
            require(!(reinterpret_cast<std::uintptr_t>(dst.data) % alignof(float)),
                    "gather root output must be float-aligned");
        Operation op{site, {payload_count(sizes.rank_bytes[0]), payload_count(sizes.rank_bytes[1])}};
        for (int rank = 0; rank < 2; ++rank) {
            require(layout.columns_per_rank[rank] <= std::numeric_limits<std::uint32_t>::max(),
                    "gather column count is out of range");
            op.columns[rank] = static_cast<std::uint32_t>(layout.columns_per_rank[rank]);
        }
        op.stride = sizes.stride;
        op.kind = Kind::gather_rows;
        op.root = root;
        // The root stages a local copy for convenient packing, and sends a
        // symmetric credit record. The peer also sends its row stripe(s).
        enqueue(reinterpret_cast<const std::byte*>(local.data), dst.data, op, stream);
    }

    CommStatus status() const override { return proxy_->status(); }
    CommStats stats() const override { return proxy_->stats(); }
    CommTrace last_trace() const override { return proxy_->last_trace(); }
    DeviceHealth device_health() const noexcept override {
        return {&gpu_control_->host_failed.value, &gpu_control_->gpu_failed.value};
    }

private:
    std::uint32_t payload_count(std::size_t bytes) const {
        require(bytes <= capacity_, "collective exceeds CommPlan::max_payload_bytes");
        return static_cast<std::uint32_t>(bytes);
    }

    template <class T>
    void reduce(Kind kind, DeviceSpan<const T> src, DeviceSpan<T> dst,
                CollectiveSite site, cudaStream_t stream) {
        valid_span(src);
        valid_span(dst);
        require(dst.size >= src.size, "allreduce destination is too small");
        const auto n = payload_count(product(src.size, sizeof(T)));
        Operation op{site, {n, n}};
        op.kind = kind;
        enqueue(reinterpret_cast<const std::byte*>(src.data), reinterpret_cast<std::byte*>(dst.data),
                op, stream);
    }

    void enqueue(const std::byte* src, std::byte* dst, Operation op, cudaStream_t stream) {
        const bool gather = op.kind == Kind::gather_rows;
        if (gather && op.bytes[rank_])
            stage_gather<<<kGatherBlocks, kThreads, 0, stream>>>(
                gpu_control_, epoch_.value, gpu_buffers_, src, op.bytes[rank_], rails_);
        const auto bytes = op.bytes[rank_];
        if (gather) {
            publish_gather<<<1, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_, op, rank_, rails_,
                                                       timeout_ns_, proxy_->session(), trace_);
        } else if ((op.kind == Kind::f32 || op.kind == Kind::bf16) &&
                   bytes >= (large_min_bytes_ ? large_min_bytes_
                                              : (op.kind == Kind::bf16 ? kLargeThresholdBytesBf16 : kLargeThresholdBytesF32)) &&
                   !(bytes % sizeof(uint4)) &&
                   !((reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)) & 15)) {
            const unsigned active_rails = rails_for(bytes, rails_);
            const bool is_bf16 = (op.kind == Kind::bf16);
            const bool bypass = (visibility_ == VisibilityMode::cache_bypass);
            const unsigned slices = choose_bulk_slices(bytes);
            if (slices > 1) {
                op.slices = static_cast<std::uint8_t>(slices);
                const auto max_rail = std::max(stripe(bytes, 0, rails_).bytes,
                                               active_rails > 1 ? stripe(bytes, 1, rails_).bytes : 0u);
                const auto slice_vecs = (slice_of(max_rail, 0, slices).bytes + sizeof(uint4) - 1) / sizeof(uint4);
                const unsigned bpr = std::max(1u, std::min<unsigned>(48u / active_rails, static_cast<unsigned>((slice_vecs + kThreads - 1) / kThreads)));
                const dim3 grid(bpr, active_rails);
                auto launch_reduce = [&](unsigned sl) {
                    if (bypass) {
                        if (is_bf16)
                            reduce_bulk_slice<true, true><<<grid, kThreads, 0, stream>>>(
                                gpu_control_, epoch_.value + 1, gpu_buffers_, src, dst, bytes, rank_, rails_, sl, slices);
                        else
                            reduce_bulk_slice<true, false><<<grid, kThreads, 0, stream>>>(
                                gpu_control_, epoch_.value + 1, gpu_buffers_, src, dst, bytes, rank_, rails_, sl, slices);
                    } else {
                        if (is_bf16)
                            reduce_bulk_slice<false, true><<<grid, kThreads, 0, stream>>>(
                                gpu_control_, epoch_.value + 1, gpu_buffers_, src, dst, bytes, rank_, rails_, sl, slices);
                        else
                            reduce_bulk_slice<false, false><<<grid, kThreads, 0, stream>>>(
                                gpu_control_, epoch_.value + 1, gpu_buffers_, src, dst, bytes, rank_, rails_, sl, slices);
                    }
                };
                // Pack slice 0 and immediately publish epoch so RDMA begins right away.
                pack_bulk_slice<<<grid, kThreads, 0, stream>>>(
                    gpu_control_, epoch_.value, gpu_buffers_, src, bytes, rails_, 0, slices);
                publish_bulk_start<<<1, kThreads, 0, stream>>>(
                    gpu_control_, epoch_.value, gpu_buffers_, op, rank_, rails_, proxy_->session(), trace_);
                for (unsigned sl = 0; sl < slices; ++sl) {
                    if (sl + 1 < slices) {
                        pack_bulk_slice<<<grid, kThreads, 0, stream>>>(
                            gpu_control_, epoch_.value + 1, gpu_buffers_, src, bytes, rails_, sl + 1, slices);
                        signal_bulk_tx<<<1, 1, 0, stream>>>(gpu_control_, epoch_.value + 1, sl + 2);
                    }
                    await_bulk_rx<<<1, 1, 0, stream>>>(gpu_control_, epoch_.value, sl + 1, timeout_ns_);
                    launch_reduce(sl);
                }
                consume_bulk<<<1, 1, 0, stream>>>(gpu_control_, epoch_.value, timeout_ns_, trace_);
            } else {
                const auto max_rail = std::max(stripe(bytes, 0, rails_).bytes,
                                               active_rails > 1 ? stripe(bytes, 1, rails_).bytes : 0u);
                const auto rail_vecs = max_rail / sizeof(uint4);
                const unsigned bpr = std::max(1u, std::min<unsigned>(48u / active_rails, static_cast<unsigned>((rail_vecs + kThreads - 1) / kThreads)));
                const dim3 grid(bpr, active_rails);
                pack_large<<<grid, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_, src, bytes, rails_);
                publish_large<<<1, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_, op, rank_, rails_,
                                                          timeout_ns_, proxy_->session(), trace_);
                if (bypass) {
                    if (is_bf16)
                        reduce_large<true, true><<<grid, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                                                src, dst, bytes, rank_, rails_);
                    else
                        reduce_large<true, false><<<grid, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                                                 src, dst, bytes, rank_, rails_);
                } else {
                    if (is_bf16)
                        reduce_large<false, true><<<grid, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                                                 src, dst, bytes, rank_, rails_);
                    else
                        reduce_large<false, false><<<grid, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                                                  src, dst, bytes, rank_, rails_);
                }
                consume_large<<<1, 1, 0, stream>>>(gpu_control_, epoch_.value);
            }
        } else if (visibility_ == VisibilityMode::cache_bypass) {
            collective<true><<<1, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                       src, dst, op, rank_, rails_, timeout_ns_, proxy_->session(), trace_);
        } else {
            collective<false><<<1, kThreads, 0, stream>>>(gpu_control_, epoch_.value, gpu_buffers_,
                                                        src, dst, op, rank_, rails_, timeout_ns_, proxy_->session(), trace_);
        }
        if (gather && rank_ == op.root) {
            if (visibility_ == VisibilityMode::cache_bypass)
                unpack_gather<true><<<kGatherBlocks, kThreads, 0, stream>>>(
                    epoch_.value + 1, gpu_buffers_, dst, op, rank_, rails_);
            else
                unpack_gather<false><<<kGatherBlocks, kThreads, 0, stream>>>(
                    epoch_.value + 1, gpu_buffers_, dst, op, rank_, rails_);
            consume_gather<<<1, 1, 0, stream>>>(gpu_control_, epoch_.value + 1);
        }
        INKLING_CUDA(cudaGetLastError());
    }

    int rank_;
    int rails_;
    std::size_t capacity_;
    std::uint64_t timeout_ns_;
    static bool read_env_bool(const char* name, bool fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        return std::strtol(v, nullptr, 10) != 0;
    }

    static std::uint32_t read_env_u32(const char* name, std::uint32_t fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint32_t>(parsed) : fallback;
    }

    unsigned choose_bulk_slices(std::uint32_t bytes) const {
        if (!bulk_pipe_ || bytes < kBulkPipeThresholdBytes) return 1;
        if (bulk_slices_override_ >= 2)
            return std::min<unsigned>(bulk_slices_override_, kMaxBulkSlices);
        return 4u;
    }

    VisibilityMode visibility_;
    bool trace_;
    bool bulk_pipe_;
    std::uint32_t bulk_slices_override_;
    std::uint32_t large_min_bytes_;
    MappedAllocation control_;
    MappedAllocation dma_;
    MappedAllocation bounce_;
    DeviceEpoch epoch_;
    SharedControl* gpu_control_ = nullptr;
    Buffers gpu_buffers_;
    std::unique_ptr<RoceProxy> proxy_;  // Destroy QPs and MRs before releasing mapped data.
};
}  // namespace

std::unique_ptr<CommBackend> make_roce_backend(const CommPlan& plan, const CommOptions& options) {
    require(plan.max_payload_bytes > 0 &&
            plan.max_payload_bytes <= std::numeric_limits<std::uint32_t>::max() - kHeaderBytes - kControlAlignment,
            "invalid RoCE slot capacity");
    require(options.rail_count == 1 || options.rail_count == 2, "RoCE requires one or two rails");
    require(options.rendezvous_port != 0, "invalid RoCE rendezvous port");
    require(options.timeout >= std::chrono::milliseconds(1) && options.timeout <= std::chrono::minutes(1),
            "RoCE timeout must be between 1 and 60000 ms");
    require(options.bootstrap_timeout >= std::chrono::milliseconds(1) &&
            options.bootstrap_timeout <= std::chrono::minutes(1),
            "RoCE bootstrap timeout must be between 1 and 60000 ms");
    for (int rail = 0; rail < options.rail_count; ++rail) {
        const auto& config = options.rails[rail];
        require(!config.device.empty() && config.port > 0 && config.port <= 255 &&
                config.gid_index >= -1 && config.gid_index <= 255, "invalid RoCE rail configuration");
    }
    require(options.rail_count != 2 || options.rails[0].device != options.rails[1].device ||
            options.rails[0].port != options.rails[1].port, "RoCE rails must use distinct ports");
    return std::make_unique<RoceUmaBackend>(plan, options);
}
}  // namespace inkling::comm::detail
