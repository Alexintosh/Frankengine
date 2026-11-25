#include "check_common.cuh"

namespace {
using namespace inkling::comm;
using namespace inkling::comm::tools;
constexpr int kOps = 64;
constexpr int kElems = 16 * 2304;
constexpr int kThreads = 256;

__global__ void prepare(unsigned short* data, std::uint64_t* sequence, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kElems) data[i] = finite_bf16_bits(*sequence, i, rank);
}

__global__ void next(std::uint64_t* sequence) { ++*sequence; }

__global__ void verify_payload(const unsigned short* ranks, const std::uint64_t* sequence, Mismatch* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * kElems) return;
    const auto want = finite_bf16_bits(*sequence, i % kElems, i / kElems);
    if (ranks[i] != want) mark(errors, *sequence, i, ranks[i], want);
}

void run(int writes) {
    auto options = comm_options_from_env();
    check(options.world_size == 2, "visibility test requires two ranks");
    auto comm = make_comm_backend({kElems * sizeof(Bf16), 0, 0, std::uint64_t(writes)}, options);
    Stream stream;
    Event done;
    DeviceBuffer<unsigned short> input(kElems);
    DeviceBuffer<unsigned short> output(2 * kElems);
    DeviceBuffer<std::uint64_t> sequence(1);
    DeviceBuffer<Mismatch> errors(1);
    sequence.fill_zero(stream);
    errors.fill_zero(stream);
    done.record(stream);
    done.wait();

    constexpr int kBlocks = (kElems + kThreads - 1) / kThreads;
    Graph graph(stream, [&] {
        for (int i = 0; i < kOps; ++i) {
            next<<<1, 1, 0, stream>>>(sequence.data());
            prepare<<<kBlocks, kThreads, 0, stream>>>(input.data(), sequence.data(), options.rank);
            CUDA(cudaGetLastError());
            // An allgather exposes every raw BF16 input bit. A rounded SUM
            // could hide some stale values that round to the correct result.
            comm->allgather_small(input.input_bytes(), output.bytes(), {1, unsigned(i)}, stream);
            verify_payload<<<2 * kBlocks, kThreads, 0, stream>>>(output.data(), sequence.data(), errors.data());
            CUDA(cudaGetLastError());
        }
    });
    const auto stats = comm->stats();
    const auto begin = real_ns();
    for (int i = 0; i < writes / kOps; ++i) {
        graph.launch(stream);
        done.record(stream);
        done.wait();
        check_health(*comm);
        verify(errors);
    }
    const auto elapsed = real_ns() - begin;
    print_stats(stats, comm->stats(), elapsed, options.rank);
    if (options.trace) print_trace(comm->last_trace(), options.rank);
    std::printf("PASS rank=%d mode=%s rails=%d overwrites=%d bytes=%zu checked_bf16=%llu elapsed_ms=%.3f\n",
        options.rank, visibility_name(options.visibility), options.rail_count, writes, kElems * sizeof(Bf16),
        static_cast<unsigned long long>(writes) * kElems * 2, elapsed / 1e6);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const int writes = argc == 2 ? std::stoi(argv[1]) : 4096;
        check(argc <= 2 && writes >= kOps && writes % kOps == 0, "usage: test_uma_visibility [multiple of 64]");
        run(writes);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "visibility failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
