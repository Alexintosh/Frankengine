#include "check_common.cuh"

namespace {
using namespace inkling::comm;
using namespace inkling::comm::tools;
constexpr int kBarriers = 96;
constexpr int kWidth = 2304;
constexpr int kThreads = 256;
constexpr int kWarmup = 30;
constexpr int kSamples = 200;

__global__ void prepare(Bf16* input, unsigned char* head, unsigned count, int record_bytes,
                        unsigned replay, int rank) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const std::uint64_t ordinal = std::uint64_t(replay) * kBarriers + i / (count / kBarriers);
        input[i] = __ushort_as_bfloat16(finite_bf16_bits(ordinal, i % (count / kBarriers), rank));
    }
    if (i < record_bytes) head[i] = static_cast<unsigned char>(pattern(replay, i, rank));
}

__global__ void verify_output(const Bf16* output, const unsigned char* head, unsigned count,
                              int record_bytes, unsigned replay, Mismatch* errors) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned elems = count / kBarriers;
        const std::uint64_t ordinal = std::uint64_t(replay) * kBarriers + i / elems;
        const float a = __bfloat162float(__ushort_as_bfloat16(finite_bf16_bits(ordinal, i % elems, 0)));
        const float b = __bfloat162float(__ushort_as_bfloat16(finite_bf16_bits(ordinal, i % elems, 1)));
        const auto want = __bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(a, b)));
        const auto got = __bfloat16_as_ushort(output[i]);
        if (got != want) mark(errors, ordinal, i % elems, got, want);
    }
    if (i < 2 * record_bytes) {
        const auto want = static_cast<unsigned char>(pattern(replay, i % record_bytes, i / record_bytes));
        if (head[i] != want) mark(errors, replay, count + i, head[i], want);
    }
}

struct Sample {
    double wall;
    float reductions;
    float gather;
};

void benchmark(CommBackend& comm, const CommOptions& options, int m, unsigned qualification_replays) {
    const unsigned elems = m * kWidth;
    const unsigned count = kBarriers * elems;
    const int record_bytes = 8 * m;
    const int blocks = (count + kThreads - 1) / kThreads;
    Stream stream;
    Event inputs_ready;
    Event sum_begin(cudaEventDefault);
    Event sum_end(cudaEventDefault);
    Event head_end(cudaEventDefault);
    Event done;
    DeviceBuffer<Bf16> input(count), output(count);
    DeviceBuffer<unsigned char> head(record_bytes), gathered(2 * record_bytes);
    DeviceBuffer<Mismatch> errors(1);
    errors.fill_zero(stream);
    done.record(stream);
    done.wait();

    Graph graph(stream, [&] {
        sum_begin.record_external(stream);
        for (unsigned i = 0; i < kBarriers; ++i) {
            comm.allreduce_bf16({input.data() + i * elems, elems}, {output.data() + i * elems, elems},
                               {unsigned(m), i}, stream);
        }
        sum_end.record_external(stream);
        comm.allgather_small(head.input_bytes(), gathered.bytes(), {unsigned(m), kBarriers}, stream);
        head_end.record_external(stream);
    });

    std::vector<Sample> samples;
    samples.reserve(kSamples);
    const auto begin = real_ns();
    const auto stats = comm.stats();
    unsigned replay = 0;
    for (; replay < kWarmup + kSamples + qualification_replays; ++replay) {
        const bool timed = replay >= kWarmup && replay < kWarmup + kSamples;
        prepare<<<blocks, kThreads, 0, stream>>>(input.data(), head.data(), count, record_bytes, replay, comm.rank());
        CUDA(cudaGetLastError());
        if (timed) {
            // Drain preparation and the previous comparison outside the sample.
            inputs_ready.record(stream);
            inputs_ready.wait();
        }
        const auto start = Clock::now();
        graph.launch(stream);
        if (timed) {
            done.record(stream);
            done.wait();
            const double wall = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            check_health(comm);
            samples.push_back({wall, sum_end.us_since(sum_begin), head_end.us_since(sum_end)});
        }
        verify_output<<<blocks, kThreads, 0, stream>>>(output.data(), gathered.data(), count,
                                                       record_bytes, replay, errors.data());
        CUDA(cudaGetLastError());
        if (!timed && (replay % 32 == 31 || replay + 1 == kWarmup)) {
            done.record(stream);
            done.wait();
            check_health(comm);
            verify(errors);
        }
    }
    done.record(stream);
    done.wait();
    check_health(comm);
    verify(errors);
    print_stats(stats, comm.stats(), real_ns() - begin, comm.rank());
    if (options.trace) print_trace(comm.last_trace(), comm.rank());
    std::fprintf(stderr, "PASS rank=%d M=%d rails=%d mode=%s replays=%u checked_bf16=%llu checked_head_bytes=%llu\n",
        comm.rank(), m, options.rail_count, visibility_name(options.visibility), replay,
        static_cast<unsigned long long>(count) * replay,
        static_cast<unsigned long long>(record_bytes) * 2 * replay);

    for (unsigned i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        std::printf("%d,%d,%s,%d,%u,%.3f,%.3f,%.3f\n", comm.rank(), options.rail_count,
                    visibility_name(options.visibility), m, i, sample.wall, sample.reductions, sample.gather);
    }
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        check(argc <= 3, "usage: bench_roce_uma_graph [qualification-replays] [M]");
        const int extra = argc >= 2 ? std::stoi(argv[1]) : 0;
        const int m = argc == 3 ? std::stoi(argv[2]) : 0;
        check(extra >= 0 && extra <= 1'000'000, "invalid qualification replay count");
        check(m == 0 || m == 1 || m == 2 || m == 4 || m == 8 || m == 16, "invalid M");
        auto options = comm_options_from_env();
        check(options.world_size == 2, "graph benchmark requires two ranks");
        CUDA(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        auto comm = make_comm_backend({16 * kWidth * sizeof(Bf16), 0, 0,
                                       std::uint64_t(extra) * 32 + m}, options);
        std::printf("rank,rails,mode,m,replay,wall_us,sum_us,head_us\n");
        for (int width : {1, 2, 4, 8, 16})
            if (!m || m == width) benchmark(*comm, options, width, extra);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "graph benchmark failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
