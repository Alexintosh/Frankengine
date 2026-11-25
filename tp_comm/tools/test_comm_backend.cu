#include "check_common.cuh"

#include <csignal>
#include <cstring>
#include <thread>

namespace {
using namespace inkling::comm;
using namespace inkling::comm::tools;


void smoke(CommBackend& comm) {
    constexpr int kBf16 = 147;
    const int kF32 = comm.world_size() == 2 ? 2048 * 2304 : 71;
    constexpr int kRows = 17;
    constexpr int kStride = 21;
    constexpr int kRecord = 259;
    constexpr int kCandidate = 15;
    const int rank = comm.rank();
    const bool two = comm.world_size() == 2;
    const int root = two ? 1 : 0;
    const GatherLayout layout{kRows, {7, 9}, kStride};
    const auto columns = layout.columns_per_rank[rank];
    Stream stream;
    Event done;
    DeviceBuffer<Bf16> bf16(kBf16), bf16_copy(kBf16);
    DeviceBuffer<float> f32(kF32), local(kRows * columns), gathered(kRows * kStride);
    DeviceBuffer<unsigned char> candidate(kCandidate), head(kCandidate * comm.world_size()), tokens(kRecord);
    std::vector<Bf16> bf16_input(kBf16);
    std::vector<float> f32_input(kF32), local_input(kRows * columns);
    std::vector<unsigned char> candidate_input(kCandidate), token_input(kRecord);
    auto bf16_value = [](int i, int rank) {
        return __float2bfloat16_rn((i - 77) * (rank == 0 ? 1.03125f : -0.0859375f));
    };
    auto f32_value = [](int i, int rank) { return (i - 28) * (rank == 0 ? 0.35f : 1.17f); };
    for (int i = 0; i < kBf16; ++i) bf16_input[i] = bf16_value(i, rank);
    for (int i = 0; i < kF32; ++i) f32_input[i] = f32_value(i, rank);
    for (int i = 0; i < kCandidate; ++i) candidate_input[i] = rank * 13 + i;
    for (int i = 0; i < kRecord; ++i) token_input[i] = rank * 17 + i;
    for (std::size_t i = 0; i < local_input.size(); ++i) local_input[i] = rank * 1024 + i;
    candidate.write(candidate_input);
    tokens.write(token_input);
    local.write(local_input);
    gathered.fill_zero(stream);
    bf16.write(bf16_input);
    f32.write(f32_input);

    Graph graph(stream, [&] {
        comm.allreduce_bf16(bf16.input(), bf16_copy.span(), {2, 0}, stream);
        comm.allreduce_f32(f32.input(), f32.span(), {2, 1}, stream);
        comm.allgather_small(candidate.input_bytes(), head.bytes(), {2, 2}, stream);
        // Consecutive alternating-size broadcasts from a single root.
        comm.broadcast_tokens(tokens.bytes(), root, {2, 3}, stream);
        comm.broadcast_tokens({tokens.bytes().data + 1, 17}, root, {2, 4}, stream);
        comm.broadcast_tokens(tokens.bytes(), root, {2, 5}, stream);
        comm.gather_rows_to_root(local.input(), rank == root ? gathered.bytes() : MutableDeviceBytes{},
                                 layout, root, {2, 6}, stream);
        comm.allreduce_bf16({}, {}, {2, 7}, stream);
    });
    for (int replay = 0; replay < 8; ++replay) {
        f32.write(f32_input);
        graph.launch(stream);
        done.record(stream);
        done.wait();
        check_health(comm);
        const auto all_bf16 = bf16_copy.read();
        for (int i = 0; i < kBf16; ++i) {
            const auto expected = two ? __float2bfloat16_rn(__bfloat162float(bf16_value(i, 0)) +
                                                           __bfloat162float(bf16_value(i, 1)))
                                      : bf16_value(i, 0);
            check(__bfloat16_as_ushort(all_bf16[i]) == __bfloat16_as_ushort(expected),
                  "BF16 reduction differs");
        }
        const auto all_f32 = f32.read();
        for (int i = 0; i < kF32; ++i) {
            // Match the separately rounded FP32 inputs that were uploaded.
            // Volatile operands prevent host compiler FMA contraction here.
            const volatile float a = f32_value(i, 0);
            const volatile float b = f32_value(i, 1);
            const float expected = two ? a + b : a;
            if (std::memcmp(&all_f32[i], &expected, sizeof(float)) != 0) {
                char message[128];
                std::snprintf(message, sizeof(message), "FP32 reduction differs at %d: got %a expected %a",
                              i, all_f32[i], expected);
                throw std::runtime_error(message);
            }
        }
        const auto records = head.read();
        for (int i = 0; i < static_cast<int>(records.size()); ++i)
            check(records[i] == i / kCandidate * 13 + i % kCandidate, "rank-order small gather differs");
        const auto received = tokens.read();
        for (int i = 0; i < kRecord; ++i)
            check(received[i] == static_cast<unsigned char>(root * 17 + i), "broadcast differs");
        if (rank == root) {
            const auto rows = gathered.read();
            for (int r = 0; r < kRows; ++r) {
                for (int c = 0; c < kStride; ++c) {
                    int expected = 0;
                    if (c < 7) expected = r * 7 + c;
                    else if (two && c < 16) expected = 1024 + r * 9 + c - 7;
                    check(rows[r * kStride + c] == expected, "root row layout differs");
                }
            }
        }
    }
    std::printf("PASS smoke rank=%d world=%d completed=%llu\n", rank, comm.world_size(),
                static_cast<unsigned long long>(comm.stats().completed));
}

void gather_bench(CommBackend& comm) {
    constexpr std::size_t columns = 100029;
    constexpr unsigned replays = 64;
    const bool root = comm.rank() == 0;
    check(comm.world_size() == 2, "gather benchmark requires two ranks");
    Stream stream;
    Event start(0), end(0);
    for (const std::size_t rows : {1, 2, 4, 8}) {
        DeviceBuffer<float> local(rows * columns), gathered(rows * columns * 2);
        const auto bits = [](std::size_t i) {
            return static_cast<std::uint32_t>(i) * 2654435761u;
        };
        std::vector<float> input(rows * columns);
        for (std::size_t i = 0; i < input.size(); ++i) {
            const auto value = bits(i);
            std::memcpy(&input[i], &value, sizeof(value));
        }
        local.write(input);
        Graph graph(stream, [&] {
            if (root)
                CUDA(cudaMemcpy2DAsync(gathered.data(), 2 * columns * sizeof(float), local.data(),
                                       columns * sizeof(float), columns * sizeof(float), rows,
                                       cudaMemcpyDeviceToDevice, stream));
            comm.gather_rows_to_root(root ? DeviceSpan<const float>{} : local.input(),
                root ? MutableDeviceBytes{gathered.bytes().data + columns * sizeof(float),
                                         (2 * rows - 1) * columns * sizeof(float)} : MutableDeviceBytes{},
                {rows, {0, columns}, 2 * columns}, 0, {7, 0}, stream);
        });
        std::vector<float> samples;
        for (unsigned i = 0; i < replays + 8; ++i) {
            start.record(stream);
            graph.launch(stream);
            end.record(stream);
            end.wait();
            check_health(comm);
            if (i >= 8) samples.push_back(end.us_since(start));
        }
        if (root) {
            const auto output = gathered.read();
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t shard = 0; shard < 2; ++shard)
                    check(std::memcmp(output.data() + (2 * r + shard) * columns,
                                      input.data() + r * columns, columns * sizeof(float)) == 0,
                          "gather benchmark FP32 bits differ");
            std::sort(samples.begin(), samples.end());
            std::printf("gather M=%zu median_us=%.3f p90_us=%.3f bit_parity=PASS\n",
                        rows, samples[samples.size() / 2], samples[samples.size() * 9 / 10]);
        }
    }
}

void failure(CommBackend& comm, const std::string& mode) {
    check(comm.world_size() == 2, "failure test requires two ranks");
    Stream stream;
    Event done;
    DeviceBuffer<Bf16> input(256), output(256);
    input.fill_zero(stream);
    output.write(std::vector<Bf16>(256, __float2bfloat16_rn(31.f)));
    if (comm.rank() == 1) {
        if (mode == "crash") {
            std::fprintf(stderr, "rank 1 injecting SIGKILL after rendezvous\n");
            std::fflush(stderr);
            std::raise(SIGKILL);
        }
        if (mode == "proxy-timeout")
            std::this_thread::sleep_for(comm_options_from_env().timeout * 3);
    }
    Graph graph(stream, [&] {
        for (unsigned i = 0; i < 3; ++i) comm.allreduce_bf16(input.input(), output.span(), {3, i}, stream);
    });
    const auto start = Clock::now();
    graph.launch(stream);
    done.record(stream);
    done.wait();
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto status = comm.status();
    check(!status.ok && status.epoch == 1 && status.site.schedule_id == 3 && status.site.op_index == 0,
          "transport did not latch the first failed epoch and site");
    if (mode == "gpu-timeout")
        check(status.message.find("GPU timed out") != std::string::npos, "GPU spin deadline was not observed");
    if (mode == "proxy-timeout" && comm.rank() == 0)
        check(status.message.find("RDMA completion timeout") != std::string::npos, "proxy deadline was not observed");
    check(__bfloat162float(output.read().front()) == 31.f, "failed chain wrote an output");
    graph.launch(stream);
    done.record(stream);
    done.wait();
    check(comm.status().epoch == 1, "failed session accepted another replay");
    std::printf("PASS failure rank=%d mode=%s epoch=%llu elapsed_ms=%.3f message=%s\n",
                comm.rank(), mode.c_str(), static_cast<unsigned long long>(status.epoch),
                elapsed, status.message.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 2 ? argv[1] : "smoke";
        check(argc <= 2 && (mode == "smoke" || mode == "gather-bench" || mode == "crash" || mode == "gpu-timeout" ||
                            mode == "fault" || mode == "absent" || mode == "proxy-timeout"), "unknown test mode");
        const CommPlan plan{mode == "gather-bench" ? 8u * 100029u * sizeof(float) : mode == "smoke" ? 2048u * 2304u * sizeof(float) : 1024u, 1, 2, 3};
        if (mode == "absent") {
            try {
                auto comm = CommBackend::init_from_env(plan);
            } catch (const std::runtime_error& error) {
                check(std::string(error.what()).find("rendezvous timeout") != std::string::npos,
                      "expected missing peer rendezvous timeout");
                std::printf("PASS absent peer: %s\n", error.what());
                return EXIT_SUCCESS;
            }
            throw std::runtime_error("connected unexpectedly");
        }
        auto comm = CommBackend::init_from_env(plan);
        if (mode == "smoke") smoke(*comm);
        else if (mode == "gather-bench") gather_bench(*comm);
        else failure(*comm, mode);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "comm test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
