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

void bulk_test(CommBackend& comm) {
    const int rank = comm.rank();
    const bool two = comm.world_size() == 2;
    const std::size_t sizes[] = {
        16u << 10,              // 16 KiB (decode hot path)
        64u << 10,              // 64 KiB (multi-CTA threshold)
        20u * 2304u * 4u,       // 184.32 KiB (BatchDecoder c=4..8)
        256u << 10,             // 256 KiB
        80u * 2304u * 4u,       // 737.28 KiB (BatchDecoder c=16..32)
        160u * 2304u * 4u,      // 1.47 MiB (BatchDecoder c=32 M=4)
        2u << 20,               // 2 MiB (pipelined bulk threshold)
        2048u * 2304u * 4u,     // 18.87 MB (Prefill T=2048 FP32)
    };
    const std::size_t max_bytes = 2048u * 2304u * 4u;
    Stream stream;
    Event done;
    DeviceBuffer<float> f32_src(max_bytes / sizeof(float)), f32_dst(max_bytes / sizeof(float));
    DeviceBuffer<Bf16> bf16_src(max_bytes / sizeof(Bf16)), bf16_dst(max_bytes / sizeof(Bf16));

    auto f32_val = [](std::size_t i, int r) {
        return static_cast<float>(static_cast<int>(i % 251) - 125) * (r == 0 ? 0.3125f : -1.171875f);
    };
    auto bf16_val = [](std::size_t i, int r) {
        return __float2bfloat16_rn(static_cast<float>(static_cast<int>(i % 197) - 98) * (r == 0 ? 0.5625f : -0.1875f));
    };

    std::vector<float> f32_in(max_bytes / sizeof(float));
    std::vector<Bf16> bf16_in(max_bytes / sizeof(Bf16));
    for (std::size_t i = 0; i < f32_in.size(); ++i) f32_in[i] = f32_val(i, rank);
    for (std::size_t i = 0; i < bf16_in.size(); ++i) bf16_in[i] = bf16_val(i, rank);

    unsigned op_idx = 0;
    for (std::size_t bytes : sizes) {
        const std::size_t nf = bytes / sizeof(float);
        const std::size_t nb = bytes / sizeof(Bf16);
        // Test both out-of-place (src != dst) and in-place (src == dst) inside CUDA Graph replays.
        for (int inplace = 0; inplace < 2; ++inplace) {
            Graph graph(stream, [&] {
                if (inplace) {
                    comm.allreduce_f32({f32_dst.data(), nf}, {f32_dst.data(), nf}, {5, op_idx++}, stream);
                    comm.allreduce_bf16({bf16_dst.data(), nb}, {bf16_dst.data(), nb}, {5, op_idx++}, stream);
                } else {
                    comm.allreduce_f32({f32_src.data(), nf}, {f32_dst.data(), nf}, {5, op_idx++}, stream);
                    comm.allreduce_bf16({bf16_src.data(), nb}, {bf16_dst.data(), nb}, {5, op_idx++}, stream);
                }
            });
            for (int replay = 0; replay < 3; ++replay) {
                if (inplace) {
                    f32_dst.write(f32_in);
                    bf16_dst.write(bf16_in);
                } else {
                    f32_src.write(f32_in);
                    bf16_src.write(bf16_in);
                    f32_dst.fill_zero(stream);
                    bf16_dst.fill_zero(stream);
                }
                graph.launch(stream);
                done.record(stream);
                done.wait();
                check_health(comm);

                const auto got_f32 = f32_dst.read();
                for (std::size_t i = 0; i < nf; ++i) {
                    const float exp = two ? (f32_val(i, 0) + f32_val(i, 1)) : f32_val(i, 0);
                    check(std::memcmp(&got_f32[i], &exp, sizeof(float)) == 0,
                          (std::string("bulk_test FP32 bit mismatch at bytes=") + std::to_string(bytes) +
                              " idx=" + std::to_string(i)).c_str());
                }
                const auto got_bf16 = bf16_dst.read();
                for (std::size_t i = 0; i < nb; ++i) {
                    const auto exp = two ? __float2bfloat16_rn(__bfloat162float(bf16_val(i, 0)) +
                                                               __bfloat162float(bf16_val(i, 1)))
                                         : bf16_val(i, 0);
                    check(__bfloat16_as_ushort(got_bf16[i]) == __bfloat16_as_ushort(exp),
                          (std::string("bulk_test BF16 bit mismatch at bytes=") + std::to_string(bytes) +
                              " idx=" + std::to_string(i)).c_str());
                }
            }
        }
    }
    std::printf("PASS bulk-test rank=%d world=%d completed=%llu (0 mismatches across 16KiB..18.87MiB FP32+BF16)\n",
                rank, comm.world_size(), static_cast<unsigned long long>(comm.stats().completed));
}

void bulk_bench(CommBackend& comm) {
    check(comm.world_size() == 2, "bulk benchmark requires two ranks");
    const std::size_t sizes[] = {
        16u << 10,
        64u << 10,
        20u * 2304u * 4u,
        256u << 10,
        80u * 2304u * 4u,
        160u * 2304u * 4u,
        2u << 20,
        2048u * 2304u * 4u,
    };
    const std::size_t max_bytes = 2048u * 2304u * 4u;
    Stream stream;
    Event start(true), end(true);
    DeviceBuffer<float> f32_src(max_bytes / sizeof(float)), f32_dst(max_bytes / sizeof(float));
    DeviceBuffer<Bf16> bf16_src(max_bytes / sizeof(Bf16)), bf16_dst(max_bytes / sizeof(Bf16));
    f32_src.fill_zero(stream);
    bf16_src.fill_zero(stream);
    end.record(stream);
    end.wait();

    unsigned op_idx = 0;
    for (int is_bf16 = 0; is_bf16 < 2; ++is_bf16) {
        for (std::size_t bytes : sizes) {
            const std::size_t elems = bytes / (is_bf16 ? sizeof(Bf16) : sizeof(float));
            Graph graph(stream, [&] {
                if (is_bf16)
                    comm.allreduce_bf16({bf16_src.data(), elems}, {bf16_dst.data(), elems}, {6, op_idx++}, stream);
                else
                    comm.allreduce_f32({f32_src.data(), elems}, {f32_dst.data(), elems}, {6, op_idx++}, stream);
            });
            constexpr int kWarm = 15;
            constexpr int kReps = 60;
            std::vector<double> samples;
            samples.reserve(kReps);
            for (int i = 0; i < kWarm + kReps; ++i) {
                start.record(stream);
                graph.launch(stream);
                end.record(stream);
                end.wait();
                check_health(comm);
                if (i >= kWarm) samples.push_back(end.us_since(start));
            }
            if (comm.rank() == 0) {
                std::sort(samples.begin(), samples.end());
                const double med_us = samples[samples.size() / 2];
                const double p90_us = samples[samples.size() * 9 / 10];
                const double gbps = (static_cast<double>(bytes) / 1e9) / (med_us / 1e6);
                std::printf("bulk-bench type=%-4s bytes=%8zu (%6.2f KiB) median_us=%8.2f p90_us=%8.2f eff_GBs=%6.2f\n",
                            is_bf16 ? "bf16" : "f32", bytes, bytes / 1024.0, med_us, p90_us, gbps);
                std::fflush(stdout);
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 2 ? argv[1] : "smoke";
        check(argc <= 2 && (mode == "smoke" || mode == "bulk-test" || mode == "bulk-bench" ||
                            mode == "gather-bench" || mode == "crash" || mode == "gpu-timeout" ||
                            mode == "fault" || mode == "absent" || mode == "proxy-timeout"), "unknown test mode");
        const CommPlan plan{mode == "gather-bench" ? 8u * 100029u * sizeof(float)
                            : (mode == "smoke" || mode == "bulk-test" || mode == "bulk-bench")
                                ? 2048u * 2304u * sizeof(float)
                                : 1024u, 1, 2, 3};
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
        else if (mode == "bulk-test") bulk_test(*comm);
        else if (mode == "bulk-bench") bulk_bench(*comm);
        else if (mode == "gather-bench") gather_bench(*comm);
        else failure(*comm, mode);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "comm test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
