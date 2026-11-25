#include "roce_uma_detail.h"

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace inkling::comm {
namespace {
using namespace detail;

class SingleRankCommBackend final : public CommBackend {
public:
    int rank() const noexcept override { return 0; }
    int world_size() const noexcept override { return 1; }

    void allreduce_bf16(DeviceSpan<const Bf16> src, DeviceSpan<Bf16> dst,
                        CollectiveSite, cudaStream_t stream) override {
        copy(src, dst, stream);
    }

    void allreduce_f32(DeviceSpan<const float> src, DeviceSpan<float> dst,
                       CollectiveSite, cudaStream_t stream) override {
        copy(src, dst, stream);
    }

    void allgather_small(DeviceBytes src, MutableDeviceBytes dst,
                         CollectiveSite, cudaStream_t stream) override {
        copy(src, dst, stream);
    }

    void broadcast_tokens(MutableDeviceBytes record, int root,
                          CollectiveSite, cudaStream_t) override {
        require(root == 0, "invalid collective root");
        valid_span(record);
    }

    void gather_rows_to_root(DeviceSpan<const float> local, MutableDeviceBytes dst,
                             const GatherLayout& layout, int root,
                             CollectiveSite, cudaStream_t stream) override {
        const auto sizes = row_sizes(layout, 1);
        validate_gather(local, dst, sizes, 0, root, 1);
        if (!local.size) return;
        const auto width = layout.columns_per_rank[0] * sizeof(float);
        if (sizes.stride == layout.columns_per_rank[0]) {
            copy(DeviceBytes{reinterpret_cast<const std::byte*>(local.data), sizes.rank_bytes[0]},
                 dst, stream);
            return;
        }
        require(static_cast<const void*>(local.data) != dst.data, "strided gather buffers must be disjoint");
        INKLING_CUDA(cudaMemcpy2DAsync(dst.data, sizes.stride * sizeof(float), local.data, width,
                                       width, layout.rows, cudaMemcpyDeviceToDevice, stream));
    }

    CommStatus status() const override { return {}; }
    CommStats stats() const override { return {}; }
    CommTrace last_trace() const override { return {}; }
    DeviceHealth device_health() const noexcept override { return {nullptr, nullptr}; }

private:
    template <class T>
    static void copy(DeviceSpan<const T> src, DeviceSpan<T> dst, cudaStream_t stream) {
        valid_span(src);
        valid_span(dst);
        require(dst.size >= src.size, "collective destination is too small");
        if (src.size && src.data != dst.data)
            INKLING_CUDA(cudaMemcpyAsync(dst.data, src.data, product(src.size, sizeof(T)),
                                        cudaMemcpyDeviceToDevice, stream));
    }
};

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    std::string_view input(value);
    int result = 0;
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size())
        throw std::invalid_argument(std::string("invalid integer in ") + name);
    return result;
}

std::string env_string(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value ? value : std::move(fallback);
}
}  // namespace

CommOptions comm_options_from_env() {
    CommOptions options;
    options.rank = env_int("INKLING_COMM_RANK", 0);
    options.world_size = env_int("INKLING_COMM_WORLD_SIZE", 1);
    options.cuda_device = env_int("INKLING_COMM_CUDA_DEVICE", 0);
    options.rendezvous_ip = env_string("INKLING_COMM_RENDEZVOUS_IP", options.rendezvous_ip);
    const int port = env_int("INKLING_COMM_PORT", options.rendezvous_port);
    require(port > 0 && port <= 65535, "invalid rendezvous port");
    options.rendezvous_port = static_cast<std::uint16_t>(port);
    options.rail_count = env_int("INKLING_COMM_RAILS", options.rail_count);
    options.proxy_cpu = env_int("INKLING_COMM_PROXY_CPU", options.proxy_cpu);
    options.bootstrap_timeout = std::chrono::milliseconds(
        env_int("INKLING_COMM_BOOTSTRAP_TIMEOUT_MS", options.bootstrap_timeout.count()));
    options.timeout = std::chrono::milliseconds(env_int("INKLING_COMM_TIMEOUT_MS", options.timeout.count()));
    options.trace = env_int("INKLING_COMM_TRACE", options.trace) != 0;

    for (auto [index, prefix] : {std::pair{0, "INKLING_COMM_RAIL1"}, {1, "INKLING_COMM_RAIL2"}}) {
        auto& rail = options.rails[index];
        const std::string base(prefix);
        rail.device = env_string((base + "_DEVICE").c_str(), rail.device);
        rail.ip[0] = env_string((base + "_RANK0_IP").c_str(), rail.ip[0]);
        rail.ip[1] = env_string((base + "_RANK1_IP").c_str(), rail.ip[1]);
        rail.port = env_int((base + "_PORT").c_str(), rail.port);
        rail.gid_index = env_int((base + "_GID").c_str(), rail.gid_index);
    }

    const auto visibility = env_string("INKLING_COMM_VISIBILITY", "direct");
    if (visibility == "direct") options.visibility = VisibilityMode::direct;
    else if (visibility == "cache_bypass") options.visibility = VisibilityMode::cache_bypass;
    else if (visibility == "host_bounce") options.visibility = VisibilityMode::host_bounce;
    else throw std::invalid_argument("unknown INKLING_COMM_VISIBILITY");

    const auto backend = env_string("INKLING_COMM_BACKEND", options.world_size == 1 ? "single" : "roce_uma");
    require((options.world_size == 1 && backend == "single") ||
            (options.world_size == 2 && backend == "roce_uma"), "backend and world size do not match");
    return options;
}

std::unique_ptr<CommBackend> CommBackend::init_from_env(const CommPlan& plan) {
    return make_comm_backend(plan, comm_options_from_env());
}

std::unique_ptr<CommBackend> make_comm_backend(const CommPlan& plan, const CommOptions& options) {
    require(options.world_size == 1 || options.world_size == 2, "supported world sizes are 1 and 2");
    require(options.rank >= 0 && options.rank < options.world_size, "invalid rank");
    require(options.cuda_device >= 0, "invalid CUDA device");
    INKLING_CUDA(cudaSetDevice(options.cuda_device));
    if (options.world_size == 1) return std::make_unique<SingleRankCommBackend>();
    return make_roce_backend(plan, options);
}
}  // namespace inkling::comm
