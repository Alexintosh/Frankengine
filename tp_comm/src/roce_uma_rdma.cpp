#include "roce_uma_detail.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>

namespace inkling::comm::detail {
namespace {
constexpr std::uint64_t kProtocol = 0x494e4b4c494e4701ULL;
constexpr unsigned kQueueDepth = 4;

std::uint64_t load(const EpochWord& word) {
    return __atomic_load_n(&word.value, __ATOMIC_ACQUIRE);
}

void store(EpochWord& word, std::uint64_t value) {
    __atomic_store_n(&word.value, value, __ATOMIC_RELEASE);
}

void spin_hint() {
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__x86_64__)
    asm volatile("pause");
#endif
}

void dma_read_barrier() {
    // rdma-core util/udma_barrier.h, outer-shareable device DMA on arm64.
#if defined(__aarch64__)
    asm volatile("dmb oshld" ::: "memory");
#elif defined(__x86_64__)
    asm volatile("" ::: "memory");
#else
#error RoCE DMA ordering must be defined for this CPU architecture
#endif
}

void system_check(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

void verbs_check(int error, const char* operation) {
    if (error) throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
}

std::uint64_t clock_ns(clockid_t clock) {
    timespec now{};
    if (clock_gettime(clock, &now)) return 0;
    return std::uint64_t(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
}

sockaddr_in ipv4(const std::string& ip, std::uint16_t port) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    require(inet_pton(AF_INET, ip.c_str(), &address.sin_addr) == 1, "expected a numeric IPv4 address");
    return address;
}

bool matches_ip(const ibv_gid& gid, const std::string& ip) {
    const auto address = ipv4(ip, 0);
    constexpr unsigned char prefix[12]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    return !std::memcmp(gid.raw, prefix, sizeof(prefix)) &&
           !std::memcmp(gid.raw + sizeof(prefix), &address.sin_addr, sizeof(address.sin_addr));
}

class Socket {
public:
    Socket() = default;
    explicit Socket(int fd) : fd_(fd) { system_check(fd_ >= 0, "socket/accept"); }
    ~Socket() { if (fd_ >= 0) close(fd_); }
    Socket(Socket&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Socket& operator=(Socket&& other) noexcept {
        std::swap(fd_, other.fd_);
        return *this;
    }

    int fd() const { return fd_; }

    void wait(short events, Clock::time_point deadline, const char* phase) const {
        for (;;) {
            const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
            if (remaining.count() <= 0) throw std::runtime_error(std::string("rendezvous timeout: ") + phase);
            pollfd ready{fd_, events, 0};
            const int result = poll(&ready, 1, static_cast<int>(remaining.count()));
            if (result > 0) return;
            if (result < 0 && errno != EINTR) system_check(false, "rendezvous poll");
        }
    }

    void bind(const sockaddr_in& local) const {
        system_check(::bind(fd_, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) == 0,
                     "rendezvous bind");
    }

    template <class T>
    T exchange(T value, Clock::time_point deadline) const {
        T remote{};
        transfer(&value, sizeof(value), true, deadline);
        transfer(&remote, sizeof(remote), false, deadline);
        return remote;
    }

    bool peer_closed() const {
        char unused{};
        const auto count = recv(fd_, &unused, 1, MSG_PEEK | MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) return false;
        // All bytes have been consumed at bootstrap. Extra traffic is also terminal.
        return true;
    }

private:
    void transfer(void* data, std::size_t bytes, bool sending, Clock::time_point deadline) const {
        auto* cursor = static_cast<char*>(data);
        while (bytes) {
            const auto count = sending ? send(fd_, cursor, bytes, MSG_NOSIGNAL)
                                       : recv(fd_, cursor, bytes, 0);
            if (count == 0) throw std::runtime_error("rendezvous peer closed");
            if (count < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    wait(sending ? POLLOUT : POLLIN, deadline, sending ? "send" : "receive");
                    continue;
                }
                system_check(false, sending ? "rendezvous send" : "rendezvous recv");
            }
            cursor += count;
            bytes -= count;
        }
    }

    int fd_ = -1;
};

Socket rendezvous(const CommOptions& options, Clock::time_point deadline) {
    const auto server = ipv4(options.rendezvous_ip, options.rendezvous_port);
    if (options.rank == 0) {
        Socket listener(socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        const int reuse = 1;
        system_check(setsockopt(listener.fd(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0,
                     "rendezvous SO_REUSEADDR");
        listener.bind(server);
        system_check(listen(listener.fd(), 1) == 0, "rendezvous listen");
        for (;;) {
            listener.wait(POLLIN, deadline, "accept");
            sockaddr_in remote{};
            socklen_t size = sizeof(remote);
            int fd = accept4(listener.fd(), reinterpret_cast<sockaddr*>(&remote), &size,
                             SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (fd < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            Socket peer(fd);
            if (remote.sin_addr.s_addr == ipv4(options.rails[0].ip[1], 0).sin_addr.s_addr)
                return peer;
        }
    }

    const auto local = ipv4(options.rails[0].ip[1], 0);
    while (Clock::now() < deadline) {
        Socket peer(socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        peer.bind(local);
        if (connect(peer.fd(), reinterpret_cast<const sockaddr*>(&server), sizeof(server)) == 0)
            return peer;
        if (errno == EINPROGRESS || errno == EINTR) {
            peer.wait(POLLOUT, deadline, "connect");
            int error = 0;
            socklen_t size = sizeof(error);
            system_check(getsockopt(peer.fd(), SOL_SOCKET, SO_ERROR, &error, &size) == 0,
                         "rendezvous SO_ERROR");
            if (!error) return peer;
            errno = error;
        }
        if (errno != ECONNREFUSED && errno != EHOSTUNREACH && errno != ENETUNREACH)
            system_check(false, "rendezvous connect");
        std::this_thread::sleep_until(std::min(deadline, Clock::now() + std::chrono::milliseconds(20)));
    }
    throw std::runtime_error("rendezvous timeout: connect");
}

struct Endpoint {
    std::uint64_t address[kSlots];
    std::uint32_t rkey;
    std::uint32_t qp_num;
    ibv_gid gid;
    std::uint32_t mtu;
    std::uint32_t reserved;
};

struct Hello {
    std::uint64_t protocol;
    std::uint64_t session;
    std::uint64_t layout;
    std::uint64_t checkpoint;
    std::uint64_t schedule;
    std::uint64_t capacity;
    std::uint32_t rank;
    std::uint32_t rails;
    Endpoint endpoints[kRails];
};

class Rail {
public:
    ~Rail() {
        if (qp_) ibv_destroy_qp(qp_);
        if (region_) ibv_dereg_mr(region_);
        if (rx_) ibv_destroy_cq(rx_);
        if (tx_) ibv_destroy_cq(tx_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (context_) ibv_close_device(context_);
    }

    Endpoint open(const RoceRail& config, int rank, const Buffers& buffers, unsigned rail,
                  std::byte* registered_base, std::size_t registered_bytes) {
        auto** devices = ibv_get_device_list(nullptr);
        system_check(devices != nullptr, "ibv_get_device_list");
        for (int i = 0; devices[i]; ++i) {
            if (config.device == ibv_get_device_name(devices[i])) {
                context_ = ibv_open_device(devices[i]);
                break;
            }
        }
        ibv_free_device_list(devices);
        if (!context_) throw std::runtime_error("cannot open HCA " + config.device);
        port_ = config.port;
        ibv_port_attr port{};
        verbs_check(ibv_query_port(context_, port_, &port), "ibv_query_port");
        require(port.state == IBV_PORT_ACTIVE && port.link_layer == IBV_LINK_LAYER_ETHERNET,
                "RoCE port is not active Ethernet");
        ibv_gid_entry gid{};
        for (int index = 0; index < port.gid_tbl_len && index <= 255; ++index) {
            if (config.gid_index >= 0 && config.gid_index != index) continue;
            if (!ibv_query_gid_ex(context_, port_, index, &gid, 0) &&
                gid.gid_type == IBV_GID_TYPE_ROCE_V2 && matches_ip(gid.gid, config.ip[rank])) {
                gid_index_ = index;
                break;
            }
        }
        if (gid_index_ < 0)
            throw std::runtime_error("no matching RoCEv2 GID for " + config.device + " " + config.ip[rank]);

        pd_ = ibv_alloc_pd(context_);
        system_check(pd_ != nullptr, "ibv_alloc_pd");
        tx_ = ibv_create_cq(context_, kQueueDepth, nullptr, nullptr, 0);
        system_check(tx_ != nullptr, "ibv_create_cq TX");
        rx_ = ibv_create_cq(context_, kQueueDepth, nullptr, nullptr, 0);
        system_check(rx_ != nullptr, "ibv_create_cq RX");
        region_ = ibv_reg_mr(pd_, registered_base, registered_bytes,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        system_check(region_ != nullptr, "ibv_reg_mr mapped UMA");

        ibv_qp_init_attr config_qp{};
        config_qp.send_cq = tx_;
        config_qp.recv_cq = rx_;
        config_qp.cap.max_send_wr = kQueueDepth;
        config_qp.cap.max_recv_wr = kQueueDepth;
        config_qp.cap.max_send_sge = 1;
        config_qp.cap.max_recv_sge = 1;
        config_qp.qp_type = IBV_QPT_RC;
        qp_ = ibv_create_qp(pd_, &config_qp);
        system_check(qp_ != nullptr, "ibv_create_qp");
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = port_;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        verbs_check(ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PORT |
                                 IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS), "QP INIT");

        Endpoint endpoint{};
        for (unsigned slot = 0; slot < kSlots; ++slot)
            endpoint.address[slot] = reinterpret_cast<std::uintptr_t>(buffers.rx[slot][rail]);
        endpoint.rkey = region_->rkey;
        endpoint.qp_num = qp_->qp_num;
        endpoint.gid = gid.gid;
        endpoint.mtu = port.active_mtu;
        mtu_ = port.active_mtu;
        return endpoint;
    }

    void connect(const Endpoint& peer, const std::string& ip) {
        require(matches_ip(peer.gid, ip), "rendezvous peer GID does not match the rail IP");
        require(peer.mtu >= IBV_MTU_256 && peer.mtu <= IBV_MTU_4096, "invalid peer MTU");
        remote_ = peer;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = std::min(mtu_, static_cast<ibv_mtu>(peer.mtu));
        attr.dest_qp_num = peer.qp_num;
        attr.min_rnr_timer = 1;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = port_;
        attr.ah_attr.grh.dgid = peer.gid;
        attr.ah_attr.grh.sgid_index = gid_index_;
        attr.ah_attr.grh.hop_limit = 64;
        verbs_check(ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                 IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                 IBV_QP_MIN_RNR_TIMER), "QP RTR");

        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        verbs_check(ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                 IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "QP RTS");
        for (unsigned i = 0; i < kQueueDepth; ++i) post_receive();
    }

    void abort() noexcept {
        if (!qp_) return;
        ibv_qp_attr error{};
        error.qp_state = IBV_QPS_ERR;
        ibv_modify_qp(qp_, &error, IBV_QP_STATE);
    }

    void post(const std::byte* data, std::uint32_t bytes, unsigned slot, std::uint64_t epoch, std::uint32_t tag) {
        ibv_sge sge{reinterpret_cast<std::uintptr_t>(data), bytes, region_->lkey};
        ibv_send_wr wr{};
        wr.wr_id = epoch;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(tag);
        wr.wr.rdma.remote_addr = remote_.address[slot];
        wr.wr.rdma.rkey = remote_.rkey;
        ibv_send_wr* bad = nullptr;
        verbs_check(ibv_post_send(qp_, &wr, &bad), "ibv_post_send");
    }

    bool sent(std::uint64_t epoch) {
        ibv_wc wc{};
        if (!poll(tx_, wc)) return false;
        if (wc.opcode != IBV_WC_RDMA_WRITE || wc.wr_id != epoch)
            throw std::runtime_error("unexpected send CQE");
        return true;
    }

    bool received(std::uint32_t tag, std::uint32_t bytes) {
        ibv_wc wc{};
        if (!poll(rx_, wc)) return false;
        if (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM || !(wc.wc_flags & IBV_WC_WITH_IMM) ||
            ntohl(wc.imm_data) != tag || wc.byte_len != bytes)
            throw std::runtime_error("unexpected receive CQE (immediate tag or byte count)");
        post_receive();
        return true;
    }

private:
    void post_receive() {
        // The receive WQE carries no SGE. The payload goes to the target MR.
        ibv_recv_wr wr{};
        ibv_recv_wr* bad = nullptr;
        verbs_check(ibv_post_recv(qp_, &wr, &bad), "ibv_post_recv");
    }

    bool poll(ibv_cq* cq, ibv_wc& wc) {
        const auto count = ibv_poll_cq(cq, 1, &wc);
        if (count < 0) throw std::runtime_error("ibv_poll_cq failed");
        if (count == 0) return false;
        if (wc.status != IBV_WC_SUCCESS)
            throw std::runtime_error(std::string(ibv_wc_status_str(wc.status)) +
                                     ", vendor error " + std::to_string(wc.vendor_err));
        if (wc.qp_num != qp_->qp_num) throw std::runtime_error("unexpected CQE QP");
        return true;
    }

    ibv_context* context_ = nullptr;
    ibv_pd* pd_ = nullptr;
    ibv_cq* tx_ = nullptr;
    ibv_cq* rx_ = nullptr;
    ibv_mr* region_ = nullptr;
    ibv_qp* qp_ = nullptr;
    int port_ = 0;
    int gid_index_ = -1;
    ibv_mtu mtu_{};
    Endpoint remote_{};
};

std::uint32_t immediate(std::uint64_t epoch, unsigned rail, unsigned slot) {
    return (static_cast<std::uint32_t>(epoch) << 2) | (rail << 1) | slot;
}

bool same_header(const WireHeader& a, const WireHeader& b) {
    return a.session == b.session && a.epoch == b.epoch && a.stride == b.stride &&
           a.site.schedule_id == b.site.schedule_id && a.site.op_index == b.site.op_index &&
           a.bytes[0] == b.bytes[0] && a.bytes[1] == b.bytes[1] &&
           a.columns[0] == b.columns[0] && a.columns[1] == b.columns[1] &&
           a.stripe_bytes == b.stripe_bytes && a.kind == b.kind && a.root == b.root &&
           a.src_rank == b.src_rank && a.rail == b.rail && a.slot == b.slot;
}

int choose_cpu(int requested) {
    cpu_set_t allowed;
    system_check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "sched_getaffinity");
    if (requested < 0) {
        for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
            if (CPU_ISSET(cpu, &allowed)) return cpu;
    }
    require(requested >= 0 && requested < CPU_SETSIZE && CPU_ISSET(requested, &allowed),
            "proxy CPU is outside the allowed affinity mask");
    return requested;
}

struct TestHooks {
    std::uint64_t stale_epoch = 0;
    std::uint64_t bad_tag_epoch = 0;
    unsigned rail_gap_us = 0;
    unsigned rank_gap_us = 0;
    unsigned stall_ms = 0;

#ifdef INKLING_COMM_TEST_HOOKS
    static unsigned read(const char* name) {
        const auto* value = std::getenv(name);
        if (!value) return 0;
        std::size_t end = 0;
        const auto count = std::stoul(value, &end);
        require(!value[end] && count <= 100'000, "invalid transport test hook");
        return static_cast<unsigned>(count);
    }

    TestHooks()
        : stale_epoch(read("INKLING_COMM_TEST_STALE_EPOCH")),
          bad_tag_epoch(read("INKLING_COMM_TEST_BAD_TAG_EPOCH")),
          rail_gap_us(read("INKLING_COMM_TEST_RAIL_GAP_US")),
          rank_gap_us(read("INKLING_COMM_TEST_RANK_GAP_US")),
          stall_ms(read("INKLING_COMM_TEST_STALL_MS")) {}
#endif
};

void busy_delay(unsigned microseconds) {
    const auto deadline = Clock::now() + std::chrono::microseconds(microseconds);
    while (Clock::now() < deadline) spin_hint();
}
}  // namespace

class RoceProxy::Impl {
public:
    Impl(const CommPlan& plan, const CommOptions& options, SharedControl* control,
         const Buffers& dma, const Buffers& visible, std::byte* registered_base,
         std::size_t registered_bytes)
        : options_(options), control_(control), dma_(dma), visible_(visible),
          capacity_(plan.max_payload_bytes), cpu_(choose_cpu(options.proxy_cpu)) {
        const auto deadline = Clock::now() + options.bootstrap_timeout;
        Hello local{};
        local.protocol = kProtocol;
        if (options.rank == 0)
            system_check(getrandom(&local.session, sizeof(local.session), 0) == sizeof(local.session),
                         "getrandom communicator ID");
        local.rank = options.rank;
        local.rails = options.rail_count;
        local.layout = plan.layout_hash;
        local.checkpoint = plan.checkpoint_hash;
        local.schedule = plan.schedule_hash;
        local.capacity = plan.max_payload_bytes;
        for (int i = 0; i < options.rail_count; ++i) {
            local.endpoints[i] = rails_[i].open(options.rails[i], options.rank, dma_, i,
                                                registered_base, registered_bytes);
        }

        socket_ = rendezvous(options, deadline);
        const auto peer = socket_.exchange(local, deadline);
        require(peer.protocol == local.protocol && peer.rank == static_cast<unsigned>(1 - options.rank) &&
                peer.rails == local.rails && peer.layout == local.layout &&
                peer.checkpoint == local.checkpoint && peer.schedule == local.schedule &&
                peer.capacity == local.capacity, "rendezvous protocol, rank, rails or plan mismatch");
        session_ = options.rank == 0 ? local.session : peer.session;
        for (int i = 0; i < options.rail_count; ++i)
            rails_[i].connect(peer.endpoints[i], options.rails[i].ip[peer.rank]);
        require(socket_.exchange(session_, deadline) == session_, "rendezvous session mismatch");

        std::promise<void> started;
        auto ready = started.get_future();
        thread_ = std::thread([this, started = std::move(started)]() mutable {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpu_, &mask);
            const int error = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
            if (error) {
                started.set_exception(std::make_exception_ptr(
                    std::runtime_error(std::string("proxy affinity: ") + std::strerror(error))));
                return;
            }
            started.set_value();
            run();
            final_cpu_ns_.store(clock_ns(CLOCK_THREAD_CPUTIME_ID), std::memory_order_relaxed);
        });
        try {
            ready.get();
        } catch (...) {
            thread_.join();
            throw;
        }
        cpu_clock_valid_ = pthread_getcpuclockid(thread_.native_handle(), &cpu_clock_) == 0;
    }

    ~Impl() {
        stop_.store(true, std::memory_order_release);
        thread_.join();
        for (int i = 0; i < options_.rail_count; ++i) rails_[i].abort();
    }

    std::uint64_t session() const noexcept { return session_; }

    CommStatus status() const {
        if (load(control_->host_failed)) {
            std::lock_guard lock(status_mutex_);
            return status_;
        }
        if (const auto epoch = load(control_->gpu_failed))
            return {false, epoch, control_->tx.operation.site, "GPU timed out waiting for peer_ready_epoch"};
        return {};
    }

    CommStats stats() const {
        CommStats out{};
        for (unsigned rail = 0; rail < kRails; ++rail) {
            out.tx_wire_bytes[rail] = tx_bytes_[rail].load(std::memory_order_relaxed);
            out.rx_wire_bytes[rail] = rx_bytes_[rail].load(std::memory_order_relaxed);
            out.dual_first_rx[rail] = first_rx_[rail].load(std::memory_order_relaxed);
        }
        out.completed = completed_.load(std::memory_order_relaxed);
        out.proxy_cpu_ns = cpu_clock_valid_ ? clock_ns(cpu_clock_) : 0;
        if (!out.proxy_cpu_ns) out.proxy_cpu_ns = final_cpu_ns_.load(std::memory_order_relaxed);
        out.proxy_cpu = cpu_;
        return out;
    }

    CommTrace last_trace() const {
        std::lock_guard lock(trace_mutex_);
        auto result = trace_;
        if (result.epoch && load(control_->consumed.epoch) == result.epoch)
            result.gpu_ready_ns = control_->consumed.trace_ready_ns[slot_for(result.epoch)];
        return result;
    }

private:
    void run() {
        std::uint64_t epoch = 1;
        CollectiveSite site{};
        try {
            while (!stop_.load(std::memory_order_acquire)) {
                const auto observed = load(control_->tx.epoch);
                if (const auto failed = load(control_->gpu_failed)) {
                    epoch = failed;
                    site = control_->tx.operation.site;
                    throw std::runtime_error("GPU timed out waiting for peer_ready_epoch");
                }
                if (observed < epoch) {
                    spin_hint();
                    continue;
                }
                const auto operation = control_->tx.operation;
                site = operation.site;
                if (observed != epoch || load(control_->consumed.epoch) != epoch - 1)
                    throw std::runtime_error("local device epoch or slot lease mismatch");
                if (operation.bytes[0] > capacity_ || operation.bytes[1] > capacity_)
                    throw std::runtime_error("device operation exceeds slot capacity");
                if (!exchange(epoch, operation)) break;
                ++epoch;
            }
        } catch (const std::exception& error) {
            {
                std::lock_guard lock(status_mutex_);
                status_ = {false, epoch, site, error.what()};
            }
            store(control_->host_failed, 1);
            shutdown(socket_.fd(), SHUT_RDWR);
            for (int i = 0; i < options_.rail_count; ++i) rails_[i].abort();
        }
    }

    bool exchange(std::uint64_t epoch, const Operation& op) {
        const unsigned slot = slot_for(epoch);
        const unsigned rank = options_.rank;
        const unsigned peer = 1 - rank;
        const unsigned sends = rails_for(wire_bytes(op, rank), options_.rail_count);
        const unsigned receives = rails_for(wire_bytes(op, peer), options_.rail_count);
        const auto deadline = Clock::now() + options_.timeout;
        unsigned tx_pending = (1u << sends) - 1;
        unsigned rx_pending = (1u << receives) - 1;
        CommTrace trace{epoch, op.site};
        if (options_.trace) trace.gpu_publish_ns = control_->tx.publish_ns;

        if (tests_.stall_ms && epoch == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(tests_.stall_ms));
            if (load(control_->gpu_failed))
                throw std::runtime_error("GPU timed out waiting for peer_ready_epoch");
        }
        if (tests_.rank_gap_us && epoch % 3 == 0) busy_delay(tests_.rank_gap_us);
        const bool stagger = tests_.rail_gap_us && sends == 2;
        const unsigned first_send = stagger ? epoch % kRails : 0;
        for (unsigned i = 0; i < sends; ++i) {
            const unsigned rail = i ^ first_send;
            try {
                const auto bytes = stripe(wire_bytes(op, rank), rail, options_.rail_count).bytes;
                if (tests_.stale_epoch == epoch) {
                    const auto stale = epoch - 1;
                    std::memcpy(dma_.tx[slot][rail] + offsetof(WireHeader, epoch), &stale, sizeof(stale));
                }
                auto tag = immediate(epoch, rail, slot);
                if (tests_.bad_tag_epoch == epoch) tag ^= 4u;
                rails_[rail].post(dma_.tx[slot][rail], kHeaderBytes + bytes, slot, epoch, tag);
                tx_bytes_[rail].fetch_add(kHeaderBytes + bytes, std::memory_order_relaxed);
            } catch (const std::exception& error) {
                throw std::runtime_error("rail " + std::to_string(rail + 1) + " post: " + error.what());
            }
            if (stagger && i == 0) busy_delay(tests_.rail_gap_us);
        }

        unsigned spins = 0;
        // Alternate CQ polling priority so it does not systematically mask a
        // near-simultaneous rail 2 arrival in the diagnostic counters.
        const unsigned poll_first = epoch % kRails;
        while (tx_pending || rx_pending) {
            for (unsigned i = 0; i < kRails; ++i) {
                const unsigned rail = i ^ poll_first;
                const unsigned bit = 1u << rail;
                try {
                    if ((rx_pending & bit) && receive(epoch, op, peer, rail, slot)) {
                        if (options_.trace) trace.host_rx_cq_ns[rail] = clock_ns(CLOCK_MONOTONIC);
                        if (rx_pending == 3) first_rx_[rail].fetch_add(1, std::memory_order_relaxed);
                        rx_pending &= ~bit;
                    }
                    if ((tx_pending & bit) && rails_[rail].sent(epoch)) {
                        if (options_.trace) trace.host_tx_cq_ns[rail] = clock_ns(CLOCK_MONOTONIC);
                        tx_pending &= ~bit;
                    }
                } catch (const std::exception& error) {
                    throw std::runtime_error("rail " + std::to_string(rail + 1) + " CQ: " + error.what());
                }
            }

            if ((tx_pending || rx_pending) && (++spins & 63u) == 0) {
                if (stop_.load(std::memory_order_acquire)) return false;
                if (load(control_->gpu_failed))
                    throw std::runtime_error("GPU timed out waiting for peer_ready_epoch");
                if (socket_.peer_closed()) throw std::runtime_error("peer connection closed during RDMA");
                if (Clock::now() >= deadline)
                    throw std::runtime_error("RDMA completion timeout, TX pending rail mask " +
                                             std::to_string(tx_pending) + ", RX pending rail mask " +
                                             std::to_string(rx_pending));
            }
            spin_hint();
        }

        if (options_.trace) {
            trace.host_ready_ns = clock_ns(CLOCK_MONOTONIC);
            std::lock_guard lock(trace_mutex_);
            trace_ = trace;
        }
        completed_.store(epoch, std::memory_order_relaxed);
        store(control_->ready, epoch);
        return true;
    }

    bool receive(std::uint64_t epoch, const Operation& op, int peer, unsigned rail, unsigned slot) {
        const auto bytes = stripe(wire_bytes(op, peer), rail, options_.rail_count).bytes;
        if (!rails_[rail].received(immediate(epoch, rail, slot), kHeaderBytes + bytes)) return false;
        dma_read_barrier();
        WireHeader actual;
        std::memcpy(&actual, dma_.rx[slot][rail], sizeof(actual));
        if (!same_header(actual, make_header(session_, op, peer, rail, epoch, slot, bytes)))
            throw std::runtime_error("wire header session, epoch, site, shape or slot mismatch");
        rx_bytes_[rail].fetch_add(kHeaderBytes + bytes, std::memory_order_relaxed);
        if (options_.visibility == VisibilityMode::host_bounce && bytes)
            std::memcpy(visible_.rx[slot][rail] + kHeaderBytes, dma_.rx[slot][rail] + kHeaderBytes, bytes);
        return true;
    }

    CommOptions options_;
    SharedControl* control_;
    Buffers dma_;
    Buffers visible_;
    std::size_t capacity_;
    int cpu_;
    TestHooks tests_;
    Socket socket_;
    std::array<Rail, kRails> rails_;
    std::uint64_t session_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    clockid_t cpu_clock_{};
    bool cpu_clock_valid_ = false;
    std::atomic<std::uint64_t> final_cpu_ns_{0};
    std::array<std::atomic<std::uint64_t>, kRails> tx_bytes_{};
    std::array<std::atomic<std::uint64_t>, kRails> rx_bytes_{};
    std::array<std::atomic<std::uint64_t>, kRails> first_rx_{};
    std::atomic<std::uint64_t> completed_{0};
    mutable std::mutex status_mutex_;
    CommStatus status_;
    mutable std::mutex trace_mutex_;
    CommTrace trace_;
};

RoceProxy::RoceProxy(const CommPlan& plan, const CommOptions& options, SharedControl* control,
                     const Buffers& dma, const Buffers& visible, std::byte* registered_base,
                     std::size_t registered_bytes)
    : impl_(std::make_unique<Impl>(plan, options, control, dma, visible, registered_base, registered_bytes)) {}
RoceProxy::~RoceProxy() = default;
std::uint64_t RoceProxy::session() const noexcept { return impl_->session(); }
CommStatus RoceProxy::status() const { return impl_->status(); }
CommStats RoceProxy::stats() const { return impl_->stats(); }
CommTrace RoceProxy::last_trace() const { return impl_->last_trace(); }

}  // namespace inkling::comm::detail
