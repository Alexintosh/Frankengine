# libinkling_comm

`inkling_comm::inkling_comm` is a standalone CMake target that produces `libinkling_comm.a`.
The public header is `inkling_comm/comm_backend.h`. The target links CUDA Runtime, libibverbs and
POSIX threads. Its test tools use CUDA 13 and target GB10 by default.

From an engine CMake project, use `add_subdirectory(path/to/tp_comm)` and
`target_link_libraries(your_target PRIVATE inkling_comm::inkling_comm)`.
Compilation defaults to `sm_121a`. Pass `INKLING_COMM_CUDA_ARCHITECTURES` to target other
CUDA architectures at build time.

## Runtime contract

- Create the backend before capture. Supply the largest byte payload per rank, and stable layout,
  checkpoint and schedule hashes for rendezvous. Use the same operation order, shapes and
  `CollectiveSite` values on both ranks. Captured device addresses must remain valid across replays.
- Serialize eager enqueues and graph replays on one CUDA stream chain per rank. AllReduce supports
  matching source and destination addresses. Rank-order gather produces a packed record for each rank.
  Row gather takes packed local FP32 rows and puts the columns into rank order on the root, using
  the configured root row stride. The other rank can pass an empty destination.
- TP=1 bypasses communication. Matching source/destination addresses are a no-op and separate
  buffers use stream-ordered copies.
- Check `status()` after the replay-completion event and before publishing results. Device-side
  consumers can acquire-load the two words from `device_health()`. A nonzero word is terminal.
  Quiesce queued work before destroying the backend, then recreate both ranks after a failure.
- The proxy measures CPU time, bytes posted/received (including transport headers), and the first
  receive CQE observed on dual-rail operations. `last_trace()` is available with tracing enabled.
  Read it after completion and compare GPU times only with GPU times, and host times only with host
  times on the same rank.

The factory reads `INKLING_COMM_RANK`, `INKLING_COMM_WORLD_SIZE`, `INKLING_COMM_CUDA_DEVICE`,
`INKLING_COMM_RENDEZVOUS_IP`, `INKLING_COMM_PORT`, `INKLING_COMM_RAILS`,
`INKLING_COMM_PROXY_CPU`, `INKLING_COMM_BOOTSTRAP_TIMEOUT_MS`, `INKLING_COMM_TIMEOUT_MS`,
`INKLING_COMM_TRACE` and
`INKLING_COMM_VISIBILITY` (`direct`, `cache_bypass`, `host_bounce`).
`INKLING_COMM_BACKEND` can explicitly select `single` or `roce_uma`.
Each `INKLING_COMM_RAIL1` / `INKLING_COMM_RAIL2` prefix takes `_DEVICE`, `_RANK0_IP`,
`_RANK1_IP`, `_PORT` and `_GID`. A GID index of `-1` scans for the configured IPv4 RoCEv2 GID.
Defaults match the two Spark nodes.

## Slot lifetime

Each rank publishes the next execution epoch only after the preceding collective's GPU receive phase completes.
A receive at epoch *n* is gated on the peer's publish for *n*. When a peer publishes *n+2*, it has
already received *n+1* from this rank, so this rank has consumed *n*. Two alternating RX slots are
therefore sufficient to guard against overwrite. Every operation uses a bidirectional wire
exchange, including a header-only send when a rank has no payload. Local send CQ completion gates
TX slot reuse and incoming completions on all participating rails gate GPU receipt. The GPU stages the wire header alongside the payload before publication. The header checks the session, full epoch, shape and collective site even when the 30-bit immediate
sequence tag wraps.

## Spark qualification

Build and deploy the test binaries from rank0.

```sh
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ssh worker-node mkdir -p /tmp/inkling_tp_comm
scp build/{test_comm_backend,test_uma_visibility,bench_roce_uma_graph} \
    worker-node:/tmp/inkling_tp_comm/
mkdir -p logs
tools/run_pair.sh test_comm_backend 2 direct logs/smoke smoke
INKLING_COMM_TRACE=1 INKLING_COMM_TEST_RAIL_GAP_US=18 INKLING_COMM_TEST_RANK_GAP_US=11 \
    tools/run_pair.sh test_uma_visibility 2 direct logs/visibility-direct 4096
tools/run_pair.sh bench_roce_uma_graph 1 direct logs/graph-single 0
tools/run_pair.sh bench_roce_uma_graph 2 direct logs/graph-dual 10000
python3 tools/summarize_graph.py logs/graph-single logs/graph-dual
```

The runner holds `/tmp/frankengine_gpu.lock` for both test ranks. Each benchmark width uses one captured
graph, 30 warmups and 200 measured launches. The numeric argument adds untimed, verified
qualification launches per width. The CSV retains 200 measured samples per rank. Optional fault and
stagger injection is built only when `INKLING_COMM_BUILD_TOOLS` is enabled.
