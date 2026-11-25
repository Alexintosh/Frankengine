#!/usr/bin/env bash
set -euo pipefail

# Run on rank0. Every timed or qualification GPU run uses the shared lock.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} != --locked ]]; then
    exec flock /tmp/frankengine_gpu.lock "$0" --locked "$@"
fi
shift
if (( $# < 4 )); then
    printf 'Usage: %s TOOL RAILS VISIBILITY LOG_PREFIX [tool args...]\n' "$0" >&2
    exit 2
fi
tool=$1 rails=$2 mode=$3 log=$4
shift 4
case "$tool" in
    test_uma_visibility|bench_roce_uma_graph|test_comm_backend) ;;
    *) printf 'Unknown tool: %s\n' "$tool" >&2; exit 2 ;;
esac

worker=${INKLING_COMM_WORKER:-worker-node}
build=${INKLING_COMM_BUILD_DIR:-"$root/build"}
worker_bin=/tmp/inkling_tp_comm/$tool
limit=${INKLING_COMM_RUN_LIMIT:-420s}
proxy_cpu=${INKLING_COMM_RUN_PROXY_CPU:-14}
common=(INKLING_COMM_WORLD_SIZE=2 "INKLING_COMM_RAILS=$rails"
        "INKLING_COMM_VISIBILITY=$mode" "INKLING_COMM_PROXY_CPU=$proxy_cpu"
        "INKLING_COMM_TRACE=${INKLING_COMM_TRACE:-0}"
        "INKLING_COMM_BOOTSTRAP_TIMEOUT_MS=${INKLING_COMM_BOOTSTRAP_TIMEOUT_MS:-10000}"
        "INKLING_COMM_TIMEOUT_MS=${INKLING_COMM_TIMEOUT_MS:-5000}"
        "INKLING_COMM_TEST_RAIL_GAP_US=${INKLING_COMM_TEST_RAIL_GAP_US:-0}"
        "INKLING_COMM_TEST_STALL_MS=${INKLING_COMM_TEST_STALL_MS:-0}"
        "INKLING_COMM_TEST_STALE_EPOCH=${INKLING_COMM_TEST_STALE_EPOCH:-0}"
        "INKLING_COMM_TEST_BAD_TAG_EPOCH=${INKLING_COMM_TEST_BAD_TAG_EPOCH:-0}")
rank0=("INKLING_COMM_RANK=0" "INKLING_COMM_TEST_RANK_GAP_US=${INKLING_COMM_TEST_RANK_GAP_US:-0}")
rank1=("INKLING_COMM_RANK=1")

printf -v remote '%q ' timeout --kill-after=3s "$limit" env "${common[@]}" "${rank1[@]}" "$worker_bin" "$@"
ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$worker" "$remote" >"$log.rank1.out" 2>"$log.rank1.err" &
worker_pid=$!
head_status=0
worker_status=0
timeout --kill-after=3s "$limit" env "${common[@]}" "${rank0[@]}" "$build/$tool" "$@" \
    >"$log.rank0.out" 2>"$log.rank0.err" || head_status=$?
wait "$worker_pid" || worker_status=$?
cat "$log.rank0.err" "$log.rank1.err" >&2
if [[ "$tool" != bench_roce_uma_graph ]]; then
    cat "$log.rank0.out" "$log.rank1.out"
fi

# The crash test deliberately kills rank 1 after bootstrap.
if [[ "$tool" == test_comm_backend && ${1:-} == crash ]] &&
   (( worker_status == 137 || worker_status == 255 )) &&
   rg -qx 'rank 1 injecting SIGKILL after rendezvous' "$log.rank1.err"; then
    worker_status=0
fi
(( head_status == 0 && worker_status == 0 )) || {
    printf 'rank 0 status %d; rank 1 status %d\n' "$head_status" "$worker_status" >&2
    exit 1
}
