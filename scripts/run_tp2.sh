#!/usr/bin/env bash
# Launch a two-rank TP=2 CLI binary (ingine-bench or ingine-validate) across rank 0 (local) and rank 1 (SSH worker).
# Configure WORKER, HEAD_MODEL_DIR, WORKER_MODEL_DIR in scripts/local.sh or via environment variables.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"
validate_config

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local-binary> [args...]" >&2
  echo "Example: $0 \"\$BENCH_EXE\" --spec --draft-type hybrid --dflash2-adaptive --dflash2-dir \"\$DFLASH2_DIR\" --dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4" >&2
  exit 2
fi

LOCAL_BIN=$1
shift
[[ -x "$LOCAL_BIN" ]] || fail E_BIN "Executable not found: $LOCAL_BIN"

BIN_NAME=$(basename "$LOCAL_BIN")
LOCAL_SHA=$(sha256sum "$LOCAL_BIN" | awk '{print $1}')
REMOTE_DIR="$REMOTE_BASE/cli-$LOCAL_SHA"
REMOTE_BIN="$REMOTE_DIR/$BIN_NAME"

ssh -o BatchMode=yes "$WORKER" "mkdir -p '$REMOTE_DIR'"
REMOTE_SHA=$(ssh -o BatchMode=yes "$WORKER" "sha256sum '$REMOTE_BIN' 2>/dev/null | awk '{print \$1}'" || true)
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
  rsync -az "$LOCAL_BIN" "$WORKER:$REMOTE_BIN"
  REMOTE_SHA=$(ssh -o BatchMode=yes "$WORKER" "sha256sum '$REMOTE_BIN' | awk '{print \$1}'")
  [[ "$REMOTE_SHA" == "$LOCAL_SHA" ]] || fail E_SHA "Worker binary SHA-256 mismatch"
fi

NET_ENV=(
  "INKLING_COMM_WORLD_SIZE=2"
  "INKLING_COMM_BACKEND=roce_uma"
  "INKLING_COMM_RAILS=$ROCE_RAILS"
  "INKLING_COMM_PORT=$COMM_PORT"
  "INKLING_COMM_RENDEZVOUS_IP=$COMM_RENDEZVOUS_IP"
  "INKLING_COMM_PROXY_CPU=$PROXY_CPU"
  "INKLING_COMM_BOOTSTRAP_TIMEOUT_MS=$COMM_BOOTSTRAP_TIMEOUT_MS"
  "INKLING_COMM_TIMEOUT_MS=$COMM_TIMEOUT_MS"
  "INKLING_TP_SIZE=2"
  "INKLING_TP_WIRE=$TP_WIRE"
)

WORKER_PID=""
cleanup() {
  if [[ -n "$WORKER_PID" ]]; then
    kill -TERM "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

printf -v REMOTE_CMD '%q ' env "${NET_ENV[@]}" INKLING_COMM_RANK=1 INKLING_TP_RANK=1 \
  "$REMOTE_BIN" --model-dir "$WORKER_MODEL_DIR" "$@"

ssh -o BatchMode=yes "$WORKER" "$REMOTE_CMD" &
WORKER_PID=$!

env "${NET_ENV[@]}" INKLING_COMM_RANK=0 INKLING_TP_RANK=0 \
  "$LOCAL_BIN" --model-dir "$HEAD_MODEL_DIR" "$@"
STATUS=$?

wait "$WORKER_PID" || true
WORKER_PID=""
exit "$STATUS"
