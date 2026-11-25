#!/usr/bin/env bash
# Launch the two-rank TP=2 OpenAI-compatible HTTP server (ingine-serve) across rank 0 (local) and rank 1 (SSH worker).
# Configure WORKER, HEAD_MODEL_DIR, WORKER_MODEL_DIR, MAX_CTX, DRAFT_TYPE in scripts/local.sh or via environment variables.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"
validate_config

[[ -x "$EXE" ]] || fail E_BIN "Missing ingine-serve binary at $EXE; build engine/ first"

LOCAL_SHA=$(sha256sum "$EXE" | awk '{print $1}')
REMOTE_DIR="$REMOTE_BASE/serve-$LOCAL_SHA"
REMOTE_BIN="$REMOTE_DIR/ingine-serve"

ssh -o BatchMode=yes "$WORKER" "mkdir -p '$REMOTE_DIR'"
REMOTE_SHA=$(ssh -o BatchMode=yes "$WORKER" "sha256sum '$REMOTE_BIN' 2>/dev/null | awk '{print \$1}'" || true)
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
  rsync -az "$EXE" "$WORKER:$REMOTE_BIN"
  REMOTE_SHA=$(ssh -o BatchMode=yes "$WORKER" "sha256sum '$REMOTE_BIN' | awk '{print \$1}'")
  [[ "$REMOTE_SHA" == "$LOCAL_SHA" ]] || fail E_SHA "Worker binary SHA-256 mismatch"
fi

ARGS=(
  --engine
  --host "$HOST"
  --port "$PORT"
  --model-id "$SERVED_NAME"
  --max-ctx "$MAX_CTX"
  --max-model-len "$MAX_CTX"
  --max-batch "$MAX_BATCH"
  --batch-ctx "$BATCH_CTX"
  --batch-spec "$BATCH_SPEC"
  --batch-spec-threshold "$BATCH_SPEC_THRESHOLD"
  --batch-spec-min "$BATCH_SPEC_MIN"
  --max-queue "$MAX_QUEUE"
  --park-mb "$PARK_MB"
  --no-vision
  --spec-widths "$SPEC_WIDTHS"
  --spec-threshold "$SPEC_THRESHOLD"
  --tp-wire "$TP_WIRE"
)
[[ "$DENSE_FP8" == "1" ]] && ARGS+=(--dense-fp8)
[[ "$KV_FP8" == "1" ]] && ARGS+=(--kv-fp8)
[[ "$HEAD_NVFP4" == "1" ]] && ARGS+=(--head-nvfp4)
[[ "$DRAFT_NVFP4" == "1" ]] && ARGS+=(--draft-nvfp4)
[[ "$SPEC" == "1" ]] && ARGS+=(--spec)
if [[ "$SPEC" == "1" && "$DRAFT_TYPE" != "dspark" ]]; then
  ARGS+=(
    --draft-type "$DRAFT_TYPE"
    --dflash2-dir "$DFLASH2_DIR"
    --dflash2-threshold "$DFLASH2_THRESHOLD"
    --dflash2-max-ctx "$DFLASH2_MAX_CTX"
  )
  [[ "$DFLASH2_ADAPTIVE" == "1" ]] && ARGS+=(--dflash2-adaptive)
fi
if [[ -n "$SERVE_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($SERVE_EXTRA_ARGS)
fi
ARGS+=("$@")

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
  "INKLING_SERVE_CTRL_STDIO=1"
)

printf -v REMOTE_CMD '%q ' env "${NET_ENV[@]}" INKLING_COMM_RANK=1 INKLING_TP_RANK=1 \
  "$REMOTE_BIN" --model-dir "$WORKER_MODEL_DIR" "${ARGS[@]}"

exec python3 - "$WORKER" "$REMOTE_CMD" "$EXE" "$HEAD_MODEL_DIR" "${NET_ENV[@]}" -- "${ARGS[@]}" <<'PY'
import os, signal, subprocess, sys

worker, remote_cmd, exe, head_model = sys.argv[1:5]
sep = sys.argv.index("--", 5)
net_pairs = sys.argv[5:sep]
args = sys.argv[sep + 1:]

env = dict(os.environ)
for pair in net_pairs:
    k, v = pair.split("=", 1)
    env[k] = v
env["INKLING_COMM_RANK"] = "0"
env["INKLING_TP_RANK"] = "0"

w = subprocess.Popen(
    ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10", worker, remote_cmd],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
)
h = subprocess.Popen(
    ["bash", "-c", 'exec 3<&"$1" 4>&"$2"; shift 2; exec "$@"', "bash",
     str(w.stdout.fileno()), str(w.stdin.fileno()), exe, "--model-dir", head_model, *args],
    env=env, pass_fds=(w.stdout.fileno(), w.stdin.fileno()),
)

def stop(*_):
    h.terminate()
    w.terminate()

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
rc = h.wait()
w.terminate()
w.wait()
sys.exit(rc)
PY
