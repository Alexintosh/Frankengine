#!/usr/bin/env bash
# Shared configuration and validation helpers for Frankengine Dual-Node TP=2 launchers.
# Precedence: defaults here < .env < scripts/local.sh < exported environment.

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

_FRANK_VARS=(
  WORKER HOST PORT SERVED_NAME MAX_CTX PARK_MB MAX_BATCH BATCH_CTX BATCH_SPEC BATCH_SPEC_THRESHOLD BATCH_SPEC_MIN MAX_QUEUE
  SPEC SPEC_WIDTHS SPEC_THRESHOLD DRAFT_TYPE DFLASH2_DIR DFLASH2_ADAPTIVE
  DFLASH2_THRESHOLD DFLASH2_MAX_CTX DRAFT_NVFP4 HEAD_NVFP4 DENSE_FP8 KV_FP8 TP_WIRE
  ROCE_RAILS COMM_PORT COMM_RENDEZVOUS_IP PROXY_CPU
  COMM_BOOTSTRAP_TIMEOUT_MS COMM_TIMEOUT_MS
  HEAD_MODEL_DIR WORKER_MODEL_DIR BUILD_DIR STATE_DIR LOG_DIR
  SERVE_EXTRA_ARGS
)
declare -A _FRANK_EXPORTED=()
for _v in "${_FRANK_VARS[@]}"; do
  if [[ ${!_v+x} ]]; then
    _FRANK_EXPORTED[$_v]=${!_v}
  fi
done

# Generic production-qualified Dual-Spark TP=2 defaults (override in scripts/local.sh or .env)
WORKER="worker-node"
HOST="127.0.0.1"
PORT=8834
SERVED_NAME="inkling-2-mini"
MAX_CTX=262144
PARK_MB=2048
MAX_BATCH=32
BATCH_CTX=32768
BATCH_SPEC=4
BATCH_SPEC_THRESHOLD=0.50
BATCH_SPEC_MIN=2
MAX_QUEUE=64

SPEC=1
SPEC_WIDTHS="fine2"
SPEC_THRESHOLD="0.65"
DRAFT_TYPE="hybrid"
DFLASH2_DIR="$REPO_ROOT/models/Inkling-2-Mini-DFlash2"
DFLASH2_ADAPTIVE=1
DFLASH2_THRESHOLD="0.50"
DFLASH2_MAX_CTX=1536
DRAFT_NVFP4=1
HEAD_NVFP4=1
DENSE_FP8=1
KV_FP8=1
TP_WIRE="fp32"

ROCE_RAILS=2
COMM_PORT=29620
COMM_RENDEZVOUS_IP="10.42.101.1"
PROXY_CPU=14
COMM_BOOTSTRAP_TIMEOUT_MS=60000
COMM_TIMEOUT_MS=15000

HEAD_MODEL_DIR="$REPO_ROOT/models/Inkling-2-Mini"
WORKER_MODEL_DIR="$REPO_ROOT/models/Inkling-2-Mini"
BUILD_DIR="$REPO_ROOT/engine/native/build-gb10-tp"
STATE_DIR="$REPO_ROOT/state"
LOG_DIR="$REPO_ROOT/logs"
SERVE_EXTRA_ARGS=""

# Optional local overrides (gitignored)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -f "$REPO_ROOT/scripts/local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/local.sh"
fi

# Re-apply caller-exported environment variables (highest precedence)
for _v in "${_FRANK_EXPORTED[@]}"; do
  :
done
for _v in "${!_FRANK_EXPORTED[@]}"; do
  printf -v "$_v" '%s' "${_FRANK_EXPORTED[$_v]}"
done
unset _v _FRANK_VARS _FRANK_EXPORTED

EXE="$BUILD_DIR/apps/serve/ingine-serve"
BENCH_EXE="$BUILD_DIR/apps/bench/ingine-bench"
VALIDATE_EXE="$BUILD_DIR/apps/validate/ingine-validate"
REMOTE_BASE="/tmp/frankengine_tp"

fail() { printf '[%s] %s\n' "$1" "$2" >&2; exit 1; }

validate_config() {
  case "$SPEC" in 0|1) ;; *) fail E_CONFIG "SPEC must be 0 or 1 (got: $SPEC)" ;; esac
  case "$SPEC_WIDTHS" in coarse|fine|fine2) ;; *) fail E_CONFIG "SPEC_WIDTHS must be coarse, fine, or fine2" ;; esac
  case "$DRAFT_TYPE" in dspark|dflash2|hybrid) ;; *) fail E_CONFIG "DRAFT_TYPE must be dspark, dflash2, or hybrid" ;; esac
  case "$DFLASH2_ADAPTIVE" in 0|1) ;; *) fail E_CONFIG "DFLASH2_ADAPTIVE must be 0 or 1" ;; esac
  case "$DRAFT_NVFP4" in 0|1) ;; *) fail E_CONFIG "DRAFT_NVFP4 must be 0 or 1" ;; esac
  case "$HEAD_NVFP4" in 0|1) ;; *) fail E_CONFIG "HEAD_NVFP4 must be 0 or 1" ;; esac
  case "$DENSE_FP8" in 0|1) ;; *) fail E_CONFIG "DENSE_FP8 must be 0 or 1" ;; esac
  case "$KV_FP8" in 0|1) ;; *) fail E_CONFIG "KV_FP8 must be 0 or 1" ;; esac
  case "$TP_WIRE" in fp32|bf16|bf16-moe) ;; *) fail E_CONFIG "TP_WIRE must be fp32, bf16, or bf16-moe" ;; esac
  [[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || fail E_CONFIG "Invalid PORT: $PORT"
  [[ "$COMM_PORT" =~ ^[0-9]+$ && "$COMM_PORT" -ge 1 && "$COMM_PORT" -le 65535 ]] || fail E_CONFIG "Invalid COMM_PORT: $COMM_PORT"
  [[ "$MAX_CTX" =~ ^[0-9]+$ && "$MAX_CTX" -ge 1024 ]] || fail E_CONFIG "Invalid MAX_CTX: $MAX_CTX"
  [[ "$BATCH_SPEC" =~ ^[0-7]$ ]] || fail E_CONFIG "BATCH_SPEC must be 0..7 (got: $BATCH_SPEC)"
  [[ "$BATCH_SPEC_MIN" =~ ^[0-7]$ ]] || fail E_CONFIG "BATCH_SPEC_MIN must be 0..7 (got: $BATCH_SPEC_MIN)"
  [[ "$BATCH_CTX" =~ ^[0-9]+$ && "$BATCH_CTX" -ge 1024 ]] || fail E_CONFIG "Invalid BATCH_CTX: $BATCH_CTX"
  if [[ "$BATCH_SPEC" -gt 0 ]]; then
    [[ "$MAX_BATCH" =~ ^[0-9]+$ && "$MAX_BATCH" -ge 1 && "$MAX_BATCH" -le 128 ]] || fail E_CONFIG "MAX_BATCH must be 1..128 (got: $MAX_BATCH)"
  else
    [[ "$MAX_BATCH" =~ ^[1-8]$ ]] || fail E_CONFIG "MAX_BATCH must be 1..8 (got: $MAX_BATCH)"
  fi
  [[ "$DFLASH2_MAX_CTX" =~ ^[0-9]+$ && "$DFLASH2_MAX_CTX" -ge 1 ]] || fail E_CONFIG "Invalid DFLASH2_MAX_CTX: $DFLASH2_MAX_CTX"
}
