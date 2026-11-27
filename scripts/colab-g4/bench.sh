#!/usr/bin/env bash
# Run spark-bench (cmpbench.sh) and/or Frankengine bench_matrix.py directly INSIDE a live Colab G4 VM,
# pull all JSON/log artifacts to local ./colab-g4-results/, and optionally stop the VM (--auto-stop).
#
# Usage:
#   ./bench.sh [--session frank-g4] [--suite spark-bench|matrix|both] [--preset fp8|exact|both]
#              [--steps "sbf think dt lc needle guard"] [--out-dir ./colab-g4-results] [--auto-stop]
#
# Examples:
#   ./bench.sh --suite spark-bench --preset fp8 --auto-stop
#   ./bench.sh --suite spark-bench --preset both --steps "sbf think dt lc needle guard quality" --auto-stop
#   ./bench.sh --suite matrix --preset fp8
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_colab_bin() {
  if [ -n "${COLAB_BIN:-}" ]; then printf '%s\n' "$COLAB_BIN"; return 0; fi
  if command -v colab >/dev/null 2>&1; then command -v colab; return 0; fi
  for cand in "$HOME/.local/bin/colab" "/opt/homebrew/bin/colab" "/usr/local/bin/colab"; do
    [ -x "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  printf '%s\n' "$HOME/.local/bin/colab"
}

COLAB_BIN="$(resolve_colab_bin)"
KEY="${COLAB_SSH_KEY:-$HOME/.ssh/id_ed25519_colab}"

SESSION="frank-g4"
SUITE="spark-bench"
PRESET="fp8"
STEPS="sbf think dt lc needle guard"
OUT_ROOT="${SCRIPT_DIR}/../../colab-g4-results"
AUTO_STOP="0"
REMOTE_PORT="8834"

while [ $# -gt 0 ]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --suite)      SUITE="$2"; shift 2 ;;
    --preset)     PRESET="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --out-dir)    OUT_ROOT="$2"; shift 2 ;;
    --auto-stop)  AUTO_STOP="1"; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$SUITE" in spark-bench|matrix|both) ;; *) echo "Invalid --suite: $SUITE" >&2; exit 1 ;; esac
case "$PRESET" in fp8|exact|both) ;; *) echo "Invalid --preset: $PRESET" >&2; exit 1 ;; esac

if [ ! -f "$KEY" ]; then
  echo "[bench] Error: SSH key '$KEY' missing. Run ./start.sh first." >&2
  exit 1
fi

if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ >/dev/null 2>&1 &
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-rtxpro6000-${PRESET}"
LOCAL_OUT="${OUT_ROOT}/${RUN_ID}"
mkdir -p "$LOCAL_OUT"

REMOTE_BENCH="$(mktemp)"
trap 'rm -f "$REMOTE_BENCH"' EXIT

cat > "$REMOTE_BENCH" <<'REMOTE_EOF'
set -euo pipefail
SUITE="$1"
PRESET_MODE="$2"
STEPS="$3"
REMOTE_PORT="$4"
RUN_ID="$5"

OUT_BASE="/content/frank-bench-out/${RUN_ID}"
mkdir -p "$OUT_BASE"
nvidia-smi > "${OUT_BASE}/nvidia-smi.txt"

if [ "$SUITE" = "spark-bench" ] || [ "$SUITE" = "both" ]; then
  mkdir -p /content/spark-bench/results
  if [ ! -d /content/spark-bench/inkling-2-mini-spark-bench/.git ]; then
    git clone https://github.com/knapcio/inkling-2-mini-spark-bench.git /content/spark-bench/inkling-2-mini-spark-bench
  fi
  if [ ! -d /content/spark-bench/speedbench/.git ]; then
    git clone https://github.com/knapcio/speedbench.git /content/spark-bench/speedbench
  fi
fi

restart_preset() {
  local p="$1"
  pkill -x ingine-serve 2>/dev/null || true
  sleep 1
  # shellcheck disable=SC1091
  source /content/frank-serve-env.sh
  if [ "$p" = "exact" ]; then
    export DENSE_FP8=0 KV_FP8=0 HEAD_NVFP4=0 DENSE_PICO12=1 CERT_HEAD=1
  else
    export DENSE_FP8=1 KV_FP8=1 HEAD_NVFP4=1 DENSE_PICO12=0 CERT_HEAD=0
  fi
  local t0=$SECONDS
  nohup /content/Frankengine/scripts/serve_tp1.sh > "${OUT_BASE}/server-${p}.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > /content/frank-server.pid
  local ready=0
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || { echo "ingine-serve exited for preset=$p"; tail -n 50 "${OUT_BASE}/server-${p}.log"; exit 1; }
    if curl -sf "http://127.0.0.1:${REMOTE_PORT}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  [ "$ready" -eq 1 ] || { echo "Timeout waiting for ingine-serve ($p)"; exit 1; }
  local boot_s=$((SECONDS - t0))
  echo "STARTUP_S=${boot_s}" | tee "${OUT_BASE}/startup-${p}.txt"
  nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.gr,power.draw --format=csv > "${OUT_BASE}/gpu-${p}.csv"
}

run_one_preset() {
  local p="$1"
  echo "================================================================"
  echo "=== RTX PRO 6000 Blackwell · Frankengine TP=1 · preset=${p} ==="
  echo "================================================================"
  restart_preset "$p"

  if [ "$SUITE" = "spark-bench" ] || [ "$SUITE" = "both" ]; then
    local label="frankengine-rtxpro6000-${p}"
    rm -rf "/content/spark-bench/results/${label}"
    BENCH_ROOT=/content/spark-bench \
    SPEEDBENCH=/content/spark-bench/speedbench \
    STEPS="$STEPS" \
    ENGINE_NAME="$label" \
      bash /content/spark-bench/inkling-2-mini-spark-bench/scripts/cmpbench.sh \
        "$label" "http://127.0.0.1:${REMOTE_PORT}" inkling-2-mini none 262 \
        | tee "${OUT_BASE}/cmpbench-${p}.log"
    cp -a "/content/spark-bench/results/${label}" "${OUT_BASE}/spark-bench-${p}"
  fi

  if [ "$SUITE" = "matrix" ] || [ "$SUITE" = "both" ]; then
    cd "$OUT_BASE"
    python3 /content/Frankengine/scripts/bench_matrix.py \
      --url "http://127.0.0.1:${REMOTE_PORT}/v1" \
      --tag "rtxpro6000-${p}" \
      --server-log "${OUT_BASE}/server-${p}.log" \
      | tee "${OUT_BASE}/bench_matrix-${p}.log"
  fi

  curl -sf "http://127.0.0.1:${REMOTE_PORT}/metrics" > "${OUT_BASE}/metrics-${p}.txt" || true
  curl -sf "http://127.0.0.1:${REMOTE_PORT}/v1/telemetry" > "${OUT_BASE}/telemetry-${p}.json" || true
}

if [ "$PRESET_MODE" = "both" ]; then
  run_one_preset fp8
  run_one_preset exact
else
  run_one_preset "$PRESET_MODE"
fi

echo "REMOTE_OUT_READY=${OUT_BASE}"
REMOTE_EOF

echo "[bench] Running suite='${SUITE}' preset='${PRESET}' steps='${STEPS}' on Colab G4 session '${SESSION}'..."
BENCH_RC=0
ssh -i "$KEY" \
  -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=6 \
  root@colab \
  "bash -s -- $(printf '%q ' "$SUITE" "$PRESET" "$STEPS" "$REMOTE_PORT" "$RUN_ID")" \
  < "$REMOTE_BENCH" || BENCH_RC=$?

echo "[bench] Pulling artifacts into ${LOCAL_OUT}..."
ssh -i "$KEY" \
  -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@colab \
  "tar -czf - -C /content/frank-bench-out ${RUN_ID} 2>/dev/null" \
  | tar -xzf - -C "$OUT_ROOT" || echo "[bench] Warning: partial artifact pull"

echo "[bench] Artifacts saved to: ${LOCAL_OUT}"

if [ "$AUTO_STOP" = "1" ]; then
  echo "[bench] --auto-stop enabled: stopping Colab G4 session '${SESSION}'..."
  "${SCRIPT_DIR}/stop.sh" --session "$SESSION"
fi

exit "$BENCH_RC"
