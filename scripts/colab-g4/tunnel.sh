#!/usr/bin/env bash
# Resilient SSH local port-forward from local machine -> Colab G4 Frankengine server (:8834).
# Usage:
#   ./tunnel.sh [SESSION=frank-g4] [LOCAL_PORT=8834] [--bg]
#   ./tunnel.sh [--session frank-g4] [--port 8834] [--bg]
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
LOCAL_PORT="8834"
REMOTE_PORT="${REMOTE_PORT:-8834}"
MODE="fg"
POS_IDX=0

while [ $# -gt 0 ]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --port|-p)    LOCAL_PORT="$2"; shift 2 ;;
    --bg)         MODE="--bg"; shift ;;
    fg)           MODE="fg"; shift ;;
    -h|--help)    sed -n '2,5p' "$0"; exit 0 ;;
    *)
      if [ "$POS_IDX" -eq 0 ]; then SESSION="$1"
      elif [ "$POS_IDX" -eq 1 ]; then LOCAL_PORT="$1"
      else echo "Unknown arg: $1" >&2; exit 1
      fi
      POS_IDX=$((POS_IDX + 1)); shift ;;
  esac
done

PIDFILE="/tmp/colab-tunnel-${SESSION}-${LOCAL_PORT}.pid"
LOGFILE="/tmp/colab-tunnel-${SESSION}-${LOCAL_PORT}.log"
METAFILE="/tmp/colab-tunnel-${SESSION}-${LOCAL_PORT}.env"
# shellcheck disable=SC1090
[ -f "$METAFILE" ] && source "$METAFILE"

probe_models() { curl -sf --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; }

wait_for_tunnel() {
  local pid="$1" timeout="${2:-45}" t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$timeout" ]; do
    kill -0 "$pid" 2>/dev/null || return 2
    probe_models && return 0
    sleep 1
  done
  return 1
}

print_ready() {
  printf '\n────────────────────────────────────────────────────────────────────────\n'
  printf '  Frankengine Colab G4 (RTX PRO 6000 96 GB) READY (%s)\n' "$1"
  printf '  OpenAI Base URL : http://127.0.0.1:%s/v1\n' "$LOCAL_PORT"
  printf '  Telemetry       : http://127.0.0.1:%s/v1/telemetry\n' "$LOCAL_PORT"
  printf '  Model           : %s (preset=%s)\n' "${MODEL_NAME:-inkling-2-mini}" "${MODEL_PRESET:-fp8}"
  printf '  Run bench       : %s/bench.sh --session %s --suite spark-bench\n' "$SCRIPT_DIR" "$SESSION"
  printf '  Stop & bill 0   : %s/stop.sh %s %s\n' "$SCRIPT_DIR" "$SESSION" "$LOCAL_PORT"
  printf '────────────────────────────────────────────────────────────────────────\n\n'
}

SSH_PID=""
cleanup() {
  [ -n "$SSH_PID" ] && kill "$SSH_PID" 2>/dev/null || true
  rm -f "$PIDFILE"
  exit 0
}

run_loop() {
  local printed=0
  trap cleanup INT TERM
  while true; do
    ssh -N \
      -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
      -i "$KEY" \
      -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 \
      -o ServerAliveCountMax=3 \
      -o ConnectTimeout=20 \
      root@colab < /dev/null &
    SSH_PID=$!

    if wait_for_tunnel "$SSH_PID" 60; then
      if [ "$printed" -eq 0 ] && [ "${TUNNEL_QUIET_SUMMARY:-0}" != "1" ]; then
        print_ready "foreground (Ctrl+C closes tunnel; VM stays running)"
        printed=1
      elif [ "$printed" -eq 1 ]; then
        printf '[tunnel] Reconnected at %s: http://127.0.0.1:%s/v1 (200 OK)\n' "$(date '+%H:%M:%S')" "$LOCAL_PORT"
      fi
    fi

    wait "$SSH_PID" || true
    SSH_PID=""
    printf '[tunnel] Disconnected at %s. Reconnecting in 3s...\n' "$(date '+%H:%M:%S')"
    sleep 3
  done
}

if [ "$MODE" = "--bg" ]; then
  if [ -f "$PIDFILE" ]; then
    existing="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
      print_ready "background (PID ${existing}, log: ${LOGFILE})"
      exit 0
    fi
    rm -f "$PIDFILE"
  fi
  TUNNEL_QUIET_SUMMARY=1 nohup "${SCRIPT_DIR}/tunnel.sh" "$SESSION" "$LOCAL_PORT" fg > "$LOGFILE" 2>&1 < /dev/null &
  bg_pid=$!
  echo "$bg_pid" > "$PIDFILE"
  if wait_for_tunnel "$bg_pid" 30; then
    print_ready "background (PID ${bg_pid}, log: ${LOGFILE})"
    exit 0
  fi
  printf '[tunnel] Started in background (PID %s); check %s\n' "$bg_pid" "$LOGFILE"
else
  run_loop
fi
