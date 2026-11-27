#!/usr/bin/env bash
# Safely tear down local SSH tunnel AND stop the billable Colab G4 session.
# Usage:
#   ./stop.sh [SESSION=frank-g4] [LOCAL_PORT=8834]
#   ./stop.sh [--session frank-g4] [--port 8834]
set -euo pipefail

resolve_colab_bin() {
  if [ -n "${COLAB_BIN:-}" ]; then printf '%s\n' "$COLAB_BIN"; return 0; fi
  if command -v colab >/dev/null 2>&1; then command -v colab; return 0; fi
  for cand in "$HOME/.local/bin/colab" "/opt/homebrew/bin/colab" "/usr/local/bin/colab"; do
    [ -x "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  printf '%s\n' "$HOME/.local/bin/colab"
}

COLAB_BIN="$(resolve_colab_bin)"
SESSION="frank-g4"
LOCAL_PORT="8834"
POS_IDX=0

while [ $# -gt 0 ]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --port|-p)    LOCAL_PORT="$2"; shift 2 ;;
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
METAFILE="/tmp/colab-tunnel-${SESSION}-${LOCAL_PORT}.env"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "[stop] Stopping background tunnel PID $PID..."
    kill "$PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi
rm -f "$METAFILE"

pkill -f "127.0.0.1:${LOCAL_PORT}:127.0.0.1:" 2>/dev/null || true
for sf in /tmp/colab-sparkdash-"${SESSION}"-*.pid; do
  if [ -f "$sf" ]; then
    SPID="$(cat "$sf" 2>/dev/null || true)"
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null || true
    rm -f "$sf"
  fi
done
pkill -f ":127.0.0.1:5555" 2>/dev/null || true

if ! command -v "$COLAB_BIN" >/dev/null 2>&1 && [ ! -x "$COLAB_BIN" ]; then
  echo "[stop] Warning: colab CLI not found at '$COLAB_BIN'; local tunnel cleaned up only." >&2
  exit 1
fi

echo "[stop] Stopping Colab G4 session '${SESSION}'..."
"$COLAB_BIN" stop -s "$SESSION" || true

echo ""
echo "=== Active sessions ==="
"$COLAB_BIN" sessions || true
echo "=== Compute unit usage ==="
"$COLAB_BIN" usage || true
