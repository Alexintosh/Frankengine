#!/usr/bin/env bash
# Deploy MiaAI-Lab/sparkDash directly INSIDE a running Colab G4 VM alongside Frankengine (:8834)
# so sparkDash monitors the RTX PRO 6000 96 GB natively via nvidia-smi and runs Decode/Quality/Tool
# benchmarks over loopback (zero SSH latency), then forwards sparkDash UI to http://127.0.0.1:13000.
#
# Usage:
#   ./sparkdash.sh [--session frank-g4] [--ui-port 13000] [--api-port 8834] [--bg]
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
UI_PORT="13000"
API_PORT="8834"
REMOTE_UI_PORT="5555"
REMOTE_API_PORT="8834"
MODE="fg"

while [ $# -gt 0 ]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --ui-port)    UI_PORT="$2"; shift 2 ;;
    --api-port)   API_PORT="$2"; shift 2 ;;
    --bg)         MODE="--bg"; shift ;;
    -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ ! -f "$KEY" ]; then
  echo "[sparkdash] Error: SSH key '$KEY' not found. Run ./start.sh first." >&2
  exit 1
fi

REMOTE_SETUP="$(mktemp)"
trap 'rm -f "$REMOTE_SETUP"' EXIT

cat > "$REMOTE_SETUP" <<'REMOTE_EOF'
set -euo pipefail
REMOTE_UI_PORT="$1"
REMOTE_API_PORT="$2"

echo "[1/3] Checking Node.js >= 22..."
need_node=1
if command -v node >/dev/null 2>&1; then
  MAJ="$(node -v | sed 's/^v//' | cut -d. -f1)"
  [ "${MAJ:-0}" -ge 22 ] && need_node=0
fi
if [ "$need_node" -eq 1 ]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/content/sparkdash-setup.log 2>&1
  apt-get -qq install -y nodejs >>/content/sparkdash-setup.log 2>&1
fi
echo "  ✓ Node $(node -v) / npm $(npm -v)"

echo "[2/3] Cloning & building MiaAI-Lab/sparkDash..."
if [ ! -d /content/sparkDash/.git ]; then
  git clone --depth 1 https://github.com/MiaAI-Lab/sparkDash.git /content/sparkDash >>/content/sparkdash-setup.log 2>&1
fi
cd /content/sparkDash
if [ ! -d /content/sparkDash/dist ]; then
  npm ci --no-audit --no-fund >>/content/sparkdash-setup.log 2>&1
  npm run build >>/content/sparkdash-setup.log 2>&1
fi
echo "  ✓ sparkDash built"

echo "[3/3] Configuring local RTX PRO 6000 unit & starting sparkDash on 127.0.0.1:${REMOTE_UI_PORT}..."
cat > /content/sparkDash/.env <<EOF
PORT=${REMOTE_UI_PORT}
BIND_HOST=127.0.0.1
SPARKDASH_ALLOW_OPEN_REMOTE=1
LLM_PORT=${REMOTE_API_PORT}
POLL_INTERVAL_STORAGE=5000
EOF

mkdir -p /content/sparkDash/config
if [ ! -f /content/sparkDash/config/sparks.json ]; then
  cat > /content/sparkDash/config/sparks.json <<EOF
[
  {
    "id": "rtx-pro-6000-g4",
    "name": "RTX PRO 6000 (Colab G4)",
    "host": "127.0.0.1",
    "local": true,
    "unitType": "gpu",
    "llmPorts": [${REMOTE_API_PORT}]
  }
]
EOF
fi

pkill -f "node server/index.js" 2>/dev/null || true
sleep 1
nohup node server/index.js > /content/sparkdash.log 2>&1 < /dev/null &
SD_PID=$!
echo "$SD_PID" > /content/sparkdash.pid

ready=0
for _ in $(seq 1 30); do
  kill -0 "$SD_PID" 2>/dev/null || { echo "sparkDash exited early:"; tail -n 40 /content/sparkdash.log; exit 1; }
  if curl -sf "http://127.0.0.1:${REMOTE_UI_PORT}/" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "sparkDash did not start in time"; tail -n 40 /content/sparkdash.log; exit 1; }
echo "  ✓ sparkDash live on remote 127.0.0.1:${REMOTE_UI_PORT} (LLM port ${REMOTE_API_PORT})"
REMOTE_EOF

ssh -i "$KEY" \
  -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ServerAliveInterval=20 \
  root@colab \
  "bash -s -- $(printf '%q ' "$REMOTE_UI_PORT" "$REMOTE_API_PORT")" \
  < "$REMOTE_SETUP"

PIDFILE="/tmp/colab-sparkdash-${SESSION}-${UI_PORT}.pid"
LOGFILE="/tmp/colab-sparkdash-${SESSION}-${UI_PORT}.log"

forward_cmd() {
  ssh -N \
    -L "127.0.0.1:${UI_PORT}:127.0.0.1:${REMOTE_UI_PORT}" \
    -L "127.0.0.1:${API_PORT}:127.0.0.1:${REMOTE_API_PORT}" \
    -i "$KEY" \
    -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    root@colab < /dev/null
}

printf '\n────────────────────────────────────────────────────────────────────────\n'
printf '  sparkDash (RTX PRO 6000 96 GB) : http://127.0.0.1:%s\n' "$UI_PORT"
printf '  Frankengine OpenAI API         : http://127.0.0.1:%s/v1\n' "$API_PORT"
printf '  Stop & bill 0                  : %s/stop.sh %s %s\n' "$SCRIPT_DIR" "$SESSION" "$API_PORT"
printf '────────────────────────────────────────────────────────────────────────\n\n'

if [ "$MODE" = "--bg" ]; then
  pkill -f "127.0.0.1:${UI_PORT}:127.0.0.1:${REMOTE_UI_PORT}" 2>/dev/null || true
  nohup bash -c "while true; do ssh -N -L 127.0.0.1:${UI_PORT}:127.0.0.1:${REMOTE_UI_PORT} -L 127.0.0.1:${API_PORT}:127.0.0.1:${REMOTE_API_PORT} -i '${KEY}' -o 'ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 root@colab </dev/null || true; sleep 3; done" \
    > "$LOGFILE" 2>&1 < /dev/null &
  echo "$!" > "$PIDFILE"
  echo "[sparkdash] Background tunnel PID $(cat "$PIDFILE") (log: ${LOGFILE})"
else
  forward_cmd
fi
