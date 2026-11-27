#!/usr/bin/env bash
# One-command launcher: provisions Google Colab G4 (RTX PRO 6000 Blackwell 96 GB, sm_120a),
# installs CUDA 13 if needed, builds Frankengine TP=1, downloads Inkling-2-Mini + DFlash 2,
# launches ingine-serve on remote 127.0.0.1:8834, and opens an SSH tunnel to http://127.0.0.1:<PORT>/v1.
#
# Usage:
#   ./start.sh [fp8|exact] [--branch main] [--ctx 262144] [--max-batch 32] [--batch-ctx 32768]
#              [--vision 1|0] [--port 8834] [--session frank-g4] [--bg] [--no-tunnel] [--no-wait]
#
# Examples:
#   ./start.sh                          # FP8 flagship (256K + 32x32K BatchDFlash2, Vision ON), fg tunnel
#   ./start.sh exact --bg               # 100% bit-exact lossless (PICO12 + CERT_HEAD), bg tunnel
#   ./start.sh fp8 --no-tunnel          # Provision & start server only (ready for ./bench.sh)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_colab_bin() {
  if [ -n "${COLAB_BIN:-}" ]; then
    printf '%s\n' "$COLAB_BIN"
    return 0
  fi
  if command -v colab >/dev/null 2>&1; then
    command -v colab
    return 0
  fi
  for cand in "$HOME/.local/bin/colab" "/opt/homebrew/bin/colab" "/usr/local/bin/colab"; do
    if [ -x "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  printf '%s\n' "$HOME/.local/bin/colab"
}

COLAB_BIN="$(resolve_colab_bin)"
KEY="${COLAB_SSH_KEY:-$HOME/.ssh/id_ed25519_colab}"
REPO_URL="${FRANKENGINE_REPO_URL:-https://github.com/Alexintosh/Frankengine.git}"

PRESET="fp8"
BRANCH="main"
CTX="262144"
MAX_BATCH="32"
BATCH_CTX="32768"
VISION="1"
LOCAL_PORT="8834"
REMOTE_PORT="8834"
SESSION="frank-g4"
TUNNEL_MODE="fg"
OPEN_TUNNEL="1"
WAIT_FOR_GPU="1"

while [ $# -gt 0 ]; do
  case "$1" in
    fp8|exact) PRESET="$1"; shift ;;
    --preset)
      [ $# -ge 2 ] || { echo "Missing value for --preset" >&2; exit 1; }
      PRESET="$2"; shift 2 ;;
    --branch)
      [ $# -ge 2 ] || { echo "Missing value for --branch" >&2; exit 1; }
      BRANCH="$2"; shift 2 ;;
    --ctx)
      [ $# -ge 2 ] || { echo "Missing value for --ctx" >&2; exit 1; }
      CTX="$2"; shift 2 ;;
    --max-batch)
      [ $# -ge 2 ] || { echo "Missing value for --max-batch" >&2; exit 1; }
      MAX_BATCH="$2"; shift 2 ;;
    --batch-ctx)
      [ $# -ge 2 ] || { echo "Missing value for --batch-ctx" >&2; exit 1; }
      BATCH_CTX="$2"; shift 2 ;;
    --vision)
      [ $# -ge 2 ] || { echo "Missing value for --vision" >&2; exit 1; }
      VISION="$2"; shift 2 ;;
    --port|-p)
      [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      LOCAL_PORT="$2"; shift 2 ;;
    --session|-s)
      [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      SESSION="$2"; shift 2 ;;
    --bg)         TUNNEL_MODE="--bg"; shift ;;
    --no-tunnel)  OPEN_TUNNEL="0"; shift ;;
    --no-wait)    WAIT_FOR_GPU="0"; shift ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$PRESET" in
  fp8|exact) ;;
  *) echo "[start] Error: --preset must be 'fp8' or 'exact' (got: $PRESET)" >&2; exit 1 ;;
esac

if ! command -v "$COLAB_BIN" >/dev/null 2>&1 && [ ! -x "$COLAB_BIN" ]; then
  echo "[start] Error: colab CLI not found at '$COLAB_BIN'." >&2
  echo "[start] Install google-colab-cli (e.g. pipx install google-colab-cli && colab auth) or set COLAB_BIN=/path/to/colab." >&2
  exit 1
fi

LOCAL_IS_TTY=0
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  LOCAL_IS_TTY=1
fi
if [ "$LOCAL_IS_TTY" -eq 1 ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""
fi

if [ ! -f "$KEY" ]; then
  printf '%s[start]%s Generating ed25519 SSH key at %s ...\n' "$C_CYAN" "$C_RESET" "$KEY"
  mkdir -p "$(dirname "$KEY")"
  ssh-keygen -t ed25519 -N "" -C "colab-cli" -f "$KEY"
fi

if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ >/dev/null 2>&1 &
fi

printf '\n'
printf '%s╭──────────────────────────────────────────────────────────────────────╮%s\n' "$C_CYAN" "$C_RESET"
printf '%s│%s  %sFrankengine · Google Colab G4 (RTX PRO 6000 96 GB, sm_120a)%s        %s│%s\n' \
  "$C_CYAN" "$C_RESET" "$C_BOLD" "$C_RESET" "$C_CYAN" "$C_RESET"
printf '%s╰──────────────────────────────────────────────────────────────────────╯%s\n' "$C_CYAN" "$C_RESET"
printf '  %s%-12s%s %s%s%s\n' "$C_DIM" "Session" "$C_RESET" "$C_BOLD" "$SESSION" "$C_RESET"
printf '  %s%-12s%s %s%s%s (%s)\n' "$C_DIM" "Preset" "$C_RESET" "$C_CYAN" "$PRESET" "$C_RESET" "$BRANCH"
printf '  %s%-12s%s %s (batch: %sx%s, Vision=%s)\n' "$C_DIM" "Context" "$C_RESET" "$CTX" "$MAX_BATCH" "$BATCH_CTX" "$VISION"
printf '  %s%-12s%s %s\n' "$C_DIM" "Local Port" "$C_RESET" "$LOCAL_PORT"

sleep_with_countdown() {
  local secs="$1" attempt="$2"
  local frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' idx=0 rem="$secs"
  if [ "$LOCAL_IS_TTY" -eq 1 ]; then
    while [ "$rem" -gt 0 ]; do
      printf '\r  %s%s%s G4 busy (503). Next check (#%s) in %02ds... %s(Ctrl+C to stop)%s\033[K' \
        "$C_YELLOW" "${frames:$((idx % 10)):1}" "$C_RESET" "$attempt" "$rem" "$C_DIM" "$C_RESET"
      sleep 1; rem=$((rem - 1)); idx=$((idx + 1))
    done
    printf '\r\033[K'
  else
    printf '  G4 busy (503). Waiting %ss before retry #%s...\n' "$secs" "$attempt"
    sleep "$secs"
  fi
}

create_g4_session() {
  local attempt=1
  local base_delays=(30 45 60 90 120)
  local max_cap="${COLAB_WAIT_CAP:-120}"
  local max_attempts="${COLAB_MAX_ATTEMPTS:-0}"

  printf '\n  %s▸%s Creating Colab G4 session %s%s%s (~10.05 CU/hr)...\n' \
    "$C_YELLOW" "$C_RESET" "$C_BOLD" "$SESSION" "$C_RESET"

  while true; do
    local out rc=0
    # Never pass --high-mem for G4; G4 already provisions 176 GiB RAM / 48 vCPU / 96 GB VRAM.
    # Disable Rich/ANSI tracebacks so surrounding source context (e.g. `except TooManyAssignmentsError:`)
    # cannot trigger false hard-stop matches on transient 503 Service Unavailable errors.
    out="$(COLAB_CLI_ASSIGN_RETRIES=0 TYPER_STANDARD_TRACEBACK=1 NO_COLOR=1 FORCE_COLOR= PY_COLORS= "$COLAB_BIN" new -s "$SESSION" --gpu G4 2>&1)" || rc=$?

    if [ "$rc" -eq 0 ] && ! grep -qiE "Error:|Traceback|Service Unavailable|Allocation refused|Backend rejected" <<<"$out"; then
      printf '  %s✓%s Colab G4 session %s%s%s allocated (attempt #%s)\n' \
        "$C_GREEN" "$C_RESET" "$C_BOLD" "$SESSION" "$C_RESET" "$attempt"
      return 0
    fi

    local clean_err
    clean_err="$(printf '%s\n' "$out" | sed -E $'s/\x1b\\[[0-9;]*[[:alpha:]]//g' | sed -E '/^[[:space:]]*[╭│╰]/d; /^[[:space:]]*(File "|except )/d' | tr '\n' ' ' | tr -s ' ')"

    if grep -qiE "Allocation refused|precondition failed|Backend rejected|(^|[^A-Za-z_])TooManyAssignmentsError:|Too Many Requests|Unauthorized|Forbidden|not entitled|no quota|insufficient|invalid_grant|token.*expired|HTTP[^0-9]*(401|403|429)" <<<"$clean_err"; then
      printf '\n  %s✗%s Colab rejected allocation (hard stop, not retrying):\n%s\n' "$C_YELLOW" "$C_RESET" "$out" >&2
      return 1
    fi

    if [ "$WAIT_FOR_GPU" = "1" ] && grep -qiE "Service[[:space:]]*Unavailable|503|502|504|ReadTimeout|timed out|Temporary|capacity|Resource[[:space:]]*Exhausted" <<<"$clean_err"; then
      if [ "$max_attempts" -gt 0 ] && [ "$attempt" -ge "$max_attempts" ]; then
        printf '\n  %s✗%s Reached COLAB_MAX_ATTEMPTS=%s without available G4.\n' "$C_YELLOW" "$C_RESET" "$max_attempts" >&2
        return 1
      fi
      local idx=$((attempt - 1))
      [ "$idx" -ge "${#base_delays[@]}" ] && idx=$((${#base_delays[@]} - 1))
      local delay="${base_delays[$idx]}"
      [ "$delay" -ge 30 ] && delay=$((delay + (RANDOM % 9) - 4))
      [ "$delay" -gt "$max_cap" ] && delay="$max_cap"
      [ "$delay" -lt 1 ] && delay=1
      attempt=$((attempt + 1))
      sleep_with_countdown "$delay" "$attempt"
      continue
    fi

    printf '\n  %s✗%s Failed to create Colab G4 session:\n%s\n' "$C_YELLOW" "$C_RESET" "$out" >&2
    return 1
  done
}

session_rc=0
session_check="$("$COLAB_BIN" status -s "$SESSION" 2>&1)" || session_rc=$?
if [ "$session_rc" -ne 0 ] || grep -qiE "not found|no active sessions|does not exist" <<<"$session_check"; then
  create_g4_session
fi

printf '\n'
"$COLAB_BIN" status -s "$SESSION" 2>&1 | sed "s/^/  /" || true
"$COLAB_BIN" usage 2>&1 | sed "s/^/  /" || true

REMOTE_SCRIPT_FILE="$(mktemp)"
trap 'rm -f "$REMOTE_SCRIPT_FILE"' EXIT

cat > "$REMOTE_SCRIPT_FILE" <<'REMOTE_EOF'
set -euo pipefail
REPO_URL="$1"
BRANCH="$2"
PRESET="$3"
CTX="$4"
MAX_BATCH="$5"
BATCH_CTX="$6"
VISION="$7"
REMOTE_PORT="$8"
IS_TTY="${9:-0}"
[ -n "${NO_COLOR:-}" ] && IS_TTY=0

if [ "$IS_TTY" -eq 1 ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""
fi

fmt_duration() { printf '%02d:%02d' $(( ${1:-0} / 60 )) $(( ${1:-0} % 60 )); }
step_header() { printf '\n%s[%s/5]%s %s%s%s\n' "$C_CYAN$C_BOLD" "$1" "$C_RESET" "$C_BOLD" "$2" "$C_RESET"; }
step_ok() { printf '  %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
fail_with_log() {
  printf '  %s✗ %s%s\n' "$C_RED" "$1" "$C_RESET" >&2
  [ -f "$2" ] && tail -n "${3:-60}" "$2" >&2 || true
  exit 42
}

export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

step_header 1 "Checking GPU, CUDA 13+ toolchain & dependencies"
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)"
step_ok "GPU: ${GPU_LINE}"

t0=$SECONDS
if ! command -v ninja >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
  apt-get -qq update > /content/frank-deps.log 2>&1
  apt-get -qq install -y ninja-build cmake build-essential curl git aria2 >> /content/frank-deps.log 2>&1 \
    || fail_with_log "Failed to install build tools." /content/frank-deps.log
fi
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  pip -q install -U "huggingface_hub[cli]" hf_transfer >> /content/frank-deps.log 2>&1 \
    || fail_with_log "Failed to install huggingface_hub." /content/frank-deps.log
fi

find_cuda13_nvcc() {
  for cand in /usr/local/cuda-13*/bin/nvcc /usr/local/cuda/bin/nvcc "$(command -v nvcc 2>/dev/null || true)"; do
    [ -x "$cand" ] || continue
    local ver
    ver="$("$cand" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -n 1)"
    local maj="${ver%%.*}"
    if [ -n "$maj" ] && [ "$maj" -ge 13 ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

NVCC_BIN="$(find_cuda13_nvcc || true)"
if [ -z "$NVCC_BIN" ]; then
  printf '  %s▸%s System nvcc is < 13.0; installing CUDA 13.0 toolkit from NVIDIA repo...\n' "$C_YELLOW" "$C_RESET"
  . /etc/os-release
  UBU_VER="${VERSION_ID//./}"
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBU_VER}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb \
    >> /content/frank-cuda13.log 2>&1 || fail_with_log "Failed to fetch NVIDIA cuda-keyring." /content/frank-cuda13.log
  dpkg -i /tmp/cuda-keyring.deb >> /content/frank-cuda13.log 2>&1
  apt-get -qq update >> /content/frank-cuda13.log 2>&1
  CUDA_PKG="$(apt-cache pkgnames cuda-toolkit-13- | sort -V | tail -n 1)"
  [ -n "$CUDA_PKG" ] || CUDA_PKG="cuda-toolkit-13-0"
  apt-get -qq install -y "$CUDA_PKG" >> /content/frank-cuda13.log 2>&1 \
    || fail_with_log "Failed to install ${CUDA_PKG}." /content/frank-cuda13.log
  NVCC_BIN="$(find_cuda13_nvcc || true)"
  [ -n "$NVCC_BIN" ] || fail_with_log "CUDA 13 nvcc still not found after install." /content/frank-cuda13.log
fi
CUDA_ROOT="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
NVCC_VER="$("$NVCC_BIN" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)"
step_ok "CUDA ${NVCC_VER} ready (${NVCC_BIN}) in $(fmt_duration "$((SECONDS - t0))")"

step_header 2 "Cloning & building Frankengine TP=1 (sm_120a)"
t0=$SECONDS
if [ ! -d /content/Frankengine/.git ]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /content/Frankengine > /content/frank-build.log 2>&1 \
    || fail_with_log "Failed to clone Frankengine." /content/frank-build.log
else
  git -C /content/Frankengine fetch origin "$BRANCH" > /content/frank-build.log 2>&1
  git -C /content/Frankengine checkout -B "$BRANCH" "origin/$BRANCH" >> /content/frank-build.log 2>&1
fi
cd /content/Frankengine
GIT_SHA="$(git rev-parse --short HEAD)"

if [ ! -x /content/Frankengine/engine/native/build-120a/apps/serve/ingine-serve ] || \
   [ "$(cat /content/Frankengine/engine/native/build-120a/.built_sha 2>/dev/null || true)" != "$GIT_SHA" ]; then
  cmake -G Ninja -S engine/native -B engine/native/build-120a \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_CUDA_ARCHITECTURES=120a \
    -DINKLING_COMM_SOURCE_DIR="" \
    -DINGINE_BUILD_TESTS=OFF >> /content/frank-build.log 2>&1 \
    || fail_with_log "CMake configure failed." /content/frank-build.log 80
  cmake --build engine/native/build-120a -j "$(nproc)" \
    --target ingine-serve ingine-bench ingine-batch-bench ingine-validate >> /content/frank-build.log 2>&1 \
    || fail_with_log "Frankengine build failed." /content/frank-build.log 80
  printf '%s\n' "$GIT_SHA" > engine/native/build-120a/.built_sha
  step_ok "Built Frankengine @ ${GIT_SHA} (sm_120a) in $(fmt_duration "$((SECONDS - t0))")"
else
  step_ok "Cached binary @ ${GIT_SHA} (sm_120a)"
fi

step_header 3 "Downloading Inkling-2-Mini (~20.2 GB) & DFlash 2 (~945 MB)"
mkdir -p /content/models
t0=$SECONDS
DFLASH_SHA="217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee"
if [ ! -f /content/models/Inkling-2-Mini/model-00001-of-00001.safetensors ] || \
   [ ! -f /content/models/Inkling-2-Mini/mtp.safetensors ]; then
  env HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="thinkingmachines/Inkling-2-Mini",
    local_dir="/content/models/Inkling-2-Mini",
    max_workers=8,
)
PY
fi

if ! echo "${DFLASH_SHA}  /content/models/Inkling-2-Mini-DFlash2/model.safetensors" | sha256sum -c --status 2>/dev/null; then
  env HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="incoai/Inkling-2-Mini-DFlash2",
    revision="4f9088ea66c21f2b99ed657d1941c306b03e84d2",
    local_dir="/content/models/Inkling-2-Mini-DFlash2",
    max_workers=8,
)
PY
  echo "${DFLASH_SHA}  /content/models/Inkling-2-Mini-DFlash2/model.safetensors" | sha256sum -c -
fi
step_ok "Models verified in $(fmt_duration "$((SECONDS - t0))")"

step_header 4 "Launching ingine-serve (${PRESET} preset, Vision=${VISION})"
pkill -x ingine-serve 2>/dev/null || true
sleep 1

if [ "$PRESET" = "exact" ]; then
  DENSE_FP8=0; KV_FP8=0; HEAD_NVFP4=0; DENSE_PICO12=1; CERT_HEAD=1
else
  DENSE_FP8=1; KV_FP8=1; HEAD_NVFP4=1; DENSE_PICO12=0; CERT_HEAD=0
fi

cat > /content/frank-serve-env.sh <<EOF
export BUILD_DIR="/content/Frankengine/engine/native/build-120a"
export HEAD_MODEL_DIR="/content/models/Inkling-2-Mini"
export DFLASH2_DIR="/content/models/Inkling-2-Mini-DFlash2"
export HOST="127.0.0.1"
export PORT="${REMOTE_PORT}"
export SERVED_NAME="inkling-2-mini"
export MAX_CTX="${CTX}"
export MAX_BATCH="${MAX_BATCH}"
export BATCH_CTX="${BATCH_CTX}"
export BATCH_SPEC=7
export BATCH_SPEC_THRESHOLD=0.30
export BATCH_SPEC_MIN=2
export MAX_QUEUE=64
export PARK_MB=2048
export PREFILL_CHUNK=8192
export VISION="${VISION}"
export DENSE_FP8="${DENSE_FP8}"
export KV_FP8="${KV_FP8}"
export HEAD_NVFP4="${HEAD_NVFP4}"
export DRAFT_NVFP4=1
export DENSE_PICO12="${DENSE_PICO12}"
export CERT_HEAD="${CERT_HEAD}"
export PRESET_NAME="${PRESET}"
EOF

# shellcheck disable=SC1091
source /content/frank-serve-env.sh
t0=$SECONDS
nohup /content/Frankengine/scripts/serve_tp1.sh > /content/frank-server.log 2>&1 < /dev/null &
SRV_PID=$!
echo "$SRV_PID" > /content/frank-server.pid

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    fail_with_log "ingine-serve exited during startup." /content/frank-server.log 60
  fi
  if curl -sf "http://127.0.0.1:${REMOTE_PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || fail_with_log "ingine-serve did not become ready within 180s." /content/frank-server.log 60
STARTUP_S=$((SECONDS - t0))
echo "STARTUP_S=${STARTUP_S}" > /content/frank-startup.env
VRAM_USED="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -n 1)"
step_ok "ingine-serve ready in ${STARTUP_S}s (VRAM: ${VRAM_USED})"

step_header 5 "Quick 1-shot sanity check"
curl -sf "http://127.0.0.1:${REMOTE_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ✓ Sanity reply:", repr(d["choices"][0]["message"].get("content","").strip()))'
REMOTE_EOF

run_remote_provision() {
  local max_tries=4 try=1
  while [ "$try" -le "$max_tries" ]; do
    local rc=0
    ssh -i "$KEY" \
      -o "ProxyCommand=${COLAB_BIN} ssh --proxy-mode -s ${SESSION} -i ${KEY}" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=20 \
      -o ServerAliveCountMax=6 \
      -o ConnectTimeout=25 \
      root@colab \
      "bash -s -- $(printf '%q ' "$REPO_URL" "$BRANCH" "$PRESET" "$CTX" "$MAX_BATCH" "$BATCH_CTX" "$VISION" "$REMOTE_PORT" "$LOCAL_IS_TTY")" \
      < "$REMOTE_SCRIPT_FILE" || rc=$?

    [ "$rc" -eq 0 ] && return 0
    if [ "$rc" -eq 42 ] || [ "$rc" -eq 130 ]; then return "$rc"; fi
    if [ "$try" -lt "$max_tries" ]; then
      printf '\n  %s▸%s SSH interrupted (rc=%s). Retrying (%s/%s)...\n' "$C_YELLOW" "$C_RESET" "$rc" "$try" "$max_tries" >&2
      sleep $((try * 5))
    fi
    try=$((try + 1))
  done
  return 1
}

run_remote_provision
rm -f "$REMOTE_SCRIPT_FILE"

METAFILE="/tmp/colab-tunnel-${SESSION}-${LOCAL_PORT}.env"
cat > "$METAFILE" <<EOF
MODEL_PRESET="${PRESET}"
MODEL_CTX="${CTX}"
MODEL_NAME="inkling-2-mini"
REMOTE_PORT="${REMOTE_PORT}"
EOF

if [ "$OPEN_TUNNEL" = "0" ]; then
  printf '\n%s✓%s Colab G4 server running on remote 127.0.0.1:%s (session=%s, preset=%s)\n' \
    "$C_GREEN" "$C_RESET" "$REMOTE_PORT" "$SESSION" "$PRESET"
  printf '  Run benchmark:  %s/bench.sh --session %s --suite spark-bench --preset %s\n' "$SCRIPT_DIR" "$SESSION" "$PRESET"
  printf '  Open tunnel:    %s/tunnel.sh %s %s\n' "$SCRIPT_DIR" "$SESSION" "$LOCAL_PORT"
  printf '  Stop & bill 0:  %s/stop.sh %s %s\n\n' "$SCRIPT_DIR" "$SESSION" "$LOCAL_PORT"
  exit 0
fi

exec "${SCRIPT_DIR}/tunnel.sh" "$SESSION" "$LOCAL_PORT" "$TUNNEL_MODE"
