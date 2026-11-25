#!/usr/bin/env bash
# Run from rank0. Use one dedicated local port per qualification.
set -euo pipefail

project="$HOME/code/inkling-2-mini-engine"
build=${TP2_BUILD_DIR:-"$project/ingine-tp/native/build-gb10-tp"}
worker=${TP2_WORKER:-worker-node}
head_model=${TP2_HEAD_MODEL_DIR:-"$project/models/Inkling-2-Mini"}
worker_model=${TP2_WORKER_MODEL_DIR:-/path/to/Inkling-2-Mini}
remote=/tmp/inkling_tp_unified
log_root=${TP2_LOG_DIR:-"$build/tp2-runs"}
mkdir -p "$log_root"
logs=$(mktemp -d "$log_root/run.XXXXXXXX")
run_id=${logs##*/}
remote_run=$remote/$run_id

app=ingine-serve
exe=$build/apps/serve/$app
defaults=(--engine --dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4
          --spec-widths fine2 --spec-threshold 0.65 --host 0.0.0.0 --port 8000
          --no-vision --park-mb 1024)
if [[ ${TP2_SPEC:-1} == 1 ]]; then defaults+=(--spec); fi
local_args=("${defaults[@]}" "$@")
worker_pid=
head_pid=

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n $head_pid ]] && kill -0 "$head_pid" 2>/dev/null; then
    kill -TERM "$head_pid"
    wait "$head_pid" || true
  fi
  if [[ -n $worker_pid ]]; then wait "$worker_pid" || true; fi
  rm -f -- "$logs/to_worker" "$logs/from_worker"
  local pids0 pids1
  pids0=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || status=1
  pids1=$(ssh -o BatchMode=yes "$worker" flock -w 1800 /tmp/frankengine_gpu.lock nvidia-smi --query-compute-apps=pid --format=csv,noheader) || status=1
  printf 'Compute PIDs after run: rank0=[%s] worker=[%s]\n' "$pids0" "$pids1"
  [[ -z $pids0 && -z $pids1 ]] || status=1
  ssh -o BatchMode=yes "$worker" "rm -rf -- '$remote_run'" || status=1
  if (( status != 0 )); then
    echo "TP=2 run exited $status. Rank logs are in $logs" >&2
    if [[ -s $logs/rank1.log ]]; then tail -n 12 "$logs/rank1.log" >&2; fi
  fi
  exit "$status"
}
trap 'exit 130' INT
trap 'exit 143' TERM HUP

ssh -o BatchMode=yes "$worker" "mkdir -p '$remote_run'"
rsync -az "$exe" "$worker:$remote_run/$app"
local_sha=$(sha256sum "$exe")
remote_sha=$(ssh -o BatchMode=yes "$worker" sha256sum "$remote_run/$app")
[[ ${local_sha%% *} == "${remote_sha%% *}" ]]
printf 'SHA-256 %s\n' "${local_sha%% *}"
exec {gpu_lock}>/tmp/frankengine_gpu.lock
flock -w 1800 "$gpu_lock"
trap cleanup EXIT
for arg in "$@"; do
  case $arg in
    --tp-size|--tp-rank|--model-dir|--ablation|--tp-size=*|--tp-rank=*|--model-dir=*|--ablation=*)
      echo "$arg is managed by the launcher" >&2
      exit 2
      ;;
  esac
done

# Use the qualified dual-rail defaults in M1. Allow a caller to use a different
# rendezvous port, proxy core, or timeout on both ranks.
net=(INKLING_COMM_WORLD_SIZE=2 INKLING_COMM_BACKEND=roce_uma INKLING_COMM_RAILS=2
     "INKLING_COMM_PORT=${INKLING_COMM_PORT:-29620}"
     "INKLING_COMM_PROXY_CPU=${INKLING_COMM_PROXY_CPU:-14}"
     "INKLING_COMM_BOOTSTRAP_TIMEOUT_MS=${INKLING_COMM_BOOTSTRAP_TIMEOUT_MS:-60000}"
     "INKLING_COMM_TIMEOUT_MS=${INKLING_COMM_TIMEOUT_MS:-5000}"
     "INKLING_TP_WIRE=${INKLING_TP_WIRE:-fp32}")
# Use the authenticated SSH session for framed commands when the ranks are
# launched together. RoCE and its rendezvous still use the dedicated interfaces.
printf -v remote_cmd '%q ' flock -w 1800 /tmp/frankengine_gpu.lock bash -c 'set -e; echo "$$" > "$1/pid"; shift; exec "$@"' bash "$remote_run" \
  env "${net[@]}" INKLING_SERVE_CTRL_STDIO=1 INKLING_COMM_RANK=1 INKLING_TP_SIZE=2 INKLING_TP_RANK=1 \
  "$remote_run/$app" --model-dir "$worker_model" "${local_args[@]}"

echo "TP=2 rank logs $logs"
to_worker=$logs/to_worker
from_worker=$logs/from_worker
mkfifo -m 600 "$to_worker" "$from_worker"
# Keep both FIFOs open while the child redirections are established.
exec 5<>"$to_worker" 6<>"$from_worker"
ssh -o BatchMode=yes "$worker" "$remote_cmd" <"$to_worker" >"$from_worker" \
  5>&- 6>&- 2>"$logs/rank1.log" &
worker_pid=$!

env "${net[@]}" INKLING_SERVE_CTRL_STDIO=1 INKLING_COMM_RANK=0 INKLING_TP_SIZE=2 INKLING_TP_RANK=0 \
  "$exe" --model-dir "$head_model" "${local_args[@]}" 3<"$from_worker" 4>"$to_worker" \
  5>&- 6>&- >"$logs/rank0.log" 2>&1 &
head_pid=$!
printf "%s\n" "$head_pid" > "$logs/rank0.pid"
exec 5>&- 6>&-

status=0
wait -n "$head_pid" "$worker_pid" || status=$?
exit "$status"
