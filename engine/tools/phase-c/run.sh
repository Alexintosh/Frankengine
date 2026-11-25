#!/usr/bin/env bash
# Run from rank0. Both processes consume the same CLI request schedule.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
project="$HOME/code/inkling-2-mini-engine"
build=${TP2_BUILD_DIR:-"$repo/native/build-phase-c"}
worker=${TP2_WORKER:-worker-node}
head_model=${TP2_HEAD_MODEL_DIR:-"$project/models/Inkling-2-Mini"}
worker_model=${TP2_WORKER_MODEL_DIR:-/path/to/Inkling-2-Mini}
remote=/tmp/inkling_tp_unified
log_root=${TP2_LOG_DIR:-"$build/tp2-runs"}
mkdir -p "$log_root"
logs=$(mktemp -d "$log_root/run.XXXXXXXX")
run_id=${logs##*/}
remote_run=$remote/$run_id
remote_exe=$remote_run/app

app=${1:-}
head_args=(--model-dir "$head_model")
worker_args=(--model-dir "$worker_model")
case $app in
  ingine-bench) bin_dir=bench ;;
  ingine-validate) bin_dir=validate ;;
  test_comm_backend) bin_dir=.; head_args=(); worker_args=() ;;
  *) echo "Usage: $0 ingine-bench|ingine-validate|test_comm_backend [options]" >&2; exit 2 ;;
esac
shift
exe=$build/apps/$bin_dir/$app
[[ $app != test_comm_backend ]] || exe=$build/tp_comm/$app
remote_args=()
local_args=("$@")
worker_pid=
head_pid=

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n $head_pid ]]; then
    if [[ -s $logs/rank0.pid ]]; then
      kill "$(cat "$logs/rank0.pid")" 2>/dev/null || true
    else
      kill "$head_pid" 2>/dev/null || true
    fi
  fi
  if [[ -n $worker_pid ]] && kill -0 "$worker_pid" 2>/dev/null; then
    # Kill only the application belonging to this invocation.
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$worker" \
      "if test -f '$remote_run/pid'; then kill \$(cat '$remote_run/pid') 2>/dev/null || true; fi" || true
    kill "$worker_pid" 2>/dev/null || true
  fi
  if [[ -n $head_pid ]]; then wait "$head_pid" 2>/dev/null || true; fi
  if [[ -n $worker_pid ]]; then wait "$worker_pid" 2>/dev/null || true; fi
  if (( status != 0 )); then
    echo "TP=2 run exited $status. Rank logs are in $logs" >&2
    if [[ -s $logs/rank1.log ]]; then tail -n 12 "$logs/rank1.log" >&2; fi
  fi
  local local_pids peer_pids
  local_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || status=1
  peer_pids=$(ssh -o BatchMode=yes "$worker" flock -w 1800 /tmp/frankengine_gpu.lock nvidia-smi --query-compute-apps=pid --format=csv,noheader) || status=1
  printf 'Compute PIDs after run: rank0=[%s] rank1=[%s]\n' "$local_pids" "$peer_pids"
  [[ -z $local_pids && -z $peer_pids ]] || status=1
  flock -u "$gpu_lock"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$worker" "rm -rf -- '$remote_run'" >/dev/null 2>&1 || true
  exit "$status"
}

trap 'exit 130' INT
trap 'exit 143' TERM HUP

ssh -o BatchMode=yes "$worker" "mkdir -p '$remote_run'"
rsync -az "$exe" "$worker:$remote_exe"
local_sha=$(sha256sum "$exe")
remote_sha=$(ssh -o BatchMode=yes "$worker" sha256sum "$remote_exe")
[[ ${local_sha%% *} == "${remote_sha%% *}" ]]
have_golden=false
while (($#)); do
  case $1 in
    --forced|--golden|--ablation|--greedy-witness)
      flag=$1
      source_path=${2:?missing path argument}
      dest=$remote_run/${flag#--}
      if [[ -d $source_path ]]; then
        rsync -az -- "$source_path/" "$worker:$dest/"
      else
        rsync -az -- "$source_path" "$worker:$dest"
      fi
      remote_args+=("$flag" "$dest")
      if [[ $flag == --golden ]]; then have_golden=true; fi
      shift 2
      ;;
    --forced-trace|--generate-trace)
      # Trace paths belong to rank 0. Rank 1 follows the same decode schedule without writing a trace.
      : "${2:?missing trace path argument}"
      shift 2
      ;;
    --tp-size|--tp-rank|--model-dir)
      echo "$1 is set by the launcher" >&2
      exit 2
      ;;
    *)
      remote_args+=("$1")
      shift
      ;;
  esac
done

if [[ $app == ingine-validate && $have_golden == false ]]; then
  golden=$repo/native/tests/golden/model
  rsync -az -- "$golden/" "$worker:$remote_run/golden/"
  local_args+=(--golden "$golden")
  remote_args+=(--golden "$remote_run/golden")
fi

# Use the qualified dual-rail defaults in M1. Allow a caller to use a different
# rendezvous port, proxy core, or timeout on both ranks.
net=(INKLING_COMM_WORLD_SIZE=2 INKLING_COMM_BACKEND=roce_uma INKLING_COMM_RAILS=2 "INKLING_COMM_VISIBILITY=${INKLING_COMM_VISIBILITY:-direct}"
     "INKLING_COMM_PORT=${INKLING_COMM_PORT:-29620}"
     "INKLING_COMM_PROXY_CPU=${INKLING_COMM_PROXY_CPU:-14}"
     "INKLING_COMM_BOOTSTRAP_TIMEOUT_MS=${INKLING_COMM_BOOTSTRAP_TIMEOUT_MS:-60000}"
     "INKLING_COMM_TIMEOUT_MS=${INKLING_COMM_TIMEOUT_MS:-5000}"
     "INKLING_TP_WIRE=${INKLING_TP_WIRE:-fp32}")
printf -v remote_cmd '%q ' flock -w 1800 /tmp/frankengine_gpu.lock bash -s -- "$remote_run" env "${net[@]}" INKLING_COMM_RANK=1 \
  INKLING_TP_SIZE=2 INKLING_TP_RANK=1 stdbuf -oL -eL "$remote_exe" \
  "${worker_args[@]}" "${remote_args[@]}"

exec {gpu_lock}>/tmp/frankengine_gpu.lock
flock -w 1800 "$gpu_lock"
trap cleanup EXIT
echo "TP=2 rank logs $logs (SHA-256 ${local_sha%% *})"
ssh -o BatchMode=yes "$worker" "$remote_cmd" >"$logs/rank1.log" 2>&1 <<'REMOTE' &
set -euo pipefail
run=$1
shift
child=
trap 'if [[ -n $child ]]; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; rm -f "$run/pid"' EXIT
trap 'exit 143' HUP INT TERM
"$@" </dev/null &
child=$!
printf '%s\n' "$child" > "$run/pid"
printf 'rank1 compute PID %s\n' "$child"
status=0
wait "$child" || status=$?
child=
exit "$status"
REMOTE
worker_pid=$!

rank0=()
if [[ ${PHASE_C_NSYS:-0} == 1 ]]; then
  rank0=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node
         --output="$logs/rank0")
fi
env "${net[@]}" INKLING_COMM_RANK=0 INKLING_TP_SIZE=2 INKLING_TP_RANK=0 stdbuf -oL -eL \
  "${rank0[@]}" bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' bash "$logs/rank0.pid" \
  "$exe" "${head_args[@]}" "${local_args[@]}" </dev/null >"$logs/rank0.log" 2>&1 &
head_pid=$!

done_pid=
status=0
wait -n -p done_pid "$head_pid" "$worker_pid" || status=$?
if [[ $done_pid == "$head_pid" ]]; then
  head_pid=
  cat "$logs/rank0.log"
  if (( status == 0 )); then wait "$worker_pid" || status=$?; worker_pid=; fi
else
  worker_pid=
  if (( status == 0 )); then wait "$head_pid" || status=$?; head_pid=; fi
  cat "$logs/rank0.log"
fi
exit "$status"
