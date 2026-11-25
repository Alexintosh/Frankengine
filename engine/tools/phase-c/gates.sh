#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build=${TP2_BUILD_DIR:?Set TP2_BUILD_DIR to the unified build}
project=$HOME/code/inkling-2-mini-engine
worker=${TP2_WORKER:-worker-node}
ctest_args=(--test-dir "$build" --output-on-failure
            -R '^(test_tp2_kernels|spec_accept|serve_scheduler|serve_server)$')
flock -w 1800 /tmp/frankengine_gpu.lock ctest "${ctest_args[@]}"
for host in local "$worker"; do
  if [[ $host == local ]]; then
    pids=$(flock -w 1800 /tmp/frankengine_gpu.lock nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  else
    pids=$(ssh -n "$host" flock -w 1800 /tmp/frankengine_gpu.lock nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  fi
  printf '%s compute PIDs after CTest [%s]\n' "$host" "$pids"
  [[ -z $pids ]]
done
for visibility in direct cache_bypass; do
  for test in smoke gather-bench; do
    INKLING_COMM_VISIBILITY=$visibility ./run.sh test_comm_backend "$test"
  done
done
common=(--dense-fp8 --head-nvfp4)
./run.sh ingine-validate "${common[@]}" --cases orig_kv16 --min-top1 0.98484848 --max-mean-kl 0.00192697476040
for draft in dspark dflash2 hybrid; do
  ./run.sh ingine-validate "${common[@]}" --cases orig_kv16_chat --generate --spec --spec-widths fine2 \
    --draft-type "$draft" --dflash2-dir "$project/models/Inkling-2-Mini-DFlash2"
done
