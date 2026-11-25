#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project=$HOME/code/inkling-2-mini-engine
out=${TP2_LOG_DIR:?Set TP2_LOG_DIR}
unified=${TP2_UNIFIED_BUILD:?Set TP2_UNIFIED_BUILD}
control=${TP2_CONTROL_BUILD:?Set TP2_CONTROL_BUILD}
mkdir -p "$out"
common=(--dense-fp8 --kv-fp8 --head-nvfp4 --draft-nvfp4 --spec --spec-widths fine2
        --temperature 0 --tokens 256 --runs 6 --cold-prefill --spec-threshold 0.65)
for tag in ctl1 unified ctl2; do
  export TP2_BUILD_DIR=$control
  route=()
  if [[ $tag == unified ]]; then
    export TP2_BUILD_DIR=$unified
    route=(--draft-type hybrid --dflash2-dir "$project/models/Inkling-2-Mini-DFlash2")
  fi
  ./run.sh ingine-bench "${common[@]}" "${route[@]}" >"$out/$tag.log" 2>&1
done
