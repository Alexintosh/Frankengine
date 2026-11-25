#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project=$HOME/code/inkling-2-mini-engine
out=${TP2_LOG_DIR:?Set TP2_LOG_DIR}
unified=${TP2_UNIFIED_BUILD:?Set TP2_UNIFIED_BUILD}
control=${TP2_CONTROL_BUILD:?Set TP2_CONTROL_BUILD}
mkdir -p "$out"
for ctx in 2048 32768 131072; do
  items=3
  policy=(--draft-nvfp4 --spec-threshold 0.65)
  if ((ctx == 32768)); then policy=(--spec-threshold 0.75); fi
  if ((ctx == 131072)); then items=2; policy=(--draft-nvfp4 --spec-threshold 0.80); fi
  corpus=$project/bench/agentic-$ctx.jsonl
  for tag in ctl1 unified ctl2; do
    export TP2_BUILD_DIR=$control
    route=()
    if [[ $tag == unified ]]; then
      export TP2_BUILD_DIR=$unified
      route=(--draft-type hybrid --dflash2-dir "$project/models/Inkling-2-Mini-DFlash2")
    fi
    ./run.sh ingine-bench --dense-fp8 --head-nvfp4 --kv-fp8 --spec --spec-widths fine2 \
      --temperature 0 --max-ctx "$((ctx+264))" --cold-prefill --forced "$corpus" \
      --forced-items "$items" --forced-free 0 --forced-trace "$out/$ctx-$tag.jsonl" \
      "${policy[@]}" "${route[@]}" >"$out/$ctx-$tag.log" 2>&1
  done
done
