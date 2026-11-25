#!/usr/bin/env bash
# Launch single-GPU (TP=1) Frankengine OpenAI-compatible HTTP server (256K context + hybrid DFlash2/DSpark).
# Configure HEAD_MODEL_DIR, DFLASH2_DIR, MAX_CTX, PORT in scripts/local.sh or via environment variables.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"
validate_config

[[ -x "$EXE" ]] || fail E_BIN "Missing ingine-serve binary at $EXE; build engine/ first"

ARGS=(
  --engine
  --tp-size 1
  --host "$HOST"
  --port "$PORT"
  --model-id "$SERVED_NAME"
  --model-dir "$HEAD_MODEL_DIR"
  --max-ctx "$MAX_CTX"
  --max-model-len "$MAX_CTX"
  --max-queue "$MAX_QUEUE"
  --park-mb "$PARK_MB"
  --no-vision
  --spec-widths "$SPEC_WIDTHS"
  --spec-threshold "$SPEC_THRESHOLD"
)
[[ "$DENSE_FP8" == "1" ]] && ARGS+=(--dense-fp8)
[[ "$KV_FP8" == "1" ]] && ARGS+=(--kv-fp8)
[[ "$HEAD_NVFP4" == "1" ]] && ARGS+=(--head-nvfp4)
[[ "$DRAFT_NVFP4" == "1" ]] && ARGS+=(--draft-nvfp4)
[[ "$SPEC" == "1" ]] && ARGS+=(--spec)
if [[ "$SPEC" == "1" && "$DRAFT_TYPE" != "dspark" ]]; then
  ARGS+=(
    --draft-type "$DRAFT_TYPE"
    --dflash2-dir "$DFLASH2_DIR"
    --dflash2-threshold "$DFLASH2_THRESHOLD"
    --dflash2-max-ctx "$DFLASH2_MAX_CTX"
  )
  [[ "$DFLASH2_ADAPTIVE" == "1" ]] && ARGS+=(--dflash2-adaptive)
fi
if [[ -n "$SERVE_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($SERVE_EXTRA_ARGS)
fi
ARGS+=("$@")

exec "$EXE" "${ARGS[@]}"
