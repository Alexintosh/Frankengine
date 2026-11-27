#!/usr/bin/env bash
# Launch single-GPU (TP=1) Frankengine OpenAI-compatible HTTP server (256K context + hybrid DFlash2/DSpark).
# Configure HEAD_MODEL_DIR, DFLASH2_DIR, MAX_CTX, PORT in scripts/local.sh or via environment variables.
# AUTO_VRAM=1 selects MAX_BATCH and BATCH_CTX for the Inkling-2-Mini TP=1 memory budget.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
auto_vram_override=${AUTO_VRAM-}
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"

auto_size_vram() {
  # Validate the inputs to the budget before replacing the two sizing defaults.
  MAX_BATCH=1
  BATCH_CTX=1024
  validate_config

  local mib=$((1024 * 1024)) gpu=${CUDA_VISIBLE_DEVICES:-0} memory_mib
  local memory_source=nvidia-smi
  if ! memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu%%,*}"); then
    fail E_VRAM "AUTO_VRAM could not query GPU memory"
  fi
  memory_mib=${memory_mib%%$'\n'*}
  memory_mib=${memory_mib//[[:space:]]/}
  case "$memory_mib" in
    '[N/A]'|N/A|'[NotSupported]'|0)
      memory_source=/proc/meminfo
      memory_mib=$(awk '$1 == "MemTotal:" { print int($2 / 1024); exit }' /proc/meminfo) ;;
  esac
  [[ "$memory_mib" =~ ^[1-9][0-9]*$ ]] || fail E_VRAM "AUTO_VRAM could not read memory capacity from $memory_source"

  # Weight and fixed workspace estimate for Inkling-2-Mini + DFlash2, and spare capacity for prefill.
  local base_mib=$((20 * 1024 + 512)) headroom_mib=2048
  local available=$(((memory_mib - base_mib - headroom_mib) * mib))
  # Eight full-attention visits, each with a K and V row of 8 heads x 128 elements.
  local kv_format=BF16 kv_bytes=$((8 * 2 * 8 * 128 * 2))
  if [[ "$KV_FP8" == 1 ]]; then
    kv_format=FP8
    kv_bytes=$((8 * 2 * 8 * 128))
  elif [[ "$KV_Q4" == 1 ]]; then
    kv_format=Q4
    kv_bytes=$((8 * 2 * 8 * (128 / 2 + 2 * 2)))
  fi
  # Forty sliding visits use BF16 K/V and int64 positions. Convolution state uses FP32.
  local cache_fixed=$((40 * 1024 * (2 * 1024 * 2 + 8) + 48 * (2 * 1024 + 2 * 2304) * 4))
  local solo=$((cache_fixed + MAX_CTX * kv_bytes))
  if [[ "$SPEC" == 1 && "$DRAFT_TYPE" != dflash2 ]]; then
    local draft_ctx=$((MAX_CTX < 65536 ? MAX_CTX : 65536))
    solo=$((solo + draft_ctx * 2048))
  fi

  # Stay within the Stage 0 profile through 128 GiB and enable 64 slots on B200-class cards.
  local capacity=32 context_cap=32768 batch_fixed batch_kv demand
  if (( memory_mib < 48 * 1024 )); then
    capacity=8
    context_cap=16384
  elif (( memory_mib >= 160 * 1024 )); then
    capacity=64
  fi
  context_cap=$((MAX_CTX < context_cap ? MAX_CTX : context_cap))
  if (( BATCH_SPEC == 0 )); then
    # Continuous batching holds a full MAX_CTX target cache for every slot.
    local extra_slot=$((cache_fixed + MAX_CTX * kv_bytes))
    (( available >= solo )) || fail E_VRAM "AUTO_VRAM cannot fit MAX_CTX=$MAX_CTX with the reserved headroom"
    MAX_BATCH=$((1 + (available - solo) / extra_slot))
    if (( MAX_BATCH > 8 )); then MAX_BATCH=8; fi
    BATCH_CTX=$context_cap
    demand=$((solo + (MAX_BATCH - 1) * extra_slot))
  else
    # Allow per-slot verifier/drafter scratch in addition to known cache allocations.
    batch_fixed=$((cache_fixed + 48 * mib))
    batch_kv=$kv_bytes
    local batch_draft=${INGINE_BATCH_DRAFT_TYPE:-dspark}
    if [[ -z "${INGINE_BATCH_DRAFT_TYPE:-}" && "$SPEC" == 1 && "$DRAFT_TYPE" != dspark && "$BATCH_SPEC" -ge 5 ]]; then
      batch_draft=$DRAFT_TYPE
    fi
    if [[ "$batch_draft" != dflash2 ]]; then
      # Two 1024-row BF16 local rings and one FP8 global ring.
      batch_fixed=$((batch_fixed + 2 * 1024 * (2 * 1024 * 2 + 4)))
      batch_kv=$((batch_kv + 2 * 1024))
    fi
    if [[ "$batch_draft" != dspark ]]; then
      # Five 2048-row DFlash2 K/V rings per slot.
      local draft_kv_mib=20
      if [[ "${INGINE_BATCH_DFLASH2_KV_BF16:-0}" != 0 ]]; then draft_kv_mib=40; fi
      batch_fixed=$((batch_fixed + draft_kv_mib * mib))
    fi
    (( available >= solo + batch_fixed + 1024 * batch_kv )) ||
      fail E_VRAM "AUTO_VRAM cannot fit MAX_CTX=$MAX_CTX and one batch slot with the reserved headroom"

    MAX_BATCH=$capacity
    while (( available - solo < MAX_BATCH * (batch_fixed + 1024 * batch_kv) )); do
      MAX_BATCH=$((MAX_BATCH / 2))
    done
    BATCH_CTX=$context_cap
    while (( available - solo < MAX_BATCH * (batch_fixed + BATCH_CTX * batch_kv) )); do
      BATCH_CTX=$((BATCH_CTX / 2 > 1024 ? BATCH_CTX / 2 : 1024))
    done
    demand=$((solo + MAX_BATCH * (batch_fixed + BATCH_CTX * batch_kv)))
  fi

  printf '[AUTO_VRAM] %s capacity %d MiB, model/workspace estimate %d MiB, headroom %d MiB, cache/slot budget %d MiB\n' \
    "$memory_source" "$memory_mib" "$base_mib" "$headroom_mib" "$((available / mib))" >&2
  printf '[AUTO_VRAM] %s target KV %d B/token, estimated cache/slot use %d MiB, MAX_CTX=%d MAX_BATCH=%d BATCH_CTX=%d\n' \
    "$kv_format" "$kv_bytes" "$(((demand + mib - 1) / mib))" "$MAX_CTX" "$MAX_BATCH" "$BATCH_CTX" >&2
}

case "${auto_vram_override:-${AUTO_VRAM:-0}}" in
  0) validate_config ;;
  1) auto_size_vram ;;
  *) fail E_CONFIG "AUTO_VRAM must be 0 or 1" ;;
esac

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
  --max-batch "$MAX_BATCH"
  --batch-ctx "$BATCH_CTX"
  --batch-spec "$BATCH_SPEC"
  --batch-spec-threshold "$BATCH_SPEC_THRESHOLD"
  --batch-spec-min "$BATCH_SPEC_MIN"
  --max-queue "$MAX_QUEUE"
  --park-mb "$PARK_MB"
  --prefill-chunk "$PREFILL_CHUNK"
  --spec-widths "$SPEC_WIDTHS"
  --spec-threshold "$SPEC_THRESHOLD"
)
if [[ "$VISION" == "1" ]]; then
  ARGS+=(
    --image-max-tokens "$IMAGE_MAX_TOKENS"
    --max-images "$MAX_IMAGES"
    --max-image-tokens "$MAX_IMAGE_TOKENS"
  )
else
  ARGS+=(--no-vision)
fi
[[ "$DENSE_FP8" == "1" ]] && ARGS+=(--dense-fp8)
[[ "$DENSE_PICO12" == "1" ]] && ARGS+=(--dense-pico12)
[[ "$CERT_HEAD" == "1" ]] && ARGS+=(--cert-head)
[[ "$KV_FP8" == "1" ]] && ARGS+=(--kv-fp8)
[[ "$KV_Q4" == "1" ]] && ARGS+=(--kv-q4)
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

exec env \
  "INGINE_BATCH_MIN_CONCURRENCY=$BATCH_MIN_CONCURRENCY" \
  "INGINE_BATCH_PREFILL_CHUNK_WHILE_DECODING=$BATCH_PREFILL_CHUNK_WHILE_DECODING" \
  "INGINE_BATCH_DECODE_STEPS_PER_PREFILL=$BATCH_DECODE_STEPS_PER_PREFILL" \
  "INGINE_BATCH_SPEC_ADAPTIVE=$BATCH_SPEC_ADAPTIVE" \
  "INGINE_BATCH_DFLASH2_MAX_CTX=$BATCH_DFLASH2_MAX_CTX" \
  "INGINE_DFLASH2_SAMPLED=$DFLASH2_SAMPLED" \
  "$EXE" "${ARGS[@]}"
