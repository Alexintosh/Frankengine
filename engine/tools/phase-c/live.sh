#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
tag=$1
port=${TP2_TEST_PORT:-18091}
out=${TP2_LOG_DIR:?Set TP2_LOG_DIR}
mkdir -p "$out"
args=(--host 127.0.0.1 --port "$port" --max-ctx 131072 --max-model-len 131072)
if [[ $tag == unified ]]; then
  args+=(--draft-type hybrid --dflash2-adaptive --dflash2-dir "${DFLASH2_DIR:?Set DFLASH2_DIR}")
fi
url=http://127.0.0.1:$port/v1
if curl -s --max-time 1 "$url/models" >/dev/null; then
  echo "Port $port is already serving HTTP" >&2
  exit 1
fi
./serve.sh "${args[@]}" >"$out/$tag-launch.log" 2>&1 &
launcher=$!
cleanup() {
  kill -TERM "$launcher" 2>/dev/null || true
  wait "$launcher" || true
}
trap cleanup EXIT
until curl -fs "$url/models" >/dev/null; do
  kill -0 "$launcher" 2>/dev/null || exit 1
  sleep 1
done
python3 check_sse.py "$url" "$tag" "$out/$tag.jsonl" "${PHASE_C_NEEDLES:-}"
