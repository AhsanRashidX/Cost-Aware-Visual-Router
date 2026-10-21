#!/usr/bin/env bash
# Download the model into $HF_CACHE once; print the snapshot path, the commit hash to pin as MODEL_REVISION in
# config.sh, and the size on disk. Docker modes download inside the vLLM image (the host needs no Python).
# Usage:  bash download_model.sh
set -eo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "$HF_CACHE"
PY='import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], revision=sys.argv[2])
print("snapshot:", p)
print("MODEL_REVISION=" + p.rstrip("/").split("/")[-1])'

case "$MODE" in
  native)
    HF_HOME="$HF_CACHE" "$PYTHON" -c "$PY" "$MODEL" "$MODEL_REVISION" ;;
  docker|docker-legacy)
    [[ "$VLLM_IMAGE" == *PIN_ME* ]] && { echo "set VLLM_IMAGE in config.sh first" >&2; exit 1; }
    sec=(); [ "$MODE" = docker-legacy ] && sec=(--security-opt seccomp=unconfined)
    docker run --rm "${sec[@]}" "${DOCKER_EXTRA[@]}" -e HF_TOKEN -v "$HF_CACHE":/root/.cache/huggingface \
      --entrypoint python3 "$VLLM_IMAGE" -c "$PY" "$MODEL" "$MODEL_REVISION" ;;
  *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac
du -sh "$HF_CACHE/hub/models--${MODEL//\//--}"
