#!/usr/bin/env bash
# Run a client script (pilot.py, loadgen.py) next to the servers: inside the vLLM image in the docker modes (the
# host needs no Python; host network, all GPUs visible for NVML energy, optional CPU pinning to CLIENT_CPUS), or
# with $PYTHON in native mode (equivalent to plain `python3 pilot.py ...` there).
# Usage:  bash client.sh pilot.py --replicas 1
cd "$(dirname "$0")"
source ./config.sh
export CAVR_MODE="$MODE" VLLM_IMAGE MODEL MODEL_REVISION SERVED_NAME IMAGE_TRANSPORT

if [ "$MODE" = native ]; then
  export SERVER_MEDIA_DIR="$PWD"
  pin=(); [ "$CPU_PIN" = 1 ] && pin=(taskset -c "$CLIENT_CPUS")
  exec "${pin[@]}" "$PYTHON" "$@"
fi
case "$MODE" in
  docker) gpu=(--gpus all) ;;
  docker-legacy) gpu=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --security-opt seccomp=unconfined) ;;
  *) echo "client.sh: unknown MODE=$MODE" >&2; exit 1 ;;
esac
[[ "$VLLM_IMAGE" == *PIN_ME* ]] && { echo "client.sh: set VLLM_IMAGE in config.sh first" >&2; exit 1; }
pin=(); [ "$CPU_PIN" = 1 ] && pin=(--cpuset-cpus "$CLIENT_CPUS")
# no -i/-t: works detached (nohup) as well; Ctrl-C is still forwarded to the script (docker's signal proxy)
exec docker run --rm --network host "${gpu[@]}" "${pin[@]}" "${DOCKER_EXTRA[@]}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -e CAVR_MODE -e VLLM_IMAGE -e MODEL -e MODEL_REVISION -e SERVED_NAME -e IMAGE_TRANSPORT -e SERVER_MEDIA_DIR=/data/t4 \
  -v "$PWD":/work -w /work --entrypoint python3 "$VLLM_IMAGE" "$@"
