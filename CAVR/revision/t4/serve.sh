#!/usr/bin/env bash
# Start N vLLM OpenAI-compatible replicas, replica i on GPU i and port BASE_PORT+i, and wait until all are healthy.
# Usage:  bash serve.sh <n_replicas>          (mode, image, model, CPU pinning and vLLM flags: config.sh)
# Writes results/serve_info.txt (mode, exact launch commands, image digest or vLLM version, host and driver);
# in native mode also logs/replica_<i>.log and run/replica_<i>.pid. Stop with: bash stop.sh
set -eo pipefail
cd "$(dirname "$0")"
source ./config.sh
N="${1:?usage: bash serve.sh <n_replicas>}"
mkdir -p results logs run

die() { echo "serve.sh: $*" >&2; exit 1; }

case "$MODE" in
  docker|docker-legacy)
    [[ "$VLLM_IMAGE" == *PIN_ME* ]] && die "set VLLM_IMAGE in config.sh to a pinned tag first"
    command -v docker >/dev/null || die "docker not found (MODE=$MODE)" ;;
  native)
    read -r -a LAUNCH <<< "$NATIVE_CMD" ;;
  *) die "unknown MODE=$MODE (docker | docker-legacy | native)" ;;
esac
if command -v nvidia-smi >/dev/null; then
  n_gpu=$(nvidia-smi -L | grep -c '^GPU' || true)
  (( N <= n_gpu )) || die "asked for $N replicas but nvidia-smi lists $n_gpu GPU(s)"
fi

# where the server sees this package (only used with IMAGE_TRANSPORT=file)
if [ "$MODE" = native ]; then MEDIA_DIR="$PWD"; else MEDIA_DIR=/data/t4; fi
MEDIA_ARGS=()
[ "$IMAGE_TRANSPORT" = file ] && MEDIA_ARGS=(--allowed-local-media-path "$MEDIA_DIR")

{
  echo "date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "mode: $MODE   replicas: $N   base_port: $BASE_PORT"
  echo "model: $MODEL   revision: $MODEL_REVISION   served_name: $SERVED_NAME"
  echo "image_transport: $IMAGE_TRANSPORT   cpu_pin: $CPU_PIN (server cpus $SERVER_CPUS)   attn_backend: ${ATTN_BACKEND:-default}"
  echo "host: $(uname -a)"
  cat /etc/redhat-release 2>/dev/null || grep PRETTY_NAME /etc/os-release 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv 2>/dev/null || true
  if [ "$MODE" = native ]; then
    "$PYTHON" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2>&1 | tail -1 || true
  else
    docker version --format 'docker server {{.Server.Version}}' 2>/dev/null || true
    echo "image: $VLLM_IMAGE"
    docker image inspect --format 'image_id: {{.Id}}  repo_digests: {{json .RepoDigests}}' "$VLLM_IMAGE" 2>/dev/null \
      || echo "image not pulled yet (docker run pulls it)"
  fi
} > results/serve_info.txt 2>&1

for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  args=(--model "$MODEL" --revision "$MODEL_REVISION" --host 0.0.0.0 --port "$port" --served-model-name "$SERVED_NAME"
        "${VLLM_ARGS[@]}" "${QUANT_ARGS[@]}" "${LIMIT_MM_ARGS[@]}" "${MM_KWARGS_ARGS[@]}" "${CONTENT_FORMAT_ARGS[@]}"
        "${GEN_CONFIG_ARGS[@]}" "${PREFIX_CACHE_ARGS[@]}" "${LOG_ARGS[@]}" "${MEDIA_ARGS[@]}" "${EXTRA_ARGS[@]}")
  if [ "$MODE" = native ]; then
    pin=(); [ "$CPU_PIN" = 1 ] && pin=(taskset -c "$SERVER_CPUS")
    sess=(); command -v setsid >/dev/null && sess=(setsid)      # own process group, so stop.sh can kill it whole
    envs=(CUDA_VISIBLE_DEVICES="$i" CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1)
    [ -n "$ATTN_BACKEND" ] && envs+=(VLLM_ATTENTION_BACKEND="$ATTN_BACKEND")
    cmd=(env "${envs[@]}" nohup "${sess[@]}" "${pin[@]}" "${LAUNCH[@]}" "${args[@]}")
    "${cmd[@]}" > "logs/replica_$i.log" 2>&1 < /dev/null &
    echo $! > "run/replica_$i.pid"
  else
    name="$NAME_PREFIX-$i"
    docker rm -f "$name" >/dev/null 2>&1 || true
    if [ "$MODE" = docker ]; then
      gpu=(--gpus "\"device=$i\"")
    else
      gpu=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$i" --security-opt seccomp=unconfined)
    fi
    pin=(); [ "$CPU_PIN" = 1 ] && pin=(--cpuset-cpus "$SERVER_CPUS")
    envs=(-e HF_HUB_OFFLINE=1)
    [ -n "$ATTN_BACKEND" ] && envs+=(-e VLLM_ATTENTION_BACKEND="$ATTN_BACKEND")
    cmd=(docker run -d --name "$name" "${gpu[@]}" --ipc=host --network host "${pin[@]}" "${DOCKER_EXTRA[@]}" "${envs[@]}"
         -v "$HF_CACHE":/root/.cache/huggingface -v "$PWD":/data/t4:ro
         --entrypoint "${DOCKER_LAUNCH[0]}" "$VLLM_IMAGE" "${DOCKER_LAUNCH[@]:1}" "${args[@]}")
    "${cmd[@]}" > /dev/null
  fi
  { printf 'launch[%d]:' "$i"; printf ' %q' "${cmd[@]}"; echo; } >> results/serve_info.txt
  echo "started replica $i (GPU $i, port $port, mode $MODE)"
done

http_ok() {   # $1 = URL; curl, else wget, else Python
  if command -v curl >/dev/null; then curl -sf -m 5 -o /dev/null "$1"
  elif command -v wget >/dev/null; then wget -q -T 5 -O /dev/null "$1"
  else "$PYTHON" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" "$1" >/dev/null 2>&1
  fi
}
alive() {     # $1 = replica index
  if [ "$MODE" = native ]; then kill -0 "$(cat "run/replica_$1.pid")" 2>/dev/null
  else [ "$(docker inspect -f '{{.State.Running}}' "$NAME_PREFIX-$1" 2>/dev/null)" = true ]
  fi
}
show_log() {  # $1 = replica index
  if [ "$MODE" = native ]; then tail -n 60 "logs/replica_$1.log"; else docker logs --tail 60 "$NAME_PREFIX-$1" 2>&1; fi
}

deadline=$((SECONDS + WAIT_TIMEOUT))
for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  until http_ok "http://127.0.0.1:$port/health" && http_ok "http://127.0.0.1:$port/v1/models"; do
    alive "$i" || { show_log "$i"; die "replica $i exited during start-up (log above)"; }
    (( SECONDS < deadline )) || { show_log "$i"; die "replica $i not healthy after ${WAIT_TIMEOUT}s"; }
    sleep 5
  done
  echo "replica $i healthy: http://127.0.0.1:$port  ($((SECONDS))s)"
done
echo "all $N replica(s) up. Stop with: bash stop.sh"
