# Settings shared by serve.sh, stop.sh, client.sh and download_model.sh (sourced, not run).
# Edit here, or override any ${VAR:-default} below from the environment, e.g.  MODE=native bash serve.sh 2

# ---- launch mode ------------------------------------------------------------------------------------------------
#   docker         Docker >= 19.03 + nvidia-container-toolkit:   --gpus '"device=i"'
#   docker-legacy  Docker 18.06 + nvidia-container-runtime 2.0:  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=i, plus
#                  --security-opt seccomp=unconfined (old Docker seccomp profiles break glibc >= 2.34 images)
#   native         no Docker: vLLM pip-installed in $PYTHON (e.g. a Kaggle notebook with 2x T4); replicas run as
#                  background processes with CUDA_VISIBLE_DEVICES=i, logs in logs/, PIDs in run/
MODE="${MODE:-docker}"

# ---- what to serve ----------------------------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:PIN_ME}"   # <-- PIN the image tag before running (docker modes)
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
MODEL_REVISION="${MODEL_REVISION:-main}"              # <-- pin to the commit hash printed by download_model.sh
SERVED_NAME="${SERVED_NAME:-qwen2.5-vl-7b-awq}"       # name the clients see in /v1/models
HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"                # Hugging Face home; model files land in $HF_CACHE/hub
PYTHON="${PYTHON:-python3}"                           # native mode: the interpreter that has vLLM installed
BASE_PORT="${BASE_PORT:-8000}"                        # replica i listens on BASE_PORT + i
NAME_PREFIX="${NAME_PREFIX:-cavr-vllm}"               # docker container names: cavr-vllm-0, cavr-vllm-1, ...
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"                  # seconds serve.sh waits for all replicas to be healthy

# ---- CPU pinning (optional) ---------------------------------------------------------------------------------------
# On the 3x T4 server all GPUs hang off NUMA node 1 (CPUs 16-31,48-63); the client then runs on node 0.
# Default: on in the docker modes, off in native mode (unknown topology, e.g. Kaggle). Set CPU_PIN=0/1 to force.
if [ "$MODE" = native ]; then CPU_PIN="${CPU_PIN:-0}"; else CPU_PIN="${CPU_PIN:-1}"; fi
SERVER_CPUS="${SERVER_CPUS:-16-31,48-63}"
CLIENT_CPUS="${CLIENT_CPUS:-0-15,32-47}"

# ---- images -------------------------------------------------------------------------------------------------------
# data = base64 data URL in every request (works with every vLLM version; default)
# file = the server reads this package's images/ folder (needs --allowed-local-media-path, set by serve.sh)
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-data}"

# Fixed by generate.py (make_t4_package.py checks these): 256*28*28 and 1280*28*28 pixels.
MIN_PIXELS=200704
MAX_PIXELS=1003520

# ---- vLLM server arguments of the study ---------------------------------------------------------------------------
VLLM_ARGS=(--dtype half --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 16 --seed 1234)

# ---- vLLM-version-sensitive arguments: check `--help` of the pinned version and adjust --------------------------
# T4 = compute capability 7.5: the awq_marlin kernels need >= 8.0, so force the plain AWQ kernels.
QUANT_ARGS=(--quantization awq)
# One image per prompt. JSON form for vLLM >= ~0.8; older releases take: (--limit-mm-per-prompt image=1)
LIMIT_MM_ARGS=(--limit-mm-per-prompt '{"image": 1}')
# Same pixel limits as generate.py. Images are already pre-resized, so this mainly bounds vLLM's memory profiling
# (the processor default allows 16k visual tokens). Key names follow the HF Qwen2-VL image processor.
MM_KWARGS_ARGS=(--mm-processor-kwargs "{\"min_pixels\": $MIN_PIXELS, \"max_pixels\": $MAX_PIXELS}")
# Pass list-style message content to the chat template unchanged (image part stays before the text, exactly as
# generate.py builds it). Available since ~0.6.4; delete the line on older releases (auto-detection picks openai).
CONTENT_FORMAT_ARGS=(--chat-template-content-format openai)
# Ignore the model's generation_config.json defaults; the client sends every sampling field explicitly
# (temperature 0, max_tokens 32, repetition_penalty 1.05, stop_token_ids). Flag exists since ~0.6.5; default
# changed to "auto" in later releases.
GEN_CONFIG_ARGS=(--generation-config vllm)
# The V1 engine caches prefixes (incl. images) by default; requests drawn with replacement would then skip prefill.
# Off for comparability with the 4090 run and the simulator. Older (V0) releases: () (off by default there).
PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
# Per-request logging: older releases log every request unless (--disable-log-requests); newer ones are quiet.
LOG_ARGS=()
# Anything else, e.g. (--enforce-eager) if CUDA-graph capture fails on the T4, or (--max-num-batched-tokens 8192).
EXTRA_ARGS=()
# Attention backend override, exported as VLLM_ATTENTION_BACKEND (empty = vLLM's choice for compute capability 7.5;
# candidates depending on the release: FLASHINFER, TRITON_ATTN, XFORMERS, FLEX_ATTENTION).
ATTN_BACKEND="${ATTN_BACKEND:-}"

# ---- launchers -----------------------------------------------------------------------------------------------------
# Module launch works in old and new vLLM releases; the model is appended as --model "$MODEL".
DOCKER_LAUNCH=(python3 -m vllm.entrypoints.openai.api_server)
NATIVE_CMD="${NATIVE_CMD:-$PYTHON -m vllm.entrypoints.openai.api_server}"   # split on spaces
# Extra `docker run` options for servers and client, e.g. (--security-opt seccomp=unconfined) in docker mode if
# threads fail to start on the old CentOS 7 kernel, or (-e NVIDIA_DISABLE_REQUIRE=1) if the legacy runtime refuses
# the image's CUDA requirement.
DOCKER_EXTRA=()
