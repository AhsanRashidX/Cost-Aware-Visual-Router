#!/usr/bin/env bash
# Experiment 3 on the 2x A10 node: vLLM pip-installed in WSL2 (Ubuntu 24.04), GPUs in MCDM mode.
# Runs the per-query service times on 2 replicas, then the load matrix on 1 and 2 replicas, unattended.
# The package lives on the WSL disk (~/cavr_serve = t4_package + this folder's config.sh), the model in ~/hf_cache.
# Usage (from Windows, detached):  wsl -d Ubuntu-24.04 -- bash <path>/run_a10_wsl.sh [stage ...]
#   stages: service matrix (default: both).   Log: ~/cavr_serve/logs/a10_driver.log
export PROFILE=a10 MODE=native
export PYTHON="$HOME/vllm-env/bin/python"
export HF_CACHE="$HOME/hf_cache"
export PATH="$HOME/vllm-env/bin:$PATH"
# FlashInfer's sampler JIT-compiles with nvcc, and WSL has no CUDA toolkit. All requests are greedy (temperature 0),
# so vLLM's PyTorch sampler returns the same tokens.
export VLLM_USE_FLASHINFER_SAMPLER=0
PKG="${PKG:-$HOME/cavr_serve}"
cd "$PKG" || exit 1
mkdir -p logs
exec >> logs/a10_driver.log 2>&1
STAGES=("${@:-service matrix}")
read -r -a STAGES <<< "${STAGES[*]}"
echo "=== $(date '+%F %T') driver start: stages ${STAGES[*]}  sampler_flashinfer=$VLLM_USE_FLASHINFER_SAMPLER"
for st in "${STAGES[@]}"; do
  case "$st" in
    service)
      bash stop.sh
      bash serve.sh 2 || { echo "serve.sh 2 failed"; bash stop.sh; exit 1; }
      bash client.sh pilot.py --replicas 2 --stages service || echo "service stage failed"
      bash stop.sh ;;
    matrix)
      REPLICA_LIST="${REPLICA_LIST:-1 2}" bash run_matrix.sh ;;
    *) echo "unknown stage $st" ;;
  esac
done
echo "=== $(date '+%F %T') driver done"
