#!/usr/bin/env bash
# Third review round: repeat the live load matrix with new arrival traces, so every live rate has 5 runs.
# Same package, server settings, rate grid (pilot.json fractions) and policies as run_a10_wsl.sh / run_matrix.sh;
# only loadgen's --seed changes (1234 = the original run in results/load/). Outputs: results/load_r3/<policy>_r<N>_s<seed>/.
# Stages:  pilot    seed 1234, always_hybrid, 1 replica, the two lowest rate fractions -> results/load_r3_pilot/
#                   (must reproduce the original run's p95 at those rates)
#          repeats  SEEDS x POLICIES x REPLICA_LIST, skipping runs whose summary.json already exists (resumable)
# Usage (from Windows, detached): wsl -d Ubuntu-24.04 -- bash ~/cavr_serve/run_repeats_r3.sh [pilot|repeats]
#   Log: ~/cavr_serve/logs/r3_driver.log
export PROFILE=a10 MODE=native
export PYTHON="$HOME/vllm-env/bin/python"
export HF_CACHE="$HOME/hf_cache"
export PATH="$HOME/vllm-env/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
PKG="${PKG:-$HOME/cavr_serve}"
cd "$PKG" || exit 1
mkdir -p logs results/load_r3
exec >> logs/r3_driver.log 2>&1
SEEDS="${SEEDS:-1235 1236 1237 1238}"
POLICIES="${POLICIES:-always_hybrid always_visual cavr_ce_valsel}"
REPLICA_LIST="${REPLICA_LIST:-1 2}"
FRACTIONS="${FRACTIONS:-0.15,0.3,0.5,0.7,0.8,0.9,1.0,1.1}"
STAGE="${1:-repeats}"
# a run is complete when its summary.json has a stop_reason (loadgen rewrites summary.json after every rate)
done_run() { [ -f "$1/summary.json" ] && "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('stop_reason') else 1)" "$1/summary.json"; }
echo "=== $(date '+%F %T') r3 driver start: stage $STAGE seeds [$SEEDS] replicas [$REPLICA_LIST]"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
uname -r

case "$STAGE" in
  pilot)
    bash stop.sh > /dev/null 2>&1
    bash serve.sh 1 || { echo "serve.sh 1 failed"; bash stop.sh; exit 1; }
    bash client.sh loadgen.py --policy always_hybrid --replicas 1 --rate-fractions 0.15,0.3 --seed 1234 \
      --tag pilot --out-dir results/load_r3_pilot --overwrite || echo "pilot loadgen failed"
    bash stop.sh ;;
  repeats)
    for n in $REPLICA_LIST; do
      todo=0
      for seed in $SEEDS; do for policy in $POLICIES; do
        done_run "results/load_r3/${policy}_r${n}_s${seed}" || todo=1
      done; done
      [ "$todo" = 1 ] || { echo "r$n: all done"; continue; }
      echo "=== $(date '+%F %T') starting $n replica(s)"
      bash stop.sh > /dev/null 2>&1
      bash serve.sh "$n" || { echo "serve.sh $n failed"; bash stop.sh; continue; }
      for seed in $SEEDS; do
        for policy in $POLICIES; do
          d="results/load_r3/${policy}_r${n}_s${seed}"
          if done_run "$d"; then echo "skip $d"; continue; fi
          [ -d "$d" ] && mv "$d" "$d.incomplete_$(date +%H%M%S)"   # keep an interrupted run, never delete it
          echo "=== $(date '+%F %T') $policy r$n seed $seed"
          bash client.sh loadgen.py --policy "$policy" --replicas "$n" --rate-fractions "$FRACTIONS" \
            --seed "$seed" --tag "s$seed" --out-dir results/load_r3 \
            || { echo "loadgen failed: $policy r$n s$seed (continuing)"; mv "$d" "$d.failed_$(date +%H%M%S)" 2>/dev/null; }
        done
      done
      bash stop.sh
    done ;;
  *) echo "unknown stage $STAGE" ;;
esac
echo "=== $(date '+%F %T') r3 driver done: $STAGE"
