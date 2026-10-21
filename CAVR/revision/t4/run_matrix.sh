#!/usr/bin/env bash
# Run the load-test matrix: for each replica count, start the replicas, run loadgen.py for every policy, stop them.
# Needs results/pilot.json (pilot.py sweep) for the capacity-relative rate grid. Long-running: start it detached,
#   nohup bash run_matrix.sh > logs/matrix.log 2>&1 &      then follow with: tail -f logs/matrix.log
# Settings (environment): REPLICA_LIST="1 2 3" (use "1 2" on a 2-GPU machine),
#   POLICIES="always_hybrid always_visual cavr_ce_valsel", FRACTIONS=..., LOADGEN_ARGS="..." (extra loadgen flags)
set -o pipefail
cd "$(dirname "$0")"
source ./config.sh
REPLICA_LIST="${REPLICA_LIST:-1 2 3}"
POLICIES="${POLICIES:-always_hybrid always_visual cavr_ce_valsel}"
FRACTIONS="${FRACTIONS:-0.15,0.3,0.5,0.7,0.8,0.9,1.0,1.1}"
mkdir -p logs results
[ -f results/pilot.json ] || { echo "results/pilot.json missing: run the pilot first (README step 6)" >&2; exit 1; }

for n in $REPLICA_LIST; do
  echo "=== $(date '+%F %T') starting $n replica(s)"
  bash stop.sh > /dev/null 2>&1
  if ! bash serve.sh "$n"; then
    echo "serve.sh failed for $n replica(s); skipping" >&2
    bash stop.sh
    continue
  fi
  for policy in $POLICIES; do
    echo "=== $(date '+%F %T') $policy on $n replica(s)"
    # shellcheck disable=SC2086
    bash client.sh loadgen.py --policy "$policy" --replicas "$n" --rate-fractions "$FRACTIONS" $LOADGEN_ARGS \
      || echo "loadgen failed: $policy r$n (continuing)" >&2
  done
  bash stop.sh
done
echo "=== $(date '+%F %T') matrix done; send back: tar czf t4_results.tar.gz results logs"
