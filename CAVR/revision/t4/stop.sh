#!/usr/bin/env bash
# Stop every replica started by serve.sh. Docker modes save each container's log to logs/ before removing it;
# native mode stops the recorded process groups and keeps a timestamped copy of each replica log.
# Usage:  bash stop.sh
cd "$(dirname "$0")"
source ./config.sh
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)

if [ "$MODE" = native ]; then
  for f in run/replica_*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f")
    i=$(basename "$f" .pid); i=${i#replica_}
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; fi
    cp "logs/replica_$i.log" "logs/replica_${i}_$ts.log" 2>/dev/null
    rm -f "$f"
    echo "stopped replica $i (pid $pid; log logs/replica_${i}_$ts.log)"
  done
else
  for name in $(docker ps -a --format '{{.Names}}' | grep "^$NAME_PREFIX-"); do
    docker logs "$name" > "logs/${name}_$ts.log" 2>&1
    docker rm -f "$name" >/dev/null && echo "stopped $name (log logs/${name}_$ts.log)"
  done
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
