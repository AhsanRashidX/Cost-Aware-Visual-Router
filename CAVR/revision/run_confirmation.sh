#!/usr/bin/env bash
# Confirmation set (fresh train-split questions, data_confirm/) through the same pipeline on the 2x A10 node.
# Frozen CAVR-CE routers are then applied with confirm_ce.py and confirm_eval.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=/d/envs/cavr/python.exe
export CAVR_DATA="$PWD/data_confirm" CAVR_RESULTS="$PWD/results_confirm"
mkdir -p "$CAVR_RESULTS" logs/confirm
# OCR (docTR) on GPU 0 and the ColPali index + per-query retrieval on GPU 1 are independent
CUDA_VISIBLE_DEVICES=0 $PY -u -W ignore ocr_pages.py > logs/confirm/ocr.log 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 $PY -u -W ignore retrieve_visual.py > logs/confirm/retrieve_visual.log 2>&1 & p1=$!
wait $p0; wait $p1
$PY -u retrieve_text.py > logs/confirm/retrieve_text.log 2>&1
HF_HUB_OFFLINE=1 $PY -u embed.py > logs/confirm/embed.log 2>&1
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u -W ignore generate.py --shard $i/2 \
    --out "$CAVR_RESULTS/path_outputs_$i.jsonl" > logs/confirm/generate_$i.log 2>&1 & pids[$i]=$!
done
wait ${pids[0]}; wait ${pids[1]}
cat "$CAVR_RESULTS"/path_outputs_0.jsonl "$CAVR_RESULTS"/path_outputs_1.jsonl > "$CAVR_RESULTS/path_outputs.jsonl"
cat "$CAVR_RESULTS"/path_outputs_*.oom.jsonl > "$CAVR_RESULTS/path_outputs_oom.jsonl" 2>/dev/null || : > "$CAVR_RESULTS/path_outputs_oom.jsonl"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -u -W ignore colpali_token_feats.py > logs/confirm/token_feats.log 2>&1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -u -W ignore confirm_ce.py --stage features > logs/confirm/ce_features.log 2>&1
unset CAVR_DATA CAVR_RESULTS
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY -u -W ignore confirm_ce.py --stage train > logs/confirm/ce_train.log 2>&1
export CAVR_DATA="$PWD/data_confirm" CAVR_RESULTS="$PWD/results_confirm"
$PY -u confirm_eval.py > logs/confirm/eval.log 2>&1
echo "confirmation pipeline done"
