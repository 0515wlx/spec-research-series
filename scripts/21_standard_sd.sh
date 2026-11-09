#!/usr/bin/env bash
# Standard (stochastic) speculative sampling on a subset.
# Usage: GPUS="0,1" DOMAINS="math500 wikitext" LIMIT=15 bash scripts/21_standard_sd.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"math500 wikitext"}
LIMIT=${LIMIT:-15}
GAMMA=${GAMMA:-4}
MAX_NEW=${MAX_NEW:-96}
GPUS=${GPUS:-"0,1"}
SEED=${SEED:-0}

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/std_sd out/logs

for DOM in $DOMAINS; do
  DATA="data/processed/${DOM}_target.jsonl"
  [ -f "$DATA" ] || { echo "missing $DATA"; continue; }
  for D in $DRAFTS; do
    OUT="out/std_sd/${DOM}_${D}_g${GAMMA}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    python -m infra.spec_sampling --target "$TARGET" --draft "$D" --domain "$DOM" \
      --data "$DATA" --limit "$LIMIT" --gamma "$GAMMA" --max-new "$MAX_NEW" \
      --seed "$SEED" --device-map-target cuda:0 --device-map-draft cuda:1 \
      --out "$OUT" 2>&1 | tee "out/logs/std_sd_${DOM}_${D}_g${GAMMA}.log"
  done
done
