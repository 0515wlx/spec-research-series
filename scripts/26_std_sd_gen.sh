#!/usr/bin/env bash
# Standard-SD real loop on generated OOD (WikiText-gen + FineWeb-gen).
#
# 2 cells run concurrently, each cell uses one GPU for the target and one for
# the draft.  30 prompts/cell, gamma=4, 96 new tokens, all five drafts.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"wikitext_gen fineweb_gen"}
LIMIT=${LIMIT:-30}
GAMMA=${GAMMA:-4}
MAX_NEW=${MAX_NEW:-96}
SEED=${SEED:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/std_sd out/logs

run_one() {
  local DOM=$1 D=$2 T=$3 DR=$4
  local OUT="out/std_sd/gen_${DOM}_${D}_g${GAMMA}.json"
  [ -f "$OUT" ] && { echo "skip $OUT"; return; }
  echo "== $DOM / $D (target cuda:$T, draft cuda:$DR) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.spec_sampling --target "$TARGET" \
    --draft "$D" --domain "$DOM" --data "data/processed/${DOM}_target.jsonl" \
    --limit "$LIMIT" --gamma "$GAMMA" --max-new "$MAX_NEW" --seed "$SEED" \
    --device-map-target "cuda:$T" --device-map-draft "cuda:$DR" \
    --out "$OUT" 2>&1 | tee "out/logs/std_sd_gen_${DOM}_${D}_g${GAMMA}.log"
}

i=0
for DOM in $DOMAINS; do
  for D in $DRAFTS; do
    if [ $((i % 2)) -eq 0 ]; then
      run_one "$DOM" "$D" 0 1 &
      P0=$!
    else
      run_one "$DOM" "$D" 2 3 &
      P1=$!
      wait $P0 $P1
    fi
    i=$((i + 1))
  done
done
# If an odd number of tasks, wait for the last one.
if [ $((i % 2)) -eq 1 ]; then wait $P0; fi
