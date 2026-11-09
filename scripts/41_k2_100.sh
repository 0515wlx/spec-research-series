#!/usr/bin/env bash
# Powered K2-Horizon-3.7B compression replication: 100 MATH + 100 WikiText.
# Target and compressed draft are the same K2-3.7B model, so full width has q=0.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MODEL=${MODEL:-k2-horizon-3.7b}
MATH=data/processed/k2_math100_target.jsonl
WIKI=data/processed/k2_wikitext100_target.jsonl
MCACHE=out/target_cache/k2_math100_k256.npz
WCACHE=out/target_cache/k2_wikitext100_k256.npz
mkdir -p data/processed out/target_cache out/rank_scan out/logs

if [ ! -f "$MATH" ]; then
  python -m infra.generate_target --target "$MODEL" --domain math500 \
    --data data/processed/math500.jsonl --limit 100 --max-new 256 --batch-size 4 \
    --out "$MATH"
fi
if [ ! -f "$WIKI" ]; then
  python -m infra.generate_target --target "$MODEL" --domain wikitext \
    --limit 100 --seq-len 1024 --out "$WIKI"
fi
if [ ! -f "$MCACHE" ]; then
  python -m infra.cache_target --target "$MODEL" --data "$MATH" --topk 256 \
    --batch-size 4 --out "$MCACHE"
fi
if [ ! -f "$WCACHE" ]; then
  python -m infra.cache_target --target "$MODEL" --data "$WIKI" --topk 256 \
    --batch-size 4 --out "$WCACHE"
fi
if [ ! -f out/importance/k2-horizon-3.7b_act.pt ]; then
  python -m infra.activation_importance --model "$MODEL" \
    --data data/processed/distill_controlled.jsonl --max-tokens 200000 \
    --out out/importance/k2-horizon-3.7b_act.pt
fi
python -m infra.joint_scan --model "$MODEL" --mode width \
  --importance out/importance/k2-horizon-3.7b_act.pt \
  --fracs 0.5,0.7,0.9,0.95,0.98,0.99 --batch-size 4 \
  --sets math:"$MATH":"$MCACHE" wikitext:"$WIKI":"$WCACHE" \
  --out out/rank_scan/k2-horizon-3.7b_actwidth_100
