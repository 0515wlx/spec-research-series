#!/usr/bin/env bash
# Sampling-matched OOD cache: target samples at T=1 instead of greedy.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p data/processed out/target_cache out/logs

if [ ! -f data/processed/wikitext_gen_sample_target.jsonl ]; then
  echo "== sample WikiText-gen with target =="
  CUDA_VISIBLE_DEVICES=2,3 python -m infra.generate_target --target "$TARGET" \
    --domain wikitext --generate-wikitext --limit 100 --max-new 512 \
    --sample --temperature 1.0 --seed 0 --batch-size 4 --device-map auto \
    --out data/processed/wikitext_gen_sample_target.jsonl \
    2>&1 | tee out/logs/wikitext_gen_sample.log
fi
if [ ! -f data/processed/fineweb_gen_sample_target.jsonl ]; then
  echo "== sample FineWeb-gen with target =="
  CUDA_VISIBLE_DEVICES=2,3 python -m infra.generate_target --target "$TARGET" \
    --domain fineweb --data data/processed/fineweb.jsonl --limit 100 --max-new 512 \
    --sample --temperature 1.0 --seed 0 --batch-size 4 --device-map auto \
    --out data/processed/fineweb_gen_sample_target.jsonl \
    2>&1 | tee out/logs/fineweb_gen_sample.log
fi

for DOM in wikitext_gen fineweb_gen; do
  CACHE="out/target_cache/${DOM}_sample_k${TOPK}.npz"
  if [ ! -f "$CACHE" ]; then
    echo "== cache sampled $DOM =="
    CUDA_VISIBLE_DEVICES=2,3 python -m infra.cache_target --target "$TARGET" \
      --data "data/processed/${DOM}_sample_target.jsonl" --topk "$TOPK" \
      --batch-size 4 --device-map auto --out "$CACHE" \
      2>&1 | tee "out/logs/${DOM}_sample_cache.log"
  fi
done
