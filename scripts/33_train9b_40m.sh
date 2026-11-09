#!/usr/bin/env bash
# 9B narrow distillation, 40M tokens, same corpus/recipe as the 2B/4B controls.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-ckpt/zero3/qwen3.5-9b-eq40m}
DATA=${DATA:-data/processed/distill_controlled.jsonl}
TOKENS=${TOKENS:-40000000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT" out/logs
if find "$OUT" -maxdepth 1 -type d -name 'tokens-*' | grep -q .; then
  echo "already has checkpoints: $OUT"; exit 0
fi
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m infra.train_zero3 \
  --student qwen3.5-9b --data "$DATA" --out "$OUT" \
  --max-tokens "$TOKENS" --seq-len 2048 --micro-seqs 8 --grad-accum 1 \
  --lr 5e-5 --warmup 50 --save-every-tokens 10000000 \
  --log-every 10 --ce-chunk 512 --no-zero-init \
  2>&1 | tee out/logs/eq40m_9b.log
