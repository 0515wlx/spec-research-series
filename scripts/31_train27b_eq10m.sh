#!/usr/bin/env bash
# 27B target-only 10M-token control on 4 GPUs (micro=1 to fit Adam states).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-ckpt/zero3/qwen3.5-27b-eq10m-target}
DATA=${DATA:-data/processed/distill_controlled.jsonl}
TOKENS=${TOKENS:-10000000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT" out/logs
if find "$OUT" -maxdepth 1 -type d -name 'tokens-*' | grep -q .; then
  echo "already trained: $OUT"; exit 0
fi
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m infra.train_zero3 \
  --student qwen3.5-27b --data "$DATA" --out "$OUT" \
  --max-tokens "$TOKENS" --seq-len 1024 --micro-seqs 1 --grad-accum 4 \
  --lr 5e-5 --warmup 50 --save-every-tokens "$TOKENS" \
  --log-every 10 --ce-chunk 512 --no-zero-init --offload-optimizer \
  2>&1 | tee out/logs/eq10m_27b.log
