#!/usr/bin/env bash
# 27B FSDP full fine-tune on the mixed target corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDENT=${STUDENT:-qwen3.5-27b}
DATA=${DATA:-data/processed/distill_mixed.jsonl}
OUT=${OUT:-ckpt/fsdp/qwen3.5-27b-mixed20m}
TOKENS=${TOKENS:-20000000}
NPROC=${NPROC:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
mkdir -p "$OUT" out/logs
torchrun --nproc_per_node "$NPROC" -m infra.train_fsdp \
  --student "$STUDENT" --data "$DATA" --out "$OUT" \
  --max-tokens "$TOKENS" --seq-len 2048 --batch-tokens 8192 \
  --lr 5e-5 --warmup 100 \
  --save-every-tokens 5000000 --log-every 5
