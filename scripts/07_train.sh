#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 2B / 4B full fine-tune examples.  For 9B/27B use FSDP (see docs/INFRA.md §7).
STUDENT=${STUDENT:-qwen3.5-2b}
DATA=${DATA:-data/processed/distill_target.jsonl}
OUT=${OUT:-ckpt/distill/${STUDENT}-100m}
TOKENS=${TOKENS:-100000000}
NPROC=${NPROC:-8}

torchrun --nproc_per_node "$NPROC" -m infra.train_distill \
  --student "$STUDENT" --data "$DATA" --out "$OUT" \
  --max-tokens "$TOKENS" --seq-len 2048 --batch-tokens 4096 \
  --lr 1e-4 --warmup 200 --objective ce --grad-checkpoint \
  --save-every-tokens 25000000 --val-every-tokens 5000000
