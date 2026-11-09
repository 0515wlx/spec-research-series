#!/usr/bin/env bash
# 27B DeepSpeed ZeRO-3 full fine-tune on the mixed target corpus.
#
# Replaces scripts/16_train27b.sh (FSDP), which deadlocked in a small NCCL
# collective on the Qwen3.5 hybrid linear-attention model.
#
# Notes:
#   * --no-zero-init: transformers 5.x rejects the [0]-shaped params that
#     deepspeed.zero.Init creates, so load on CPU and let deepspeed.initialize
#     partition instead.
#   * entry 8 x 1024 tokens was padded waste; length-grouped dynamic batches
#     average ~6 x 850 tokens with ~zero padding.
#   * micro-seqs 6 + ce-chunk 512 keeps peak memory ~71/80 GB (micro 8 hit
#     74.5 GB and occasionally failed 20 MB mappings).
set -euo pipefail
cd "$(dirname "$0")/.."
STUDENT=${STUDENT:-qwen3.5-27b}
DATA=${DATA:-data/processed/distill_mixed.jsonl}
OUT=${OUT:-ckpt/zero3/qwen3.5-27b-mixed20m}
TOKENS=${TOKENS:-20000000}
NPROC=${NPROC:-8}
MICRO=${MICRO:-6}
LR=${LR:-5e-5}
WARMUP=${WARMUP:-50}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
mkdir -p "$OUT" out/logs
torchrun --nproc_per_node "$NPROC" -m infra.train_zero3 \
  --student "$STUDENT" --data "$DATA" --out "$OUT" \
  --max-tokens "$TOKENS" --seq-len 2048 \
  --micro-seqs "$MICRO" --grad-accum 1 \
  --lr "$LR" --warmup "$WARMUP" --save-every-tokens 5000000 \
  --log-every 10 --ce-chunk 512 --no-zero-init
