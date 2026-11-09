#!/usr/bin/env bash
# Train gentle 0.8B and 4B positive-control checkpoints (500k, lr 5e-6, seeds 0/1/2).
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN=data/processed/fineweb_adapt_train.jsonl
VAL=data/processed/fineweb_adapt_val.jsonl
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/logs
for S in 0 1 2; do
  for SIZE in 0.8b 4b; do
    OUT="ckpt/distill/qwen3.5-${SIZE}-fineweb-adapt500000-lr5e-6-s${S}"
    if find "$OUT" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | grep -q .; then
      echo "skip $OUT"; continue
    fi
    G=0; [ "$SIZE" = "0.8b" ] && G=1
    echo "== train $SIZE seed $S on GPU $G =="
    CUDA_VISIBLE_DEVICES=$G torchrun --nproc_per_node 1 -m infra.train_distill \
      --student "qwen3.5-${SIZE}" --data "$TRAIN" --val "$VAL" --out "$OUT" \
      --max-tokens 500000 --seq-len 1024 --batch-tokens 2048 \
      --lr 5e-6 --warmup 50 --objective ce --grad-checkpoint \
      --save-every-tokens 500000 --save-checkpoint-dirs --seed "$S" \
      > "out/logs/positive_sizes_${SIZE}_s${S}.log" 2>&1 &
  done
  wait
done
