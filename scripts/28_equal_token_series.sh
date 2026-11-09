#!/usr/bin/env bash
# Equal-token controlled distillation series (10M tokens, same corpus/recipe).
#
# Modes:
#   bash scripts/28_equal_token_series.sh 0.8b
#   bash scripts/28_equal_token_series.sh 4b
#   bash scripts/28_equal_token_series.sh 9b
#   bash scripts/28_equal_token_series.sh 27b
#   bash scripts/28_equal_token_series.sh eval 0.8b [checkpoint]
#
# 0.8B/4B use the DDP trainer; 9B/27B use the ZeRO-3 trainer (4 GPUs).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:?mode required: 0.8b|4b|9b|27b|eval}
DATA=${DATA:-data/processed/distill_controlled.jsonl}
TOKENS=${TOKENS:-10000000}
SEQ=${SEQ:-1024}
LR=${LR:-5e-5}
WARMUP=${WARMUP:-50}
DOMAINS=${DOMAINS:-"math500 wikitext wikitext_gen fineweb fineweb_gen"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/eq10m out/logs ckpt/distill ckpt/zero3

train_ddp() {
  local KEY=$1 OUT=$2
  if find "$OUT" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | grep -q .; then
    echo "== skip existing $OUT =="; return 0
  fi
  echo "== DDP $KEY 10M -> $OUT =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m infra.train_distill \
    --student "qwen3.5-${KEY}" --data "$DATA" --out "$OUT" \
    --max-tokens "$TOKENS" --seq-len "$SEQ" --batch-tokens 8192 \
    --lr "$LR" --warmup "$WARMUP" --objective ce --grad-checkpoint \
    --save-every-tokens "$TOKENS" --save-checkpoint-dirs \
    2>&1 | tee "out/logs/eq10m_${KEY}.log"
}

train_zero3() {
  local KEY=$1 OUT=$2 MICRO=$3
  if find "$OUT" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | grep -q .; then
    echo "== skip existing $OUT =="; return 0
  fi
  echo "== ZeRO-3 $KEY 10M -> $OUT (micro=$MICRO) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m infra.train_zero3 \
    --student "qwen3.5-${KEY}" --data "$DATA" --out "$OUT" \
    --max-tokens "$TOKENS" --seq-len 2048 --micro-seqs "$MICRO" --grad-accum 1 \
    --lr "$LR" --warmup "$WARMUP" --save-every-tokens "$TOKENS" \
    --log-every 10 --ce-chunk 512 --no-zero-init \
    2>&1 | tee "out/logs/eq10m_${KEY}.log"
}

case "$MODE" in
  0.8b) train_zero3 0.8b ckpt/zero3/qwen3.5-0.8b-eq10m 8 || echo "0.8b trainer returned nonzero" ;;
  4b)   train_zero3 4b   ckpt/zero3/qwen3.5-4b-eq10m   8 || echo "4b trainer returned nonzero" ;;
  9b)   train_zero3 9b  ckpt/zero3/qwen3.5-9b-eq10m  8 || echo "9b trainer returned nonzero" ;;
  27b)  train_zero3 27b ckpt/zero3/qwen3.5-27b-eq10m-target  2 || echo "27b trainer returned nonzero" ;;
  eval)
    KEY=${2:?eval needs key}
    CK=${3:-}
    if [ -z "$CK" ]; then
      case "$KEY" in
        0.8b|4b|9b)
          BASE=ckpt/zero3/qwen3.5-${KEY}-eq10m
          CK="$(find "$BASE" -maxdepth 1 -type d -name 'tokens-*' | sort | tail -1)"
          ;;
        27b)
          BASE=ckpt/zero3/qwen3.5-27b-eq10m-target
          CK="$(find "$BASE" -maxdepth 1 -type d -name 'tokens-*' | sort | tail -1)"
          ;;
      esac
    fi
    echo "== eval $KEY from $CK =="
    for DOM in $DOMAINS; do
      OUTJ="out/accept/eq10m/${KEY}_${DOM}.json"
      [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.accept_eval --target qwen3.5-35b-a3b \
        --draft "$CK" --data "data/processed/${DOM}_target.jsonl" \
        --cache "out/target_cache/${DOM}_k256.npz" --topk 256 \
        --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map auto \
        --out "$OUTJ" 2>&1 | tee "out/logs/eq10m_${KEY}_${DOM}.log"
    done
    ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
