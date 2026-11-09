#!/usr/bin/env bash
# Multi-seed gentle positive control (2B, 500k tokens, lr 5e-6).
#
# The paper's positive control is a single seed.  This script trains seeds
# 0/1/2 on the same held-out target-generated FineWeb split and evaluates all
# five checkpoints on the common caches.
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT=${STUDENT:-qwen3.5-2b}
TOKENS=${TOKENS:-500000}
LR=${LR:-5e-6}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0 1 2"}
DOMAINS=${DOMAINS:-"fineweb fineweb_gen wikitext wikitext_gen math500"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/positive_seeds out/logs

TRAIN=data/processed/fineweb_adapt_train.jsonl
VAL=data/processed/fineweb_adapt_val.jsonl
[ -f "$TRAIN" ] || { echo "missing $TRAIN"; exit 1; }

read -r -a GPU_ARR <<< "$GPUS"
i=0
for S in $SEEDS; do
  G=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
  OUT="ckpt/distill/${STUDENT}-fineweb-adapt${TOKENS}-lr${LR}-s${S}"
  i=$((i + 1))
  if ! find "$OUT" -maxdepth 1 -type d -name 'tokens-*' | grep -q .; then
    echo "== train seed $S on GPU $G -> $OUT =="
    CUDA_VISIBLE_DEVICES="$G" torchrun --nproc_per_node 1 -m infra.train_distill \
      --student "$STUDENT" --data "$TRAIN" --val "$VAL" --out "$OUT" \
      --max-tokens "$TOKENS" --seq-len 1024 --batch-tokens 2048 \
      --lr "$LR" --warmup 50 --objective ce --grad-checkpoint \
      --save-every-tokens "$TOKENS" --save-checkpoint-dirs \
      --seed "$S" 2>&1 | tee "out/logs/positive_seed${S}.log"
  fi
done

i=0
for S in $SEEDS; do
  G=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
  OUT="ckpt/distill/${STUDENT}-fineweb-adapt${TOKENS}-lr${LR}-s${S}"
  i=$((i + 1))
  CK="$(find "$OUT" -maxdepth 1 -type d -name 'tokens-*' | sort | tail -1)"
  [ -n "$CK" ] && [ -d "$CK" ] || { echo "missing checkpoint under $OUT"; continue; }
  for DOM in $DOMAINS; do
    DATA="data/processed/${DOM}_target.jsonl"
    CACHE="out/target_cache/${DOM}_k256.npz"
    OUTJ="out/accept/positive_seeds/${STUDENT}-fineweb-adapt${TOKENS}-lr${LR}-s${S}_${DOM}.json"
    [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
    echo "== eval seed $S / $DOM on GPU $G =="
    CUDA_VISIBLE_DEVICES="$G" python -m infra.accept_eval --target qwen3.5-35b-a3b \
      --draft "$CK" --data "$DATA" --cache "$CACHE" --topk 256 \
      --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map cuda:0 \
      --out "$OUTJ" 2>&1 | tee "out/logs/positive_seed${S}_${DOM}.log"
  done
done
