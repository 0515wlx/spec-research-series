#!/usr/bin/env bash
# Small positive control for the alignment/coverage axis:
# continue-train 2B on held-out target-generated FineWeb, then re-measure q.
#
# The first 100 FineWeb prompts are the evaluation set (same ids as
# data/processed/fineweb_target.jsonl).  We train only on prompts 100--1899 and
# validate on 1900--1999.
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT=${STUDENT:-qwen3.5-2b}
OUT=${OUT:-ckpt/distill/qwen3.5-2b-fineweb-adapt2m}
TOKENS=${TOKENS:-2000000}
NPROC=${NPROC:-1}
DOMAINS=${DOMAINS:-"fineweb fineweb_gen wikitext wikitext_gen math500"}
DBS=${DBS:-8}
TOPK=${TOPK:-256}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/positive out/logs

TRAIN=data/processed/fineweb_adapt_train.jsonl
VAL=data/processed/fineweb_adapt_val.jsonl
[ -f "$TRAIN" ] || { echo "missing $TRAIN"; exit 1; }

if [ ! -d "$OUT/tokens-$TOKENS" ]; then
  torchrun --nproc_per_node "$NPROC" -m infra.train_distill \
    --student "$STUDENT" --data "$TRAIN" --val "$VAL" --out "$OUT" \
    --max-tokens "$TOKENS" --seq-len 1024 --batch-tokens 2048 \
    --lr 2e-5 --warmup 50 --objective ce --grad-checkpoint \
    --save-every-tokens 500000 --save-checkpoint-dirs \
    --val-every-tokens 500000 2>&1 | tee out/logs/positive_control_train.log
fi

# Evaluate every intermediate checkpoint on the common caches.
for CK in "$OUT"/tokens-*; do
  [ -d "$CK" ] || continue
  TAG="$(basename "$OUT")_$(basename "$CK")"
  for DOM in $DOMAINS; do
    case "$DOM" in
      fineweb)      DATA=data/processed/fineweb_target.jsonl ;;
      fineweb_gen)  DATA=data/processed/fineweb_gen_target.jsonl ;;
      wikitext)     DATA=data/processed/wikitext_target.jsonl ;;
      wikitext_gen) DATA=data/processed/wikitext_gen_target.jsonl ;;
      math500)      DATA=data/processed/math500_target.jsonl ;;
      *) echo "unknown domain $DOM"; continue ;;
    esac
    CACHE="out/target_cache/${DOM}_k${TOPK}.npz"
    OUTJ="out/accept/positive/${TAG}_${DOM}.json"
    [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
    [ -f "$CACHE" ] || { echo "missing $CACHE"; continue; }
    echo "== $TAG / $DOM =="
    CUDA_VISIBLE_DEVICES=0 python -m infra.accept_eval --target qwen3.5-35b-a3b \
      --draft "$CK" --data "$DATA" --cache "$CACHE" --topk "$TOPK" \
      --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map cuda:0 --out "$OUTJ" 2>&1 | tee "out/logs/positive_${TAG}_${DOM}.log"
  done
done
