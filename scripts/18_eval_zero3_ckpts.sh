#!/usr/bin/env bash
# Evaluate the trained 27B checkpoints with the existing target caches.
#
# The cache files were built once from the Qwen3.5-35B-A3B target; accept_eval
# then only runs the draft, so a 27B checkpoint fits on 1-2 GPUs.  Run this
# after scripts/17_train27b_zero3.sh has produced tokens-5M/10M/20M.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-ckpt/zero3/qwen3.5-27b-mixed20m}
CKPTS=${CKPTS:-"$ROOT/tokens-5000000 $ROOT/tokens-10000000 $ROOT/tokens-20000000"}
DOMAINS=${DOMAINS:-"math500 wikitext fineweb wikitext_gen fineweb_gen c4_gen gutenberg_gen code_gen"}
DRAFT_DM=${DRAFT_DM:-auto}
DBS=${DBS:-1}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/zero3 out/logs

for CK in $CKPTS; do
  [ -d "$CK" ] || { echo "missing checkpoint $CK"; continue; }
  TAG="$(basename "$ROOT")_$(basename "$CK")"
  for DOM in $DOMAINS; do
    case "$DOM" in
      math500)      DATA=data/processed/math500_target.jsonl ;;
      wikitext)     DATA=data/processed/wikitext_target.jsonl ;;
      fineweb)      DATA=data/processed/fineweb_target.jsonl ;;
      wikitext_gen) DATA=data/processed/wikitext_gen_target.jsonl ;;
      fineweb_gen)  DATA=data/processed/fineweb_gen_target.jsonl ;;
      c4_gen)       DATA=data/processed/c4_gen_target.jsonl ;;
      gutenberg_gen) DATA=data/processed/gutenberg_gen_target.jsonl ;;
      code_gen)     DATA=data/processed/code_gen_target.jsonl ;;
      *) echo "unknown domain $DOM"; continue ;;
    esac
    CACHE="out/target_cache/${DOM}_k${TOPK}.npz"
    OUT="out/accept/zero3/${TAG}_${DOM}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    [ -f "$DATA" ] || { echo "missing $DATA"; continue; }
    [ -f "$CACHE" ] || { echo "missing $CACHE"; continue; }
    echo "== $TAG / $DOM =="
    python -m infra.accept_eval --target qwen3.5-35b-a3b --draft "$CK" \
      --data "$DATA" --cache "$CACHE" --topk "$TOPK" \
      --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map "$DRAFT_DM" --out "$OUT"
  done
done
