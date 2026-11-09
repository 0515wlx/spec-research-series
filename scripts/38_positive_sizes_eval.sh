#!/usr/bin/env bash
# Evaluate the 0.8B/4B gentle positive-control checkpoints on the larger held-out sets.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/positive_more out/logs
i=0
for SIZE in 0.8b 4b; do
  for S in 0 1 2; do
    CK="$(find "ckpt/distill/qwen3.5-${SIZE}-fineweb-adapt500000-lr5e-6-s${S}" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1)"
    [ -n "$CK" ] && [ -d "$CK" ] || { echo "missing $SIZE s$S"; continue; }
    for DOM in wikitext_gen500 fineweb_heldout200; do
      OUTJ="out/accept/positive_more/${SIZE}_s${S}_${DOM}.json"
      [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
      case "$DOM" in
        wikitext_gen500) DATA=data/processed/wikitext_gen500_target.jsonl ;;
        fineweb_heldout200) DATA=data/processed/fineweb_heldout200_target.jsonl ;;
      esac
      G=$((i % 4))
      echo "== $SIZE s$S / $DOM -> GPU$G =="
      CUDA_VISIBLE_DEVICES=$G python -m infra.accept_eval --target "$TARGET" \
        --draft "$CK" --data "$DATA" --cache "out/target_cache/${DOM}_k${TOPK}.npz" --topk "$TOPK" \
        --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map cuda:0 \
        --out "$OUTJ" 2>&1 | tee "out/logs/positive_sizes_eval_${SIZE}_s${S}_${DOM}.log"
      i=$((i+1))
    done
  done
done
