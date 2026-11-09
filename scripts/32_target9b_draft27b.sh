#!/usr/bin/env bash
# Total-parameter super-capacity control: target Qwen3.5-9B, drafts up to 27B.
#
# 27B total parameters > 9B target total parameters, so this tests whether a draft
# with more *total* parameters than the target removes the OOD residual.
# Uses GPUs 0 (target/cache) and 1 (draft eval).
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-9b}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"math500 wikitext"}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p data/processed out/target_cache out/accept/target9b out/logs

MATH=data/processed/math500_target_qwen3.5-9b.jsonl
if [ ! -f "$MATH" ]; then
  echo "== generate MATH-500 with 9B target =="
  CUDA_VISIBLE_DEVICES=0 python -m infra.generate_target --target "$TARGET" \
    --domain math500 --limit 500 --max-new 512 --batch-size 8 \
    --max-prompt-tokens 1536 --device-map cuda:0 \
    --out "$MATH" 2>&1 | tee out/logs/target9b_math_gen.log
fi

for DOM in $DOMAINS; do
  case "$DOM" in
    math500) DATA=$MATH ;;
    wikitext) DATA=data/processed/wikitext_target.jsonl ;;
    *) echo "unknown $DOM"; continue ;;
  esac
  CACHE="out/target_cache/${DOM}_target9b_k${TOPK}.npz"
  if [ ! -f "$CACHE" ]; then
    echo "== cache $DOM with 9B target =="
    CUDA_VISIBLE_DEVICES=0 python -m infra.cache_target --target "$TARGET" \
      --data "$DATA" --topk "$TOPK" --batch-size 4 --device-map cuda:0 \
      --out "$CACHE" 2>&1 | tee "out/logs/target9b_cache_${DOM}.log"
  fi
  for D in $DRAFTS; do
    OUT="out/accept/target9b/${DOM}_${D}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    case "$D" in
      *0.8b|*2b) DBS=8 ;;
      *4b|*9b)   DBS=4 ;;
      *27b)      DBS=1 ;;
      *)         DBS=2 ;;
    esac
    echo "== eval $DOM / $D with 9B target =="
    CUDA_VISIBLE_DEVICES=1 python -m infra.accept_eval --target "$TARGET" \
      --draft "$D" --data "$DATA" --cache "$CACHE" --topk "$TOPK" \
      --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" --device-map cuda:0 \
      --out "$OUT" 2>&1 | tee "out/logs/target9b_${DOM}_${D}.log"
  done
done
