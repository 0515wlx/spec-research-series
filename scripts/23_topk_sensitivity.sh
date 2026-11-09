#!/usr/bin/env bash
# Top-k truncation sensitivity for the OOD residual.
#
# Re-caches the target with top-1024 and re-runs all five drafts on the same
# four OOD sets.  The paper's main cache is top-256 (+ residual bucket); this
# checks whether low-margin raw-OOD tokens are sensitive to the tail cut.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
TOPK=${TOPK:-1024}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"wikitext fineweb wikitext_gen fineweb_gen"}
GPUS=${GPUS:-"0,1"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/target_cache out/accept/topk${TOPK} out/logs

for DOM in $DOMAINS; do
  DATA="data/processed/${DOM}_target.jsonl"
  CACHE="out/target_cache/${DOM}_k${TOPK}.npz"
  [ -f "$CACHE" ] || {
    echo "== cache $DOM k$TOPK =="
    CUDA_VISIBLE_DEVICES="$GPUS" python -m infra.cache_target --target "$TARGET" \
      --data "$DATA" --topk "$TOPK" --batch-size 4 --device-map auto \
      --out "$CACHE" 2>&1 | tee "out/logs/topk_cache_${DOM}.log"
  }
  for D in $DRAFTS; do
    OUT="out/accept/topk${TOPK}/${DOM}_${D}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    case "$D" in
      *0.8b|*2b) DBS=8 ;;
      *4b|*9b)   DBS=4 ;;
      *27b)      DBS=1 ;;
      *)         DBS=2 ;;
    esac
    echo "== $DOM / $D =="
    CUDA_VISIBLE_DEVICES="$GPUS" python -m infra.accept_eval --target "$TARGET" \
      --draft "$D" --data "$DATA" --cache "$CACHE" --topk "$TOPK" \
      --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" --device-map auto \
      --out "$OUT" 2>&1 | tee "out/logs/topk_${DOM}_${D}.log"
  done
done
