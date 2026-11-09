#!/usr/bin/env bash
# Qwen3 draft-family replication against a Qwen3-32B target.
# Requires Qwen3-tokenized raw WikiText/FineWeb files (created separately).
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3-32b}
TARGET_DM=${TARGET_DM:-cuda:6}
DRAFT_DM=${DRAFT_DM:-cuda:7}
DRAFTS=${DRAFTS:-"qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/target_cache out/accept/qwen3 out/fit out/logs

for NAME in wikitext fineweb; do
  DATA="data/processed/qwen3_${NAME}_target.jsonl"
  CACHE="out/target_cache/qwen3_${NAME}_k256.npz"
  [ -f "$DATA" ] || { echo "missing $DATA"; exit 1; }
  if [ ! -f "$CACHE" ]; then
    python -m infra.cache_target --target "$TARGET" --data "$DATA" \
      --topk 256 --batch-size 1 --device-map "$TARGET_DM" --out "$CACHE"
  fi
  for D in $DRAFTS; do
    OUT="out/accept/qwen3/qwen3_${NAME}_${D}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    case "$D" in
      *0.6b|*1.7b) DBS=8 ;;
      *4b)          DBS=4 ;;
      *8b)          DBS=2 ;;
      *14b)         DBS=1 ;;
      *)            DBS=2 ;;
    esac
    python -m infra.accept_eval --target "$TARGET" --draft "$D" --data "$DATA" \
      --cache "$CACHE" --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map "$DRAFT_DM" --out "$OUT"
  done
  python -m infra.scaling_fit --glob "out/accept/qwen3/qwen3_${NAME}_*.json" \
    --out "out/fit/qwen3_${NAME}"
done
