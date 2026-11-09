#!/usr/bin/env bash
# Qwen3.6 target replication on raw FineWeb-Edu (no generated-OOD set).
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.6-35b-a3b}
DATA=${DATA:-data/processed/fineweb_target.jsonl}
CACHE=${CACHE:-out/target_cache/repl_fineweb_k256.npz}
TARGET_DM=${TARGET_DM:-cuda:6}
DRAFT_DM=${DRAFT_DM:-cuda:5}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/target_cache out/accept/repl out/fit/repl_fineweb out/logs

if [ ! -f "$CACHE" ]; then
  python -m infra.cache_target --target "$TARGET" --data "$DATA" \
    --topk 256 --batch-size 4 --device-map "$TARGET_DM" --out "$CACHE"
fi
for D in $DRAFTS; do
  OUT="out/accept/repl/repl_fineweb_${D}.json"
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  case "$D" in
    *0.8b|*2b) DBS=8 ;;
    *4b|*9b)   DBS=4 ;;
    *27b)      DBS=1 ;;
    *)         DBS=2 ;;
  esac
  python -m infra.accept_eval --target "$TARGET" --draft "$D" --data "$DATA" \
    --cache "$CACHE" --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
    --device-map "$DRAFT_DM" --out "$OUT"
done
python -m infra.scaling_fit --glob 'out/accept/repl/repl_fineweb_*.json' --out out/fit/repl_fineweb
