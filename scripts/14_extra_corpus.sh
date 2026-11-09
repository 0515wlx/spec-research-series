#!/usr/bin/env bash
# Extra raw-text OOD corpus: target continuations + 5-draft scan.
# Usage: CORPUS=c4 bash scripts/14_extra_corpus.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CORPUS=${CORPUS:?set CORPUS=c4|gutenberg|code}
TARGET=${TARGET:-qwen3.5-35b-a3b}
PROMPTS=${PROMPTS:-data/processed/${CORPUS}.jsonl}
DATA=${DATA:-data/processed/${CORPUS}_gen_target.jsonl}
CACHE=${CACHE:-out/target_cache/${CORPUS}_gen_k256.npz}
TARGET_DM=${TARGET_DM:-cuda:0}
DRAFT_DM=${DRAFT_DM:-cuda:1}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/target_cache out/accept out/fit/${CORPUS}_gen out/logs

if [ ! -f "$DATA" ]; then
  python -m infra.generate_target --target "$TARGET" --domain "$CORPUS" \
    --data "$PROMPTS" --limit 100 --max-new 512 --batch-size 4 \
    --device-map "$TARGET_DM" --out "$DATA"
fi
if [ ! -f "$CACHE" ]; then
  python -m infra.cache_target --target "$TARGET" --data "$DATA" \
    --topk 256 --batch-size 4 --device-map "$TARGET_DM" --out "$CACHE"
fi
for D in $DRAFTS; do
  OUT="out/accept/${CORPUS}_gen_${D}.json"
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
python -m infra.scaling_fit --glob "out/accept/${CORPUS}_gen_*.json" --out out/fit/${CORPUS}_gen
