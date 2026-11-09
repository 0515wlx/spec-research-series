#!/usr/bin/env bash
# Evaluate the 9B 40M narrow-distillation checkpoints on MATH and OOD.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
BASE=${BASE:-ckpt/zero3/qwen3.5-9b-eq40m}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/eq40m out/logs

for CK in $(find "$BASE" -maxdepth 1 -type d -name 'tokens-*' | sort); do
  TAG=$(basename "$CK")
  for DOM in math500 wikitext fineweb fineweb_gen; do
    OUTJ="out/accept/eq40m/${TAG}_${DOM}.json"
    [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
    echo "== $TAG / $DOM =="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.accept_eval --target "$TARGET" \
      --draft "$CK" --data "data/processed/${DOM}_target.jsonl" \
      --cache "out/target_cache/${DOM}_k256.npz" --topk 256 \
      --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map auto \
      --out "$OUTJ" 2>&1 | tee "out/logs/eq40m_${TAG}_${DOM}.log"
  done
done
