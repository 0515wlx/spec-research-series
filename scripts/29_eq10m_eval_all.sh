#!/usr/bin/env bash
# Evaluate all equal-token (10M) checkpoints on the five common sets.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
DOMAINS=${DOMAINS:-"math500 wikitext wikitext_gen fineweb fineweb_gen"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/eq10m out/logs

declare -A CK
CK[0.8b]="$(find ckpt/zero3/qwen3.5-0.8b-eq10m -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1 || true)"
CK[2b]=ckpt/distill/qwen3.5-2b-ctrl10m
CK[4b]="$(find ckpt/zero3/qwen3.5-4b-eq10m -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1 || true)"
CK[9b]="$(find ckpt/zero3/qwen3.5-9b-eq10m -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1 || true)"
CK[27b]="$(find ckpt/zero3/qwen3.5-27b-eq10m-target -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1 || true)"

for KEY in 0.8b 2b 4b 9b 27b; do
  D=${CK[$KEY]:-}
  [ -n "$D" ] && [ -d "$D" ] || { echo "missing checkpoint for $KEY"; continue; }
  for DOM in $DOMAINS; do
    OUTJ="out/accept/eq10m/${KEY}_${DOM}.json"
    [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
    echo "== eval $KEY / $DOM =="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.accept_eval --target "$TARGET" \
      --draft "$D" --data "data/processed/${DOM}_target.jsonl" \
      --cache "out/target_cache/${DOM}_k256.npz" --topk 256 \
      --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map auto \
      --out "$OUTJ" 2>&1 | tee "out/logs/eq10m_${KEY}_${DOM}.log"
  done
done
