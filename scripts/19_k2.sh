#!/usr/bin/env bash
# K2 Horizon cross-family replication.
# Target: K2-Horizon-MoVA-36B-A4B (36B total / 4B active)
# Drafts: K2-Horizon-3.7B / 7B / 32B (same family/tokenizer; 0.9B has a smaller vocab)
# GPU budget: at most 4 A100s. Set CUDA_VISIBLE_DEVICES=0,1,2,3 before running.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TARGET=${TARGET:-k2-horizon-mova-36b-a4b}
TARGET_DM=${TARGET_DM:-auto}
DRAFTS=${DRAFTS:-"k2-horizon-3.7b k2-horizon-7b k2-horizon-32b"}
DOMAINS=${DOMAINS:-"math500 wikitext fineweb"}
TOPK=${TOPK:-256}
mkdir -p data/processed out/target_cache out/accept/k2 out/logs

gen() {
  local dom=$1 lim=$2 new=$3 data=$4 out=$5
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  if [ "$dom" = wikitext ]; then
    python -m infra.generate_target --target "$TARGET" --domain wikitext \
      --limit "$lim" --seq-len 1024 --device-map "$TARGET_DM" --out "$out"
  else
    python -m infra.generate_target --target "$TARGET" --domain "$dom" \
      --data "$data" --limit "$lim" --max-new "$new" --batch-size 2 \
      --device-map "$TARGET_DM" --out "$out"
  fi
}

for DOM in $DOMAINS; do
  case "$DOM" in
    math500) LIM=50; NEW=128; SRC=data/processed/math500.jsonl ;;
    wikitext) LIM=50; NEW=0; SRC="" ;;
    fineweb) LIM=50; NEW=128; SRC=data/processed/fineweb.jsonl ;;
    gsm8k) LIM=200; NEW=512; SRC=data/processed/gsm8k.jsonl ;;
    humaneval) LIM=164; NEW=512; SRC=data/processed/humaneval.jsonl ;;
    mtbench) LIM=80; NEW=512; SRC=data/processed/mtbench.jsonl ;;
    *) echo "unknown domain $DOM"; continue ;;
  esac
  DATA="data/processed/k2_${DOM}_target.jsonl"
  CACHE="out/target_cache/k2_${DOM}_k${TOPK}.npz"
  gen "$DOM" "$LIM" "$NEW" "$SRC" "$DATA"
  if [ ! -f "$CACHE" ]; then
    python -m infra.cache_target --target "$TARGET" --data "$DATA" \
      --topk "$TOPK" --batch-size 2 --device-map "$TARGET_DM" --out "$CACHE"
  fi
  for D in $DRAFTS; do
    OUT="out/accept/k2/${DOM}_${D}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    case "$D" in
      *3.7b) DBS=8; DDM=cuda:0 ;;
      *7b)   DBS=4; DDM=cuda:0 ;;
      *32b)  DBS=1; DDM=auto ;;
      *)     DBS=2; DDM=auto ;;
    esac
    python -m infra.accept_eval --target "$TARGET" --draft "$D" --data "$DATA" \
      --cache "$CACHE" --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map "$DDM" --out "$OUT"
  done
done
