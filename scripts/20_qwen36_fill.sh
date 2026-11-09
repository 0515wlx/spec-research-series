#!/usr/bin/env bash
# Fill missing Qwen3.6 replication cells: GSM8K, HumanEval, MT-Bench, and generated OOD.
# GPU budget: at most 4 A100s. Set CUDA_VISIBLE_DEVICES=0,1,2,3 before running.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TARGET=${TARGET:-qwen3.6-35b-a3b}
TARGET_DM=${TARGET_DM:-auto}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"gsm8k humaneval mtbench wikitext_gen fineweb_gen"}
TOPK=${TOPK:-256}
mkdir -p data/processed out/target_cache out/accept/repl out/logs

for DOM in $DOMAINS; do
  case "$DOM" in
    gsm8k) LIM=500; NEW=512; SRC=data/processed/gsm8k.jsonl; TAG=gsm8k ;;
    humaneval) LIM=164; NEW=512; SRC=data/processed/humaneval.jsonl; TAG=humaneval ;;
    mtbench) LIM=80; NEW=512; SRC=data/processed/mtbench.jsonl; TAG=mtbench ;;
    wikitext_gen) LIM=100; NEW=512; SRC=""; TAG=wikitext_gen ;;
    fineweb_gen) LIM=100; NEW=512; SRC=data/processed/fineweb.jsonl; TAG=fineweb_gen ;;
    *) echo "unknown domain $DOM"; continue ;;
  esac
  DATA="data/processed/qwen3.6_${TAG}_target.jsonl"
  CACHE="out/target_cache/repl_${TAG}_k${TOPK}.npz"
  if [ ! -f "$DATA" ]; then
    if [ "$DOM" = wikitext_gen ]; then
      python -m infra.generate_target --target "$TARGET" --domain wikitext \
        --limit "$LIM" --max-new "$NEW" --seq-len 1024 --generate-wikitext \
        --batch-size 2 --device-map "$TARGET_DM" --out "$DATA"
    else
      python -m infra.generate_target --target "$TARGET" --domain "${DOM%_gen}" \
        --data "$SRC" --limit "$LIM" --max-new "$NEW" --batch-size 4 \
        --device-map "$TARGET_DM" --out "$DATA"
    fi
  fi
  if [ ! -f "$CACHE" ]; then
    python -m infra.cache_target --target "$TARGET" --data "$DATA" \
      --topk "$TOPK" --batch-size 4 --device-map "$TARGET_DM" --out "$CACHE"
  fi
  for D in $DRAFTS; do
    OUT="out/accept/repl/repl_${TAG}_${D}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    case "$D" in
      *0.8b|*2b) DBS=8; DDM=cuda:0 ;;
      *4b|*9b)   DBS=4; DDM=cuda:0 ;;
      *27b)      DBS=1; DDM=auto ;;
      *)         DBS=2; DDM=auto ;;
    esac
    python -m infra.accept_eval --target "$TARGET" --draft "$D" --data "$DATA" \
      --cache "$CACHE" --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map "$DDM" --out "$OUT"
  done
done
