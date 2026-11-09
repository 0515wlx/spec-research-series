#!/usr/bin/env bash
# Positive-control lr x token grid (2B, seeds 0/1).
#
# Separates the learning-rate and token-budget axes that the original gentle
# vs aggressive contrast confounded.  Existing 5e-6 / 0.5M seeds 0/1/2 are
# reused; this script adds the other cells.
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT=${STUDENT:-qwen3.5-2b}
SEEDS=${SEEDS:-"0 1"}
LRS=${LRS:-"5e-6 1e-5 2e-5"}
TOKS=${TOKS:-"500000 1000000"}
DOMAINS=${DOMAINS:-"fineweb fineweb_gen wikitext wikitext_gen math500"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/positive_grid out/logs

TRAIN=data/processed/fineweb_adapt_train.jsonl
VAL=data/processed/fineweb_adapt_val.jsonl
[ -f "$TRAIN" ] || { echo "missing $TRAIN"; exit 1; }

tasks=()
for LR in $LRS; do
  for T in $TOKS; do
    for S in $SEEDS; do
      OUT="ckpt/distill/${STUDENT}-fineweb-adapt${T}-lr${LR}-s${S}"
      # Skip the already-run 5e-6 / 0.5M seeds (3 seeds exist).
      if [ "$LR" = "5e-6" ] && [ "$T" = "500000" ]; then continue; fi
      tasks+=("$LR|$T|$S|$OUT")
    done
  done
done

# Four concurrent training jobs, one per GPU.
i=0
pids=()
for task in "${tasks[@]}"; do
  IFS='|' read -r LR T S OUT <<< "$task"
  if ! find "$OUT" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | grep -q .; then
    G=$((i % 4))
    echo "== train lr=$LR tokens=$T seed=$S GPU=$G =="
    CUDA_VISIBLE_DEVICES="$G" torchrun --nproc_per_node 1 -m infra.train_distill \
      --student "$STUDENT" --data "$TRAIN" --val "$VAL" --out "$OUT" \
      --max-tokens "$T" --seq-len 1024 --batch-tokens 2048 \
      --lr "$LR" --warmup 50 --objective ce --grad-checkpoint \
      --save-every-tokens "$T" --save-checkpoint-dirs \
      --seed "$S" > "out/logs/positive_grid_lr${LR}_t${T}_s${S}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then
      for p in "${pids[@]}"; do wait "$p"; done
      pids=()
    fi
  fi
done
for p in "${pids[@]:-}"; do wait "$p"; done

# Evaluate every (lr, token, seed) checkpoint on the five common sets.
i=0
pids=()
for LR in $LRS; do
  for T in $TOKS; do
    for S in $SEEDS; do
      if [ "$LR" = "5e-6" ] && [ "$T" = "500000" ]; then continue; fi
      OUT="ckpt/distill/${STUDENT}-fineweb-adapt${T}-lr${LR}-s${S}"
      CK="$(find "$OUT" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1)"
      [ -n "${CK:-}" ] && [ -d "$CK" ] || { echo "missing $OUT"; continue; }
      for DOM in $DOMAINS; do
        OUTJ="out/accept/positive_grid/${STUDENT}-adapt${T}-lr${LR}-s${S}_${DOM}.json"
        [ -f "$OUTJ" ] && continue
        G=$((i % 4))
        echo "== eval lr=$LR tokens=$T seed=$S $DOM GPU=$G =="
        CUDA_VISIBLE_DEVICES="$G" python -m infra.accept_eval --target qwen3.5-35b-a3b \
          --draft "$CK" --data "data/processed/${DOM}_target.jsonl" \
          --cache "out/target_cache/${DOM}_k256.npz" --topk 256 \
          --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map cuda:0 \
          --out "$OUTJ" > "out/logs/positive_grid_${LR}_${T}_${S}_${DOM}.log" 2>&1 &
        pids+=($!)
        i=$((i + 1))
        if [ $((i % 4)) -eq 0 ]; then
          for p in "${pids[@]}"; do wait "$p"; done
          pids=()
        fi
      done
    done
  done
done
for p in "${pids[@]:-}"; do wait "$p"; done
echo "done"
