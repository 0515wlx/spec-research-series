#!/usr/bin/env bash
# Full MATH-500 regeneration at 768 tokens, sharded over 2x2 GPUs.
#
# The original target set is capped at 512 tokens (386/500 hit the cap).  This
# script regenerates all 500 prompts with max_new=768, merges the shards, builds
# a top-256 target cache, and re-runs all five drafts.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
MAXNEW=${MAXNEW:-768}
TOPK=${TOPK:-256}
SRC=${SRC:-data/processed/math500.jsonl}
OUTDATA=${OUTDATA:-data/processed/math500_target_768_full.jsonl}
CACHE=${CACHE:-out/target_cache/math500_768_full_k${TOPK}.npz}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/math768 out/logs data/processed

if [ ! -f "$OUTDATA" ]; then
  python - <<'PY'
import json, os
src = "data/processed/math500.jsonl"
rows = [json.loads(l) for l in open(src)]
os.makedirs("data/processed/shards", exist_ok=True)
n = len(rows)
mid = n // 2
for i, part in enumerate([rows[:mid], rows[mid:]]):
    with open(f"data/processed/shards/math500_768_s{i}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("shards", mid, n - mid)
PY
  echo "== generate shard 0 on GPUs 0,1 =="
  CUDA_VISIBLE_DEVICES=0,1 python -m infra.generate_target --target "$TARGET" \
    --domain math500 --data data/processed/shards/math500_768_s0.jsonl \
    --max-new "$MAXNEW" --batch-size 4 --max-prompt-tokens 1536 --device-map auto \
    --out data/processed/shards/math500_768_out0.jsonl 2>&1 | tee out/logs/math768_gen0.log &
  P0=$!
  echo "== generate shard 1 on GPUs 2,3 =="
  CUDA_VISIBLE_DEVICES=2,3 python -m infra.generate_target --target "$TARGET" \
    --domain math500 --data data/processed/shards/math500_768_s1.jsonl \
    --max-new "$MAXNEW" --batch-size 4 --max-prompt-tokens 1536 --device-map auto \
    --out data/processed/shards/math500_768_out1.jsonl 2>&1 | tee out/logs/math768_gen1.log &
  P1=$!
  wait $P0 $P1
  python - <<'PY'
import json
outs = []
for i in range(2):
    outs += [json.loads(l) for l in open(f"data/processed/shards/math500_768_out{i}.jsonl")]
with open("data/processed/math500_target_768_full.jsonl", "w") as f:
    for r in outs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("merged", len(outs))
PY
fi

if [ ! -f "$CACHE" ]; then
  echo "== cache top-$TOPK =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.cache_target --target "$TARGET" \
    --data "$OUTDATA" --topk "$TOPK" --batch-size 4 --device-map auto \
    --out "$CACHE" 2>&1 | tee out/logs/math768_cache.log
fi

for D in $DRAFTS; do
  OUT="out/accept/math768/${D}.json"
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  case "$D" in
    *0.8b|*2b) DBS=8 ;;
    *4b|*9b)   DBS=4 ;;
    *27b)      DBS=1 ;;
    *)         DBS=2 ;;
  esac
  echo "== eval $D =="
  CUDA_VISIBLE_DEVICES=0,1 python -m infra.accept_eval --target "$TARGET" \
    --draft "$D" --data "$OUTDATA" --cache "$CACHE" --topk "$TOPK" \
    --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" --device-map auto \
    --out "$OUT" 2>&1 | tee "out/logs/math768_${D}.log"
done
