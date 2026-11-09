#!/usr/bin/env bash
# Resume the 768-token MATH generation after a timeout / partial shard.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-qwen3.5-35b-a3b}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
MAXNEW=${MAXNEW:-768}
TOPK=${TOPK:-256}
OUTDATA=${OUTDATA:-data/processed/math500_target_768_full.jsonl}
CACHE=${CACHE:-out/target_cache/math500_768_full_k${TOPK}.npz}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/accept/math768 out/logs data/processed/shards

python - <<'PY'
import json, os
rows = [json.loads(l) for l in open("data/processed/math500.jsonl")]
want = [r["id"] for r in rows]
have = {}
for f in ["data/processed/shards/math500_768_out0.jsonl",
          "data/processed/shards/math500_768_out1.jsonl"]:
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            have[r["id"]] = r
missing = [r for r in rows if r["id"] not in have]
print("have", len(have), "missing", len(missing))
if missing:
    mid = len(missing) // 2
    for i, part in enumerate([missing[:mid], missing[mid:]]):
        with open(f"data/processed/shards/math500_768_resume_s{i}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("resume shards", mid, len(missing) - mid)
PY

if [ -s data/processed/shards/math500_768_resume_s0.jsonl ]; then
  CUDA_VISIBLE_DEVICES=0,1 python -m infra.generate_target --target "$TARGET" \
    --domain math500 --data data/processed/shards/math500_768_resume_s0.jsonl \
    --max-new "$MAXNEW" --batch-size 4 --max-prompt-tokens 1536 --device-map auto \
    --out data/processed/shards/math500_768_resume_out0.jsonl 2>&1 | tee out/logs/math768_resume0.log &
  P0=$!
  CUDA_VISIBLE_DEVICES=2,3 python -m infra.generate_target --target "$TARGET" \
    --domain math500 --data data/processed/shards/math500_768_resume_s1.jsonl \
    --max-new "$MAXNEW" --batch-size 4 --max-prompt-tokens 1536 --device-map auto \
    --out data/processed/shards/math500_768_resume_out1.jsonl 2>&1 | tee out/logs/math768_resume1.log &
  P1=$!
  wait $P0 $P1
fi

python - <<'PY'
import json, os
rows = [json.loads(l) for l in open("data/processed/math500.jsonl")]
want = [r["id"] for r in rows]
have = {}
for f in ["data/processed/shards/math500_768_out0.jsonl",
          "data/processed/shards/math500_768_out1.jsonl",
          "data/processed/shards/math500_768_resume_out0.jsonl",
          "data/processed/shards/math500_768_resume_out1.jsonl"]:
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            have[r["id"]] = r
missing = [i for i in want if i not in have]
print("total have", len(have), "missing", len(missing))
if missing:
    print("first missing", missing[:5])
    raise SystemExit(1)
with open("data/processed/math500_target_768_full.jsonl", "w") as f:
    for i in want:
        f.write(json.dumps(have[i], ensure_ascii=False) + "\n")
print("merged", len(want))
PY

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
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m infra.accept_eval --target "$TARGET" \
    --draft "$D" --data "$OUTDATA" --cache "$CACHE" --topk "$TOPK" \
    --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" --device-map auto \
    --out "$OUT" 2>&1 | tee "out/logs/math768_${D}.log"
done
