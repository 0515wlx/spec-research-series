#!/usr/bin/env bash
# Larger held-out evaluation for the 2B positive control.
# WikiText: 500 target-generated continuations; FineWeb: 200 held-out prompts
# (0-99 eval + 1900-1999 val from the 2000-prompt split).
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p data/processed out/target_cache out/accept/positive_more out/logs

WIKI=data/processed/wikitext_gen500_target.jsonl
if [ ! -f "$WIKI" ]; then
  echo "== generate 500 WikiText continuations =="
  CUDA_VISIBLE_DEVICES=2,3 python -m infra.generate_target --target "$TARGET" \
    --domain wikitext --generate-wikitext --limit 500 --max-new 512 \
    --batch-size 4 --device-map auto --out "$WIKI" \
    2>&1 | tee out/logs/positive_more_wikitext_gen.log
fi

FINE=data/processed/fineweb_heldout200_target.jsonl
if [ ! -f "$FINE" ]; then
  python - <<'PY'
import json
rows=[json.loads(l) for l in open('data/processed/fineweb2000_target.jsonl')]
keep=[r for r in rows if r['id'] in {f'fineweb-{i}' for i in list(range(100))+list(range(1900,2000))}]
with open('data/processed/fineweb_heldout200_target.jsonl','w') as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False)+'\n')
print('fineweb heldout', len(keep))
PY
fi

for DOM in wikitext_gen500 fineweb_heldout200; do
  case "$DOM" in
    wikitext_gen500) DATA=$WIKI ;;
    fineweb_heldout200) DATA=$FINE ;;
  esac
  CACHE="out/target_cache/${DOM}_k${TOPK}.npz"
  [ -f "$CACHE" ] || {
    echo "== cache $DOM =="
    CUDA_VISIBLE_DEVICES=2,3 python -m infra.cache_target --target "$TARGET" \
      --data "$DATA" --topk "$TOPK" --batch-size 4 --device-map auto --out "$CACHE" \
      2>&1 | tee "out/logs/positive_more_cache_${DOM}.log"
  }
done

# Evaluate base 2B and the three gentle seeds.
for S in base 0 1 2; do
  if [ "$S" = "base" ]; then
    CK=qwen3.5-2b; TAG=base
  else
    CK=$(find "ckpt/distill/qwen3.5-2b-fineweb-adapt500000-lr5e-6-s${S}" -maxdepth 1 -type d -name 'tokens-*' 2>/dev/null | sort | tail -1)
    TAG=s${S}
    [ -n "$CK" ] && [ -d "$CK" ] || { echo "missing seed $S"; continue; }
  fi
  for DOM in wikitext_gen500 fineweb_heldout200; do
    DATA="data/processed/${DOM}_target.jsonl"
    [ "$DOM" = "wikitext_gen500" ] && DATA=$WIKI
    [ "$DOM" = "fineweb_heldout200" ] && DATA=$FINE
    OUTJ="out/accept/positive_more/${TAG}_${DOM}.json"
    [ -f "$OUTJ" ] && { echo "skip $OUTJ"; continue; }
    echo "== eval $TAG / $DOM =="
    CUDA_VISIBLE_DEVICES=1 python -m infra.accept_eval --target "$TARGET" \
      --draft "$CK" --data "$DATA" --cache "out/target_cache/${DOM}_k${TOPK}.npz" --topk "$TOPK" \
      --gammas 1,2,4,8,16,32,64 --batch-size 4 --device-map cuda:0 \
      --out "$OUTJ" 2>&1 | tee "out/logs/positive_more_${TAG}_${DOM}.log"
  done
done
