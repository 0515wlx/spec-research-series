#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
DM=${DM:-balanced}
BS=${BS:-8}
mkdir -p data/processed out/logs

# in-domain / code / chat
for spec in "math500 500 512" "gsm8k 500 512" "humaneval 164 512" "mtbench 80 512"; do
  set -- $spec
  DOM=$1; LIM=$2; NEW=$3
  python -m infra.generate_target --target "$TARGET" --domain "$DOM" \
    --limit "$LIM" --max-new "$NEW" --batch-size "$BS" --device-map "$DM" \
    --out "data/processed/${DOM}_target.jsonl"
done

# OOD prose: raw continuation, no generation
python -m infra.generate_target --target "$TARGET" --domain wikitext \
  --limit 100 --seq-len 1024 --out data/processed/wikitext_target.jsonl
