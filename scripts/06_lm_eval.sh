#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=${TASKS:-"gsm8k,minerva_math,humaneval,wikitext"}
MODELS=${MODELS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
for M in $MODELS; do
  python -m infra.eval_lm --model "$M" --tasks "$TASKS" \
    --batch-size 8 --apply-chat-template --out "out/lm_eval/$M"
done
