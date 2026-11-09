#!/usr/bin/env bash
# Run the equal-token series with the stable ZeRO-3 trainer.
# Order: 0.8B -> eval -> 4B -> eval -> 27B target-only -> eval -> 9B -> eval.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== train 0.8B @10M =="
bash scripts/28_equal_token_series.sh 0.8b
bash scripts/29_eq10m_eval_all.sh

echo "== train 4B @10M =="
bash scripts/28_equal_token_series.sh 4b
bash scripts/29_eq10m_eval_all.sh

echo "== train 27B target-only @10M =="
bash scripts/28_equal_token_series.sh 27b
bash scripts/29_eq10m_eval_all.sh

echo "== train 9B @10M =="
bash scripts/28_equal_token_series.sh 9b
bash scripts/29_eq10m_eval_all.sh

echo "== all equal-token runs finished =="
