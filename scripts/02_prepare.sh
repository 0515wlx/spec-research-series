#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m infra.prepare_data --domains math500 gsm8k humaneval wikitext mtbench --out data/processed
