#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Retries ModelScope; safe to rerun. 4B/9B are already present.
python -m infra.download --models qwen3.5-0.8b qwen3.5-2b qwen3.5-27b qwen3.5-35b-a3b \
  --retry 10 --sleep 30 --max-workers 8
