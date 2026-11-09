#!/usr/bin/env bash
# Compute-bound batch scan up to B=128 (prefill through the backbone).
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python -m infra.batch_latency --model qwen3.5-9b \
  --importance out/importance/qwen3.5-9b_act.pt --widths 0.5,0.98 \
  --batch-sizes 1,16,32,64,128 --prefix 512 --gamma 4 --repeats 8 \
  --out out/latency/batch_scan_9b_largeB.json

echo "done -> out/latency/batch_scan_9b_largeB.json"
