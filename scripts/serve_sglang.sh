#!/usr/bin/env bash
# Launch SGLang with speculative decoding.  Install a recent SGLang first.
set -euo pipefail
TARGET=${TARGET:-/path/to/models/Qwen3.5-35B-A3B}
TP=${TP:-2}
PORT=${PORT:-8000}
python -m sglang.launch_server --model-path "$TARGET" --port "$PORT" --tp-size "$TP" \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --mem-fraction-static 0.8
