#!/usr/bin/env bash
# Launch vLLM with speculative decoding.  Install a recent vLLM first (Qwen3.5 needs main).
set -euo pipefail
TARGET=${TARGET:-/path/to/models/Qwen3.5-35B-A3B}
DRAFT=${DRAFT:-}
TP=${TP:-2}
PORT=${PORT:-8000}
EXTRA=${EXTRA:-}

if [ -z "$DRAFT" ]; then
  # native MTP / NEXTN head
  SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
else
  SPEC=$(printf '{"model":"%s","num_speculative_tokens":5}' "$DRAFT")
fi

vllm serve "$TARGET" --port "$PORT" --tensor-parallel-size "$TP" \
  --max-model-len 32768 --speculative-config "$SPEC" $EXTRA
