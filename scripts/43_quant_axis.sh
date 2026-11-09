#!/usr/bin/env bash
# Real packed INT4/INT8 weight-only axis: quality + batch-1 cost.
# Uses <= 4 GPUs; the four quality runs are independent.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p out/quant
MODEL=${MODEL:-qwen3.5-9b}
IMPORTANCE=${IMPORTANCE:-out/importance/qwen3.5-9b_act.pt}

# W4 quality (LM head bf16), one corpus per GPU.
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --skip-lm-head --data data/processed/wikitext_target.jsonl \
  --cache out/target_cache/wikitext_target9b_k256.npz --batch-size 2 \
  --out out/quant/wikitext_w4_skiplm.json &
CUDA_VISIBLE_DEVICES=1 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --skip-lm-head --data data/processed/fineweb_target.jsonl \
  --cache out/target_cache/fineweb_target9b_k256.npz --batch-size 2 \
  --out out/quant/fineweb_w4_skiplm.json &
CUDA_VISIBLE_DEVICES=2 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --skip-lm-head --data data/processed/c4_target.jsonl \
  --cache out/target_cache/c4_target9b_k256.npz --batch-size 2 \
  --out out/quant/c4_w4_skiplm.json &
CUDA_VISIBLE_DEVICES=3 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --skip-lm-head --data data/processed/math500_target_qwen3.5-9b.jsonl \
  --cache out/target_cache/math500_target9b_k256.npz --batch-size 2 \
  --out out/quant/math500_w4_skiplm.json &
wait

# W4 all-linears (LM head quantized) on two corpora.
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --data data/processed/wikitext_target.jsonl \
  --cache out/target_cache/wikitext_target9b_k256.npz --batch-size 2 \
  --out out/quant/wikitext_w4_alllinear.json &
CUDA_VISIBLE_DEVICES=1 python -m infra.quant_scan --model "$MODEL" --bits 4 \
  --data data/processed/fineweb_target.jsonl \
  --cache out/target_cache/fineweb_target9b_k256.npz --batch-size 2 \
  --out out/quant/fineweb_w4_alllinear.json &
wait

# W8 quality (30 sequences; the INT8 tinygemm is slow on A100).
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_scan --model "$MODEL" --bits 8 \
  --skip-lm-head --data data/processed/wikitext_target.jsonl \
  --cache out/target_cache/wikitext_target9b_k256.npz --limit 30 --batch-size 1 \
  --out out/quant/wikitext_w8_30.json

# Batch-1 cost.
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_latency --model "$MODEL" --bits 4 \
  --skip-lm-head --out out/quant/latency_w4_skiplm.json
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_latency --model "$MODEL" --bits 4 \
  --out out/quant/latency_w4_alllinear.json
CUDA_VISIBLE_DEVICES=0 python -m infra.quant_latency --model "$MODEL" --bits 8 \
  --skip-lm-head --out out/quant/latency_w8_skiplm.json

echo "done -> out/quant"
