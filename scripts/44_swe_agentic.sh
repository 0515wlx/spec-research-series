#!/usr/bin/env bash
# Agentic SWE-bench-Lite issue->patch prefix: build, cache, width scan, certificate.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
python -m infra.prepare_swe --n 100 --prefix-tokens 1024 --cont-tokens 512 \
  --out data/processed/swe_lite_target.jsonl \
  --prompts-out data/processed/swe_lite.jsonl

CUDA_VISIBLE_DEVICES=0 python -m infra.cache_target --target qwen3.5-9b \
  --data data/processed/swe_lite_target.jsonl --topk 256 --batch-size 2 \
  --out out/target_cache/swe_lite_target9b_k256.npz

CUDA_VISIBLE_DEVICES=0 python -m infra.joint_scan --model qwen3.5-9b --mode width \
  --importance out/importance/qwen3.5-9b_act.pt --fracs 0.5,0.9,0.98,0.99 \
  --sets swe:data/processed/swe_lite_target.jsonl:out/target_cache/swe_lite_target9b_k256.npz \
  --batch-size 2 --out out/rank_scan/qwen3.5-9b_actwidth_swe

python -m infra.compression_certificate --scan out/rank_scan/qwen3.5-9b_actwidth_swe \
  --gamma 4 --mlp-share 0.6088 --out out/rank_scan/certificate_swe_9b.json

echo "done -> out/rank_scan/certificate_swe_9b.json"
