#!/usr/bin/env bash
# Raw-text breadth sweep: multilingual Wikipedia, Chinese web text, code,
# books, and scientific prose.  Uses at most two GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
MODEL=${MODEL:-qwen3.5-9b}
FRACS=${FRACS:-0.5,0.7,0.9,0.95,0.98,0.99}
mkdir -p data/processed out/target_cache out/rank_scan out/logs

for c in wikipedia_zh wikipedia_de wikipedia_fr wikipedia_ja fineweb2_zh code_raw pg19 arxiv; do
  if [ ! -f "data/processed/${c}_target.jsonl" ]; then
    python -m infra.prepare_hf_corpus --corpus "$c" --n-chunks 100 --seq-len 1024
  fi
done

# Cache the 9B target in parallel: two corpora per GPU.
declare -a A=(wikipedia_zh wikipedia_de wikipedia_fr wikipedia_ja fineweb2_zh code_raw pg19 arxiv)
for i in 0 1 2 3; do
  c1=${A[$((2*i))]}; c2=${A[$((2*i+1))]}
  (
    export CUDA_VISIBLE_DEVICES=$i
    for c in $c1 $c2; do
      if [ ! -f "out/target_cache/${c}_target9b_k256.npz" ]; then
        python -m infra.cache_target --target "$MODEL" --data "data/processed/${c}_target.jsonl" \
          --topk 256 --batch-size 4 --device-map auto \
          --out "out/target_cache/${c}_target9b_k256.npz"
      fi
    done
  ) > "out/logs/cache_${i}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES=0 python -m infra.joint_scan --model "$MODEL" --mode width \
  --importance out/importance/qwen3.5-9b_act.pt --fracs "$FRACS" --batch-size 4 \
  --sets wikipedia_zh:data/processed/wikipedia_zh_target.jsonl:out/target_cache/wikipedia_zh_target9b_k256.npz \
         wikipedia_de:data/processed/wikipedia_de_target.jsonl:out/target_cache/wikipedia_de_target9b_k256.npz \
         wikipedia_fr:data/processed/wikipedia_fr_target.jsonl:out/target_cache/wikipedia_fr_target9b_k256.npz \
         wikipedia_ja:data/processed/wikipedia_ja_target.jsonl:out/target_cache/wikipedia_ja_target9b_k256.npz \
  --out out/rank_scan/qwen3.5-9b_actwidth_multilingual > out/logs/raw_suite_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m infra.joint_scan --model "$MODEL" --mode width \
  --importance out/importance/qwen3.5-9b_act.pt --fracs "$FRACS" --batch-size 4 \
  --sets fineweb2_zh:data/processed/fineweb2_zh_target.jsonl:out/target_cache/fineweb2_zh_target9b_k256.npz \
         code_raw:data/processed/code_raw_target.jsonl:out/target_cache/code_raw_target9b_k256.npz \
         pg19:data/processed/pg19_target.jsonl:out/target_cache/pg19_target9b_k256.npz \
         arxiv:data/processed/arxiv_target.jsonl:out/target_cache/arxiv_target9b_k256.npz \
  --out out/rank_scan/qwen3.5-9b_actwidth_raw2 > out/logs/raw_suite_raw2.log 2>&1 &
wait

python -m infra.compression_certificate --scan out/rank_scan/qwen3.5-9b_actwidth_multilingual \
  --gamma 4 --mlp-share 0.6088 --out out/rank_scan/certificate_multilingual_9b.json
python -m infra.compression_certificate --scan out/rank_scan/qwen3.5-9b_actwidth_raw2 \
  --gamma 4 --mlp-share 0.6088 --out out/rank_scan/certificate_raw2_9b.json
