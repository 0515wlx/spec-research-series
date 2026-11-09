# Positive control: deployment-distribution adaptation (2026-11-04)

The review asked for a positive baseline for the alignment/coverage axis. We use the held-out split of
`data/processed/fineweb2000_target.jsonl`: prompts 0--99 are the evaluation set (same ids as
`fineweb_target.jsonl` / `fineweb_gen_target.jsonl`); prompts 100--1899 are training; 1900--1999 are
validation. The training file is `data/processed/fineweb_adapt_train.jsonl` (1800 rows, 0.92M target
tokens).

## Runs

| run | data | tokens | lr | checkpoint |
|---|---|---:|---:|---|
| aggressive | FineWeb adapt train | 2M | 2e-5 | `ckpt/distill/qwen3.5-2b-fineweb-adapt2m/tokens-*` |
| gentle | FineWeb adapt train | 0.5M | 5e-6 | `ckpt/distill/qwen3.5-2b-fineweb-adapt500k-lr5e6/tokens-500736` |

The aggressive run is the script default; the gentle run was a follow-up because the aggressive run
overfit. Both are single-seed 2B full fine-tunes. The gentle run uses the same train/val split.

```bash
# aggressive 2M (script also evaluates all tokens-* checkpoints)
bash scripts/22_positive_control.sh

# gentle 500k
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 -m infra.train_distill \
  --student qwen3.5-2b --data data/processed/fineweb_adapt_train.jsonl \
  --val data/processed/fineweb_adapt_val.jsonl \
  --out ckpt/distill/qwen3.5-2b-fineweb-adapt500k-lr5e6 \
  --max-tokens 500000 --seq-len 1024 --batch-tokens 2048 --lr 5e-6 --warmup 50 \
  --objective ce --grad-checkpoint --save-every-tokens 500000 --save-checkpoint-dirs

# evaluate one checkpoint on one domain (repeat for fineweb, fineweb_gen,
# wikitext, wikitext_gen, math500)
CUDA_VISIBLE_DEVICES=2 python -m infra.accept_eval --target qwen3.5-35b-a3b \
  --draft ckpt/distill/qwen3.5-2b-fineweb-adapt500k-lr5e6/tokens-500736 \
  --data data/processed/fineweb_gen_target.jsonl \
  --cache out/target_cache/fineweb_gen_k256.npz --topk 256 \
  --gammas 1,2,4,8,16,32,64 --batch-size 8 --device-map cuda:0 \
  --out out/accept/positive/qwen3.5-2b-fineweb-adapt500k-lr5e6_tokens-500736_fineweb_gen.json

python -m infra.positive_control_stats
python paper/make_positive_table.py
```

## Results

See `out/accept/positive/summary.md` and `paper/tab_positive.tex`.

- Gentle 500k, lr 5e-6: FineWeb raw 0.2769 -> 0.2718, FineWeb generated 0.0750 -> 0.0733, WikiText raw
  0.3174 -> 0.3008, WikiText generated 0.1348 -> 0.1297, MATH 0.0728 -> 0.0731.
- Aggressive 2M, lr 2e-5: every set worsens; FineWeb raw rises to 0.3131, WikiText raw to 0.3533,
  MATH to 0.0830.
- The gentle run improves OOD point estimates by 1.8--5.2% without in-domain damage. Only the raw
  WikiText change is individually significant (delta -0.0166, 95% CI [-0.0272, -0.0055]); the other
  changes are small point improvements. The effect is real but small, and the aggressive run shows that
  more target data is not automatically better.

## Caveats

- Single seed, 2B only; the contrast confounds learning rate and token budget.
- The training corpus is target-generated FineWeb, so the FineWeb evaluation is the matched set;
  WikiText is a transfer set. The gentle run improving WikiText too is encouraging but should be
  replicated.
- The first 100 FineWeb prompts are held out, but the train/val/test split is at the prompt level, not
  the document level. FineWeb prompts are distinct documents, so this is a reasonable split.
