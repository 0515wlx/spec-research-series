# Pilot Validation (2026-11-02)

The full Route A pipeline was smoke-tested on the already-present Qwen3.5-4B / 9B checkpoints,
before the 35B target and the smaller drafts finish downloading.

## What was run

```bash
python -m infra.env_check --json out/env.json
python -m infra.generate_target --target qwen3.5-9b --domain math500 --limit 5 \
  --max-new 64 --batch-size 2 --device-map cuda:0 \
  --out data/processed/pilot_math_target.jsonl
python -m infra.cache_target --target qwen3.5-9b \
  --data data/processed/pilot_math_target.jsonl --topk 256 --batch-size 2 \
  --device-map cuda:0 --out out/target_cache/pilot_math_k256.npz
python -m infra.accept_eval --target qwen3.5-9b --draft qwen3.5-4b \
  --data data/processed/pilot_math_target.jsonl \
  --cache out/target_cache/pilot_math_k256.npz \
  --gammas 1,2,4,8,16,32 --batch-size 2 --device-map cuda:0 \
  --out out/accept/pilot_math_qwen3.5-4b.json
python -m infra.spec_loop --target qwen3.5-9b --draft qwen3.5-4b \
  --domain math500 --limit 2 --gamma 4 --max-new 32 \
  --device-map-draft cuda:1 --device-map-target cuda:0 \
  --out out/loop/pilot_math_4b_g4.json
python -m infra.eval_lm --model qwen3.5-4b --tasks wikitext --batch-size 1 --limit 5 \
  --out out/lm_eval/pilot_qwen3.5-4b
python -m infra.train_distill --student qwen3.5-4b \
  --data data/processed/pilot_math_target.jsonl --out ckpt/pilot_train_4b \
  --max-tokens 512 --batch-tokens 256 --seq-len 128 --grad-checkpoint \
  --save-every-tokens 100000000 --val-every-tokens 100000000
```

## Results

| item | result |
|---|---|
| text-only loader | `Qwen3.5-4B` loads as `Qwen3_5ForCausalLM` (426 text tensors, no vision) |
| target generation | 5 MATH prompts, 64 tokens each, greedy, ~9 s on one A100 |
| target cache | top-256 log-probs, `[5, 64, 256]`, residual mass included |
| acceptance (4B draft vs 9B target) | `q=0.050`, `E[TV]=0.0465`, `q_h=0.0625`, `tau_inf=20.0` |
| capacity sanity (10 MATH prompts, 128 tok) | 0.8B: `q=0.094`, `E[TV]=0.114`; 2B: `q=0.0695`, `E[TV]=0.0816`; 4B: `q=0.050`, `E[TV]=0.046` — larger draft has lower q, as expected |
| 3-point fit (in-domain only) | floor model is underdetermined (`q_inf≈0`); log-linear wins by BIC. **Not** evidence against H2: 3 in-domain points cannot identify a floor. The real test needs 27B + OOD domains. |
| real loop (γ=4) | `α=0.920`, `τ=4.68` over 2 prompts / 15 blocks |
| lm-eval | WikiText loads Qwen3.5 directly; word perplexity 10.13 on 5 samples (pilot only) |
| trainer | 4B CE step runs; loss decreases; checkpoint save works (39 s incl. load/save) |
| scaling fit | synthetic 5-point floor/power/loglinear fits + bootstrap work |

**Note:** the pilot's `q=0.05` is *not* a Route A result — 4B and 9B are close in capability and the
sample is 5 prompts. The real scan uses the 0.8B–27B series against the 35B target on 100k+ tokens
per domain.

## Bugs found and fixed during the pilot

- `metrics_topk` JS residual term had a shape mismatch (`p_k` vs `p0`); fixed.
- `cache_target` was refactored to expose `build_cache()` so `accept_eval` can reuse it.
- `spec_loop` was doing a redundant target forward for the bonus token; fixed to reuse `o2`.
- `train_distill` warned about `float(loss)` on a tensor with grad; harmless for logging.

## Next

1. Finish Qwen3.5-0.8B / 2B / 27B / 35B-A3B downloads (HF mirror + `HF_HUB_DISABLE_XET=1`).
2. Run `scripts/03_target.sh` for 5 domains with the 35B target.
3. Run `scripts/04_capacity_scan.sh` (5 drafts × 5 domains).
4. Run `scripts/05_fit.sh` and inspect `out/fit/capacity/capacity_wall.png`.
5. Decide whether the training arm is needed (see `docs/ROUTE_A_PLAN.md` §10).
