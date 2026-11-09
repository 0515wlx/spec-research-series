# Experiment Plan (Route A)

> This is the operational checklist. It is intentionally self-contained so it can be executed
> without re-reading the chat context. Commands assume the repo root:
> `/path/to/spec-research`.
>
> Legend: **P0** = blocking, **P1** = core paper, **P2** = controls/ablations, **P3** = optional.
> GPU-hours are rough A100-80G estimates; adjust after the first run.

## Master table

| ID | Experiment | Models | Data | Key output | Priority | Est. GPU-h |
|---|---|---|---|---|---|---|
| E0 | Env / model / data prep | all | all | `env.json`, model dirs, `data/processed` | P0 | 1 |
| E1 | Target generation | Qwen3.5-35B-A3B | 5 domains | `*_target.jsonl` | P0 | 3–6 |
| E2 | Capacity scan (teacher forcing) | 0.8/2/4/9/27B | 5 domains | `out/accept/*.json` + npz | P1 | 25–45 |
| E3 | Scaling fit | — | E2 | `out/fit/capacity/*` | P1 | <1 |
| E4 | Real speculative loop | 2B, 9B, 27B | math500, wikitext, humaneval | `out/loop/*.json` | P1 | 30–60 |
| E5 | Quantization control | 9B BF16/W8/W4; 35B W4 | math500, wikitext | `out/accept/quant_*` | P2 | 8–15 |
| E6 | lm-eval quality | target + all drafts | GSM8K/MATH/HumanEval/WikiText | `out/lm_eval/*` | P2 | 15–30 |
| E7 | Distillation arm | 2B/4B/9B | target-generated | `ckpt/distill_*`, q vs tokens | P3 | 40–100 |
| E8 | Replication family | Qwen3.6-35B-A3B target / Qwen3.8-27B | 2 domains | `out/accept/repl_*` | P2 | 15–25 |
| E9 | Ablations | — | E2/E4 | clustering, block-start, beta | P2 | 5–10 |

## E0 — Environment, models, data

**Goal:** reproducible environment and a clean data manifest.

```bash
python -m infra.env_check --json out/env.json
python -m infra.download --models qwen3.5-0.8b qwen3.5-2b qwen3.5-27b qwen3.5-35b-a3b
python -m infra.prepare_data --domains math500 gsm8k humaneval wikitext mtbench --out data/processed
```

**Checks:**
- 8×A100-80G visible; >= 1TB free on `/path/to`.
- `transformers>=5.15`, `lm_eval>=0.4.13`, `modelscope`, `scipy`, `matplotlib`.
- Each model loads as text-only (`infra.model_loader.load_text_model`).
- Tokenizer is identical across the Qwen3.5 series (assert `len(tokenizer) == 248320` and same vocab hash).

**Outputs:** `out/env.json`, `data/processed/manifest.json`.

**Failure modes:** ModelScope 500 / network; use `--retry 10 --sleep 30`; models can be copied from
another machine. Do not start E1 until the tokenizer assertion passes.

## E1 — Target generation

**Goal:** for each domain, generate the target model's greedy continuation. These sequences define the
deployment distribution `pi` and the prefixes on which every draft is scored.

```bash
# one domain at a time; use 2-4 GPUs with --device-map auto for the 35B MoE
python -m infra.generate_target \
  --target qwen3.5-35b-a3b --domain math500 --limit 500 \
  --max-new 512 --batch-size 8 --device-map auto \
  --out data/processed/math500_target.jsonl

python -m infra.generate_target --target qwen3.5-35b-a3b --domain gsm8k   --limit 500 ...
python -m infra.generate_target --target qwen3.5-35b-a3b --domain humaneval --limit 164 ...
python -m infra.generate_target --target qwen3.5-35b-a3b --domain mtbench  --limit 80  ...
# wikitext: no generation; chunk the raw test set into 1024-token continuations
python -m infra.generate_target --target qwen3.5-35b-a3b --domain wikitext --chunks 100 ...
```

**Record format** (JSONL, same as old `data/teacher_math500_nothink.jsonl`):
```json
{"id": "...", "domain": "math500", "prompt": "...", "input_ids": [...], "output_ids": [...],
 "text": "...", "meta": {"target": "qwen3.5-35b-a3b", "max_new": 512, "thinking": false}}
```

**Controls:**
- Greedy, temperature 0, same `max_new` for all drafts.
- Thinking: Qwen3.5 thinks by default. Use the template's no-think mode if available; otherwise
  prepend an empty ` thinking\n\n<｜end▁of▁thinking｜>` block and document it. **The same convention must be used
  for every draft.**
- For WikiText, raw continuation: tokenize text, split into 1024-token chunks, use first 512 as
  prefix and next 512 as the scored continuation. No chat template.
- Save the exact tokenizer revision and chat template hash in `meta`.

**Outputs:** `data/processed/{domain}_target.jsonl`, `data/processed/manifest.json`.

**Success:** >= 95% of prompts produce >= 32 output tokens; report truncation rate.

## E2 — Capacity scan (teacher forcing)

**Goal:** the core measurement. For each `(draft, domain)`, run the draft on the target-generated
prefixes and compute per-token acceptance + divergence metrics + block simulation.

```bash
for D in 0.8b 2b 4b 9b 27b; do
  for DOM in math500 gsm8k humaneval wikitext mtbench; do
    python -m infra.accept_eval \
      --target qwen3.5-35b-a3b --draft qwen3.5-$D \
      --data data/processed/${DOM}_target.jsonl \
      --gammas 1,2,4,8,16,32,64 --limit 500 \
      --device-map auto --out out/accept/${DOM}_qwen3.5-${D}.json
  done
done
```

**Per-token arrays saved to `.npz`:** `match`, `tv`, `kl_td`, `kl_dt`, `js`, `ent`, `margin`, `rho`,
`rank`, `p0`, `q0`, `target_ids`, `draft_pred`, `seq_id`.
**Summary JSON:** `q`, `E[TV]`, `q_h`, `alpha(gamma)`, `tau(gamma)`, `survival`, `P(C|C)`, `P(C|I)`,
`block_start_p`, token counts, timing.

**Notes:**
- Teacher forcing: the draft always sees the target's true prefix (the standard approximation;
  validate with E4).
- Full-vocab TV/KL: compute in float32 on GPU, chunk over positions; do not truncate unless memory
  forces it. If top-k is used, include the residual bucket `p0/q0` (as the old `metric_arrays` does).
- For the 27B draft, use `--device-map auto` across 2 GPUs; batch by length to avoid OOM.
- Deterministic: greedy; no sampling.

**Outputs:** `out/accept/{domain}_qwen3.5-{D}.json` + `.npz`.

**Success:** all 25 cells (5 drafts × 5 domains) with >= 100k scored tokens per OOD domain and >= 50k
per in-domain domain; `q` and `E[TV]` differ by < 5% (sanity).

## E2b — Generated-OOD WikiText (distribution-matched OOD)

**Goal:** the raw-text WikiText teacher forcing is harder than the real loop because the real loop
generates model text. Generate target continuations from the same raw prefixes, then rerun the
capacity scan on that distribution.

```bash
python -m infra.generate_target --target qwen3.5-35b-a3b --domain wikitext \
  --generate-wikitext --limit 100 --max-new 512 --batch-size 2 \
  --max-prompt-tokens 1024 --device-map cuda:7 \
  --out data/processed/wikitext_gen_target.jsonl

TARGET=qwen3.5-35b-a3b TARGET_DM=cuda:7 DRAFT_DM=cuda:4 bash scripts/11_generated_ood.sh
```

**Outputs:** `out/accept/wikitext_gen_qwen3.5-{size}.json`; the fit parser groups these under domain
`gen`, separate from raw-text `wikitext`.

**Success:** generated-OOD q is closer to the real-loop q than raw-text q; the floor ordering
(in-domain < generated-OOD < raw-text) is preserved.

## E2c — FineWeb-Edu as a second OOD corpus

**Goal:** test whether the floor is WikiText-specific. Stream FineWeb-Edu from ModelScope, build
raw 1024-token continuations, and rerun the 5-draft scan.

```bash
python -m infra.prepare_fineweb --mode ood --n-chunks 100 --seq-len 1024 \
  --out data/processed/fineweb_target.jsonl --prompts-out data/processed/fineweb.jsonl
# (the streaming loader may print a harmless GIL error at exit; the file is complete)
python -m infra.cache_target --target qwen3.5-35b-a3b \
  --data data/processed/fineweb_target.jsonl --topk 256 --device-map cuda:6 \
  --out out/target_cache/fineweb_k256.npz
# then run the 5 drafts as in E2, output out/accept/fineweb_qwen3.5-{size}.json
```

**Success:** the floor model wins BIC and `q_inf` is positive; the domain ordering
`in-domain < FineWeb < WikiText` is plausible.

## E3 — Scaling fit

**Goal:** fit and compare `q(D)` models; produce the paper's core figure.

```bash
python -m infra.scaling_fit --glob 'out/accept/*.json' --out out/fit/capacity
```

Models compared per domain:
1. `q = q_inf + A * D^(-alpha)` (floor model)
2. `q = A * D^(-alpha)` (pure power law)
3. `q = a + b * log10(D)` (log-linear, Scylla-style)
4. `q = a + b / log10(D)` (alternative saturating form)

Report: params, R², AIC, BIC, bootstrap 95% CI for `q_inf` (1000 resamples over sequences).
Plot `q` vs `D` (log x), `tau_inf` vs `D`, and `q_inf` per domain.

**Decision rule:** H2 supported iff the floor model wins AIC/BIC on >= 3 OOD domains and `q_inf` CI
excludes 0. Otherwise document the falsification and pivot (see `ROUTE_A_PLAN.md` §5).

## E4 — Real speculative loop

**Goal:** validate that teacher-forcing results are not an artifact; measure real `tau`, per-depth
acceptance, and wall-clock.

```bash
python -m infra.spec_loop \
  --target qwen3.5-35b-a3b --draft qwen3.5-2b \
  --domain math500 --limit 30 --gamma 8 --max-new 256 \
  --out out/loop/math500_qwen3.5-2b_g8.json
```

Run for:
- drafts: 2B, 9B, 27B (at least 2B and 27B)
- domains: math500, wikitext, humaneval
- gamma: 4, 8, 16, 32
- limit: 30–50 prompts each

**Metrics:** `alpha`, `tau`, `steps`, `tokens`, `draft_s`, `target_s`, `verify_s`, `wall_s`,
`p_k` per depth, `accepted_len` distribution.

**Controls:** same prompt set and `max_new`; greedy; target and draft on separate GPUs if possible;
record the draft/target cost ratio `c`.

**Success:** real-loop `tau(gamma)` within ~10% of teacher-forcing simulation for gamma <= 16; report
the gap for gamma = 32 (expected: real loop lower due to within-block drift).

## E5 — Quantization control

**Goal:** separate capacity from quantization. The old report's `3.4B -> 35B-W4` comparison is
confounded; this experiment fixes it.

```bash
# export text-only checkpoints first (needed for quant tools / vLLM)
python -m infra.export_text_model --model qwen3.5-9b --out ckpt/text/qwen3.5-9b

# BF16 baseline (no quant)
python -m infra.accept_eval --draft ckpt/text/qwen3.5-9b ...

# W8 / W4: use GPTQ/AWQ or bitsandbytes; keep the same calibration set and group size
# document the exact tool/version in the output JSON
```

Cells: `{9B-BF16, 9B-W8, 9B-W4} × {math500, wikitext}` and `{35B-W4, 35B-W8 (if feasible)}`.
Report `q`, `E[TV]`, and the additive quantization gap `q(W4) - q(BF16)` at matched capacity.

**Decision:** if the W4 gap is ~0.1–0.2 and the BF16 35B is near 0, the old "0.17 residual" is
quantization, not an OOD capacity floor; rewrite the claim accordingly.

**Fifth-round update (2026-11-07):** the compression-floor paper now uses real packed INT4/INT8
kernels (`infra/quant_kernel.py`, `infra/quant_scan.py`, `infra/quant_latency.py`), not fake
quantization. The W4/W8 draft of the *same* 9B target is a memory-only axis: it passes the memory
roofline but our kernels are not faster at batch 1. See the private lab experiment log and
`scripts/43_quant_axis.sh`.

## E6 — LM quality eval (lm-eval)

**Goal:** characterize target/draft capability and ensure the acceptance story is not a quality
artifact. Also provides the "generalist vs specialist" evidence.

```bash
python -m infra.eval_lm --model qwen3.5-9b \
  --tasks gsm8k,minerva_math,humaneval,wikitext \
  --batch-size 8 --apply-chat-template --out out/lm_eval/qwen3.5-9b.json
```

Run for target + 0.8B/2B/4B/9B/27B. Tasks:
- `gsm8k` (generation, exact match)
- `minerva_math` or `math` (if available in lm-eval)
- `humaneval` (pass@1)
- `wikitext` (word perplexity)
- optional: `mmlu`, `arc_challenge`, `hellaswag` for general capability.

**Note:** Qwen3.5 is multimodal; use the text-only export (`infra/export_text_model.py`) or the
custom HF wrapper. If `lm-eval` cannot load the architecture, export a text-only checkpoint and point
`--model_args pretrained=ckpt/text/...`.

## E7 — Distillation arm (optional, Phase 2)

**Goal:** test whether training shifts the floor. The old runs used only ~3M tokens and did not
converge; this arm uses 50M–200M tokens and tracks convergence.

```bash
# 1) build a distillation corpus from target outputs (same prompts + more)
python -m infra.generate_target --target qwen3.5-35b-a3b --domain numina \
  --pool data/train_pool.jsonl --limit 20000 --max-new 768 \
  --out data/processed/distill_target.jsonl

# 2) distill each draft size with the same token budget
torchrun --nproc_per_node 8 -m infra.train_distill \
  --student qwen3.5-2b --data data/processed/distill_target.jsonl \
  --max-tokens 100000000 --seq-len 2048 --lr 1e-4 --warmup 500 \
  --objective ce --save-every 5e7 --out ckpt/distill/qwen3.5-2b-100m
```

Run at 2B / 4B / 9B, token budgets 50M / 100M / 200M (at least 2 points per size). Evaluate with E2
and E3; plot `q` vs tokens and `q_inf` vs tokens.

**2D scaling law (H2b).** For the paper's strongest version, run a small grid:

| student | 50M | 100M | 200M |
|---|---|---|---|
| 2B | yes | yes | optional |
| 4B | yes | yes | yes |
| 9B (FSDP/LoRA) | optional | optional | optional |

Fit `q(D,T) = q_inf + A D^-alpha T^-beta` on the grid (and the off-the-shelf points as `T=pretrain`
proxies only if you are explicit about the confound). Test whether a shared `q_inf` across `(D,T)` fits
better than separate floors. This is the experiment that directly answers "is the old quick validation
just undertrained, or is there a real floor?"

Cost: 2B/4B grid ≈ 40–80 GPU-hours including data generation; 9B needs FSDP/LoRA and is optional.

**Convergence criteria:** val token accuracy and val expected acceptance length plateau for >= 10% of
the budget; otherwise extend. Log every 1M tokens to `out/train/*.jsonl`.

**Cost:** 9B × 100M tokens ≈ 12–20 GPU-hours on 8×A100 (plus data generation). Feasible, but not
blocking for the first paper.

## E7b — Coverage-controlled training with FineWeb-Edu

**Goal:** test whether adding general web text fixes the OOD degradation caused by narrow
target-data distillation. The 2B distilled only on math/chat data improves in-domain q 8x but
worsens OOD q 1.5x; mixing in FineWeb should recover OOD coverage.

```bash
# full 1024-token chunks, self-supervised (input_ids=[], output_ids=chunk)
python -m infra.prepare_fineweb --mode train --n-chunks 2000 --seq-len 1024 \
  --out data/processed/fineweb_train.jsonl
cat data/processed/distill_controlled.jsonl data/processed/fineweb_train.jsonl \
  > data/processed/distill_mixed.jsonl

CUDA_VISIBLE_DEVICES=1 python -m infra.train_distill \
  --student qwen3.5-2b --data data/processed/distill_mixed.jsonl \
  --out ckpt/distill/qwen3.5-2b-mixed10m --max-tokens 10000000 \
  --seq-len 1536 --batch-tokens 2048 --lr 5e-5 --warmup 100 --grad-checkpoint
```

**Compare:** narrow 2B@10M (`math500 q=0.00935`, `raw WikiText q=0.4668`,
`generated-OOD q=0.2019`) vs mixed 2B@10M.

**Success:** mixed training keeps the in-domain gain and improves OOD q. If it does, **coverage is
the lever**; if it does not, the OOD floor is capacity/target-alignment limited.

## E7c — Target-aligned FineWeb coverage

**Goal:** raw FineWeb text did **not** fix OOD (see `docs/RESULTS.md` §12). Test whether
target-aligned OOD data does: generate target continuations on FineWeb prompts, then distill.

```bash
python -m infra.prepare_fineweb --mode ood --n-chunks 2000 --seq-len 1024 \
  --out data/processed/fineweb2000_target.jsonl \
  --prompts-out data/processed/fineweb2000_prompts.jsonl

python -m infra.generate_target --target qwen3.5-35b-a3b --domain fineweb \
  --data data/processed/fineweb2000_prompts.jsonl --limit 2000 --max-new 512 \
  --batch-size 8 --device-map balanced \
  --out data/processed/fineweb2000_gen_target.jsonl   # ~1.02M target tokens

cat data/processed/distill_controlled.jsonl data/processed/fineweb2000_gen_target.jsonl \
  > data/processed/distill_target_aligned.jsonl      # ~4.15M tokens

CUDA_VISIBLE_DEVICES=0 python -m infra.train_distill \
  --student qwen3.5-2b --data data/processed/distill_target_aligned.jsonl \
  --out ckpt/distill/qwen3.5-2b-targetaligned10m --max-tokens 10000000 \
  --seq-len 1536 --batch-tokens 2048 --lr 5e-5 --warmup 100 --grad-checkpoint
```

**Compare:** narrow 2B@10M, raw-FineWeb-mixed 2B@10M, target-aligned 2B@10M, on math500 /
WikiText raw / generated-OOD / FineWeb.

**Success:** target-aligned mix improves OOD while keeping most of the in-domain gain. If it does,
the practical recipe is "generate target outputs on deployment-like prompts", not "add raw web
text". If it does not, OOD acceptance is limited by target alignment/capacity, not data volume.

## E8 — Replication family

**Goal:** address "single family" reviewer objection.

- Target `Qwen3.6-35B-A3B` (old teacher) with drafts `Qwen3.5-2B/4B/9B` — cross-family, but different
  tokenizer? Check; if tokenizer differs, use a shared-vocab subset or skip.
- Target `Qwen3.8-27B` with drafts `Qwen3.5-2B/4B/9B` — same tokenizer family.
- If time: Llama-3.2-1B/3B drafts against Llama-3.1-8B (tokenizer differs; only use if vocabulary
  overlap is high).

Run E2 on 2 domains (math500 + wikitext) for each replication. Report whether the floor ordering is
preserved.

## E9 — Ablations

- Clustering: `P(C|C)`, `P(C|I)`, block-start `P(C|I)`; Markov chain prediction of `alpha(gamma)`.
- Heterogeneity: stretched exponential `beta` vs draft quality; Beta mixture fit.
- Margin certificate: `P(rho < 1, match=0)` should be ~0; report violations (top-k truncation caveat).
- `q_h` vs `q`: is the certificate tight or loose? (Expected: loose for high-capacity drafts.)
- Domain ordering: `q_inf(math) < q_inf(code) < q_inf(chat) < q_inf(prose)`.
- Length / position: first-block cold start vs mid-stream; report separately.

## Run order

```
E0 -> E1 -> E2 -> E3 -> (E4, E5, E6 in parallel) -> E8 -> E9 -> E7 (optional)
```

Do **not** start E7 before E2/E3: if the off-the-shelf series already shows a clean floor, E7 is an
ablation, not the main result.

## Artifacts and naming

```
data/processed/{domain}_target.jsonl
out/accept/{domain}_qwen3.5-{size}.json
out/accept/{domain}_qwen3.5-{size}.npz
out/loop/{domain}_qwen3.5-{size}_g{gamma}.json
out/lm_eval/{model}_{task}.json
out/fit/capacity/{domain}_fit.json
out/fig/{name}.pdf
out/train/{model}_{tokens}.jsonl
```

## Stop / pivot rules

- If ModelScope stays down for > 24h: use the already-present 4B/9B and the old Qwen3.6-35B-A3B
  target; the 2-point series is enough for a pilot, not for the paper.
- If `q(D)` does not saturate on any OOD domain: pivot to "capacity scaling works; the wall is
  quantization/data coverage". Still publishable, but change the title and claims.
- If the real loop deviates strongly from teacher forcing at gamma=8: investigate within-block drift
  before scaling up E4.
- If the 35B target does not fit / is too slow: use Qwen3.5-27B as target and 0.8B/2B/4B/9B as drafts;
  document the change.
