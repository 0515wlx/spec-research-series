# 27B training result (2026-11-04)

First completed 27B full fine-tune on the mixed distillation corpus, evaluated at
5M/10M/15M/20M tokens.  Run: `ckpt/zero3/qwen3.5-27b-mixed20m`, log
`out/logs/zero3_27b_20m.log`, evals in `out/accept/zero3/`.

## Setup

- Draft: Qwen3.5-27B text backbone, **26.90B** params (155 GFLOP/token).
- Corpus: `data/processed/distill_mixed.jsonl`, 5.18M target tokens =
  3.13M target math/chat + **2.05M raw FineWeb self-supervised** (the rows with
  `n_prompt=0` that the old FSDP loop dropped).
- 20.1M global tokens = 3.9 epochs; 536 steps; global batch ~37.3k tokens
  (6 sequences/GPU, no gradient accumulation).
- Full fine-tune, lr 5e-5 cosine, warmup 50, bf16, ZeRO-3, activation
  checkpointing.  Single seed.
- Evaluation: teacher-forced q against the Qwen3.5-35B-A3B target, existing
  caches; off-the-shelf 27B baseline from `out/accept/*_qwen3.5-27b.json`.

## q (lower is better)

| corpus | 27B off-the-shelf | 5M | 10M | 15M | 20M | Δ(20M) |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 | 0.0388 | 0.1165 | 0.1007 | 0.0799 | 0.0669 | +0.0281 |
| WikiText-2 raw | 0.1998 | 0.2899 | 0.2906 | -- | 0.2682 | +0.0684 |
| FineWeb-Edu raw | 0.1721 | 0.2801 | 0.2929 | -- | 0.2713 | +0.0991 |
| WikiText gen. | 0.1037 | 0.1368 | 0.1395 | -- | 0.1294 | +0.0257 |
| FineWeb gen. | 0.0492 | 0.0749 | 0.0837 | -- | 0.0742 | +0.0251 |
| C4 gen. | 0.0529 | -- | -- | -- | 0.0685 | +0.0156 |
| Gutenberg gen. | 0.0830 | -- | -- | -- | 0.1125 | +0.0295 |
| Code gen. | 0.0405 | -- | -- | -- | 0.0486 | +0.0081 |

Every corpus gets worse, including the raw corpora that the training mix
covers.  The OOD floor does **not** drop after 20M tokens.

Paired sequence-level bootstrap (2,000 resamples, baseline vs 20M) excludes zero
for every corpus: MATH $+0.030$ [0.026, 0.035], WikiText raw $+0.068$ [0.063,
0.074], FineWeb raw $+0.099$ [0.086, 0.114], WikiText gen $+0.026$ [0.017,
0.036], FineWeb gen $+0.025$ [0.021, 0.029], C4 gen $+0.016$ [0.012, 0.019],
Gutenberg gen $+0.030$ [0.024, 0.036], Code gen $+0.008$ [0.006, 0.010].

## Trajectory

The damage is non-monotone: it peaks around 10M and partially recovers by 20M.

- MATH: 0.0388 → 0.1165 (5M) → 0.1007 (10M) → 0.0799 (15M) → 0.0669 (20M)
- WikiText raw: 0.1998 → 0.2899 → 0.2906 → 0.2682
- FineWeb raw: 0.1721 → 0.2801 → 0.2929 → 0.2713

This shape is consistent with an optimization/forgetting transient (high LR for
a 27B full fine-tune) followed by partial recovery, rather than monotone
specialization.  It is **not** evidence that coverage would eventually fix OOD;
it is evidence that 20M tokens at this recipe do not.

Sanity check: the checkpoint is not corrupted.  Greedy generations on a GSM8K
prompt are coherent and correct for both models; lm_head / layer-0 MLP weights
differ from the base with cosine 0.9992 / 0.9978, and both models answer the
sample problem correctly.  The q gap is a token-level style/argmax shift, not a
broken export.

## Entropy strata

Within target-entropy quintiles (Q1 = most confident, Q5 = least), the damage is
not a composition effect:

| corpus | model | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| WikiText raw | base | 0.017 | 0.067 | 0.178 | 0.291 | 0.446 |
| | 20M | 0.016 | 0.105 | 0.267 | 0.403 | 0.549 |
| FineWeb raw | base | 0.004 | 0.063 | 0.157 | 0.253 | 0.384 |
| | 20M | 0.015 | 0.133 | 0.267 | 0.396 | 0.545 |
| WikiText gen. | base | 0.016 | 0.024 | 0.022 | 0.102 | 0.354 |
| | 20M | 0.008 | 0.006 | 0.015 | 0.156 | 0.462 |

Raw corpora worsen in every stratum.  On generated WikiText the trained model is
actually better on confident tokens (Q1–Q3) and worse only in Q4–Q5; the
aggregate q increase (0.104 → 0.129) is driven by the high-entropy tail.  This
is consistent with a distribution shift in the draft's confident predictions,
not a uniform loss of language ability.

## Near-miss, not collapse

The trained draft's top-1 agreement drops, but the target token stays near the
top of its distribution:

| corpus | model | top-1 | top-5 | top-10 | top-64 |
|---|---|---:|---:|---:|---:|
| WikiText raw | base | 0.810 | 0.973 | 0.987 | 0.997 |
| | 20M | 0.737 | 0.943 | 0.971 | 0.996 |
| FineWeb raw | base | 0.841 | 0.986 | 0.995 | 0.999 |
| | 20M | 0.734 | 0.944 | 0.972 | 0.994 |
| WikiText gen. | base | 0.900 | 0.976 | 0.984 | 0.991 |
| | 20M | 0.873 | 0.970 | 0.981 | 0.992 |

The argmax moves, but the target token is still in the top-5/top-10 at nearly
the baseline rate.  The fine-tune changes *which* near-tied token wins, not the
model's support.  This is consistent with the entropy-strata pattern and with a
style/distribution shift rather than a loss of language modeling.

## Train/eval overlap (important)

The mixed corpus is not disjoint from the evaluation sets:

- **In-sample:** MATH-500, GSM8K, HumanEval, MT-Bench prompts and target outputs are
  the same rows used to build `distill_mixed` (500/500/164/80 rows).  The
  in-domain numbers are training-set fits, **not** generalization.
- **In-sample:** the 100 raw FineWeb-Edu evaluation sequences are prefixes of the
  first 100 of the 2,000 raw FineWeb rows in the training mix (verified: the
  `fineweb-0` text is identical).  FineWeb raw q is therefore not a
  distribution-shift measurement.
- **Held out:** raw WikiText-2 (the corpus has no WikiText rows) and the
  generated WikiText/C4/Gutenberg/Code sets (their prompts are not in the mix;
  generated FineWeb shares its prompts with the raw FineWeb rows).

So the clean held-out OOD evidence is **WikiText raw + C4/Gutenberg/Code
generated**; all of them worsen after 20M training (0.200→0.268, 0.053→0.069,
0.083→0.113, 0.041→0.049).  This overlap also applies to the earlier 2B/4B
in-domain results and should be disclosed in the paper rather than presented as
in-domain generalization.

## Confound and control

The main confound is that 20M tokens is early in distillation.  Early
distillation commonly damages generalisation before the draft re-fits the
target distribution; the 10M peak and partial recovery by 20M are consistent
with that.  The 27B dense draft has more effective capacity than the
35B-total/3B-active target, so with enough tokens it should fit the target's
in-domain distribution very well.  The 20M result is therefore a lower bound on
what this draft can fit, not a statement that training cannot help.

An lr $1\times10^{-5}$, 10M-token control was trained
(`ckpt/zero3/qwen3.5-27b-mixed10m-lr1e-5`, checkpoints at 5M/10M) but not
evaluated: further distillation runs are deprioritised because the paper's claim
is about the OOD floor, not the draft's in-domain fitting ability.

Other caveats: single seed; one corpus mix; teacher-forced q only (no real
loop); 20M tokens is ~4 epochs of a 5.18M corpus, far below pretraining scale.

## Effect on the q(D) fit

The trained 27B is the same capacity point (D=27), so we replace the
off-the-shelf 27B in the five-point capacity fit and refit the floor model.
The fitted intercept rises on every corpus (it does not fall):

| corpus | q_inf, off-the-shelf 27B | q_inf, trained 27B |
|---|---:|---:|
| MATH-500 | 0.029 | 0.056 |
| WikiText raw | 0.137 | 0.245 |
| FineWeb raw | 0.126 | 0.228 |
| WikiText gen. | 0.000 | 0.000 |
| FineWeb gen. | 0.033 | 0.063 |
| C4 gen. | 0.037 | 0.062 |
| Gutenberg gen. | 0.054 | 0.099 |
| Code gen. | 0.034 | 0.044 |

This is descriptive (one model per capacity, single seed).  Because the 20M
checkpoint is still under-fitted in-domain (early distillation), the refit is
**not** evidence that the floor grows with training; it only shows that this
early checkpoint does not lower the intercept.

## What this means for the paper

The 27B dense draft has more effective capacity than the 3B-active target and
should eventually fit in-domain very well.  The 20M run is an early-training
control: it shows the usual early-distillation damage, not a capacity or
coverage limit.  The paper's floor claim rests on the off-the-shelf capacity
series and on the measurement distribution, not on the draft's fitting ability.
Keep the 27B training paragraph in the appendix, phrase it as a bounded early
control, and do not use it to claim that training cannot fix OOD.
