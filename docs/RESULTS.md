# Core Results: Capacity Scan + Floor Fit (revised 2026-11-03)

> **Revision note.** The original draft called `q_inf` an "irreducible floor". The revised paper
> narrows this to a **capacity-axis asymptotic residual**. Key changes: (i) the generated-OOD
> WikiText control has a zero fitted intercept and is reported as a boundary condition; (ii) the
> five-point fit is weakly identified, so profile-likelihood CIs and leave-one-capacity-out errors
> are reported alongside the sequence bootstrap; (iii) the floor model wins BIC on raw sets by only
> 2--3 points and does not always win LOOCV; (iv) real-loop ratios are reported with prompt-level
> CIs, replacing the unsupported "within 3%" claim; (v) the 27B dense draft is explicitly flagged as
> a capacity probe, not a cheaper draft. See `paper/main.tex` and `out/paper_stats/summary.md`.
>
> Target: `Qwen3.5-35B-A3B` (35B total / ~3B active, MoE).
> Drafts: `Qwen3.5-0.8B / 2B / 4B / 9B / 27B` (dense, same family/tokenizer).
> Metrics: greedy top-1 disagreement `q`, full/top-256 target cache with residual.
> Data: target-generated greedy continuations, max_new 512 (MATH/GSM8K/MT-Bench/HumanEval),
> WikiText-2 raw 1024-token chunks (512 scored).
> Fit: `q(D) = q_inf + A D^-alpha`, compared with pure power-law and log-linear by BIC;
> `q_inf` CI from 500 bootstrap resamples over sequences.

## 1. Headline (revised)

**The floor model wins in-sample BIC on the raw sets, but the intercept is only weakly identified;
the generated-OOD intercept is zero on WikiText and small (CI includes 0) on FineWeb.**

- Raw WikiText-2: `q_inf = 0.137` (sequence bootstrap 0.098–0.161; profile 0.000–0.194).
  The 27B draft reaches `q = 0.200` (`tau_inf = 5.0`).
- Raw FineWeb-Edu: `q_inf = 0.126` (bootstrap 0.098–0.144; profile 0.000–0.178).
- Generated-OOD WikiText: `q_inf = 0`; the floor model loses BIC by 1.8 points. Adjacent paired
  tests show 0.8B–9B are statistically indistinguishable, and only 27B improves.
- Generated-OOD FineWeb: `q_inf = 0.033` (bootstrap 0.000–0.052; profile 0.000–0.049); the floor
  model wins BIC by only 0.97 and loses LOOCV. The curve is shallow but monotone.
- MT-Bench: `q_inf = 0.086` (bootstrap and profile CIs include 0); do not claim a positive residual.
- In-domain MATH/GSM8K: `q_inf ≈ 0.029`; HumanEval: `q_inf = 0.016`.
- Pooled raw-OOD fit with a shared exponent: WikiText-Qwen3.5 0.149 [0.101, 0.178],
  FineWeb-Qwen3.5 0.109 [0.061, 0.138], WikiText-Qwen3.6 0.170 [0.122, 0.200],
  FineWeb-Qwen3.6 0.123 [0.075, 0.152]. This is a sensitivity analysis, not a per-set
  identification.

**Interpretation:** the raw-OOD point estimates are ~4.5× the in-domain ones, but the five-point
fit cannot support a claim of an irreducible lower bound. The generated-OOD controls show that the
answer depends on the measurement distribution and the corpus: target-generated WikiText gives zero,
target-generated FineWeb gives a small positive point estimate whose CI includes zero. The
defensible claim is conditional: for a fixed target, draft family, recipe, measurement distribution,
and capacity range, the fitted capacity-axis intercept is positive on raw OOD text and small or
zero on target-generated OOD text.

## 2. q(D) table

| draft | D (B) | math500 | gsm8k | humaneval | wikitext | mtbench |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 0.8 | 0.0958 | 0.0973 | 0.0374 | 0.3805 | 0.2497 |
| 2B | 2.0 | 0.0728 | 0.0756 | 0.0292 | 0.3174 | 0.2046 |
| 4B | 4.0 | 0.0525 | 0.0544 | 0.0228 | 0.2581 | 0.1577 |
| 9B | 9.0 | 0.0470 | 0.0485 | 0.0201 | 0.2350 | 0.1419 |
| 27B | 27.0 | 0.0388 | 0.0407 | 0.0191 | 0.1998 | 0.1216 |

## 3. tau_inf = 1/q

| draft | math500 | gsm8k | humaneval | wikitext | mtbench |
|---|---:|---:|---:|---:|---:|
| 0.8B | 10.4 | 10.3 | 26.7 | 2.6 | 4.0 |
| 2B | 13.7 | 13.2 | 34.2 | 3.2 | 4.9 |
| 4B | 19.0 | 18.4 | 43.8 | 3.9 | 6.3 |
| 9B | 21.3 | 20.6 | 49.7 | 4.3 | 7.0 |
| 27B | 25.8 | 24.6 | 52.3 | 5.0 | 8.2 |

## 4. Floor fit

| domain | q_inf | 95% CI | A | alpha | R² | BIC(floor) | BIC(power) | BIC(loglin) | best |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| math500 | 0.0287 | [0.0135, 0.0334] | 0.0599 | 0.557 | 0.9870 | **-55.75** | -52.75 | -47.18 | floor |
| gsm8k | 0.0293 | [0.0212, 0.0325] | 0.0611 | 0.533 | 0.9835 | **-54.51** | -52.42 | -47.23 | floor |
| humaneval | 0.0165 | [0.0065, 0.0205] | 0.0183 | 0.667 | 0.9869 | **-66.71** | -60.23 | -56.64 | floor |
| wikitext | 0.1368 | [0.0967, 0.1615] | 0.2252 | 0.390 | 0.9909 | **-46.18** | -43.77 | -38.88 | floor |
| mtbench | 0.0858 | [0.0000, 0.1238] | 0.1501 | 0.449 | 0.9851 | **-46.94** | -45.08 | -40.49 | floor |

The sequence-bootstrap CI excludes 0 on 4/5 domains; MT-Bench is the exception (only 80 prompts /
28k tokens). This is **not** the whole story: with five capacity points and three fit parameters,
the profile-likelihood CI for `q_inf` includes 0 on most sets, and leave-one-capacity-out prediction
is worse for the floor model than for a simpler alternative (e.g. WikiText 0.027 vs 0.015; FineWeb
0.062 vs 0.019). `Delta`BIC in favor of the floor is only +2.0 to +2.4 on raw OOD. The floor model is
therefore an in-sample description, not a validated extrapolation. See `out/paper_stats/summary.md`
for the full table.

A pooled raw-OOD fit with one intercept per set and a shared `A, alpha` gives WikiText 0.145
[0.084, 0.179], FineWeb 0.104 [0.043, 0.138], and Qwen3.6 WikiText 0.166 [0.105, 0.200], with shared
`A = 0.215`, `alpha = 0.404`, `R^2 = 0.989`. Under the shared-shape assumption the residual is
positive; this does not identify each set separately.

## 5. Diagnostics: margin certificate and survival shape (not a mechanism proof)

**Margin certificate.** Across all 25 `(draft, domain)` cells, the violation rate
`P(rho < 1, disagreement)` is at most `1.7e-4` (top-256 truncation; exact TV would be >= the
cached TV, so this is a finite-sample/truncation artifact). The certificate `rho < 1 => accept`
holds. `q_h = P(rho >= 1)` is a **loose** upper bound: e.g. WikiText 27B has `q=0.200` but
`q_h=0.490`; MATH-500 27B has `q=0.039` but `q_h=0.067`. Use it as a guarantee, not an estimator.

**Survival shape.** Fit `S(k) = exp(-(k/lambda)^beta)` to each cell:

| domain | beta (0.8B -> 27B) | lambda (0.8B -> 27B) |
|---|---|---|
| math500 | 0.80 -> 0.88 | 15 -> 35 |
| gsm8k | 0.81 -> 0.88 | 14 -> 33 |
| humaneval | 0.68 -> 0.79 | 82 -> 127 |
| mtbench | 0.67 -> 0.75 | 5 -> 12 |
| wikitext | 0.93 -> 0.96 | 2.2 -> 4.7 |

Interpretation: WikiText is nearly memoryless (`beta≈1`): errors are dense and not strongly
clustered, so the finite-γ curve is close to the geometric `q^k` prediction. Chat/code have
`beta≈0.7`, i.e. strong clustering; finite-γ acceptance can be worse than the mean `q` suggests.
This is the second axis of the story: **`q` sets the asymptote `tau_inf`, `beta` sets how fast you
approach it.**

## 6. Real-loop validation

Greedy real loop, target 35B, max_new=128. Teacher-forced values are computed on **the same first
N sequences** (not the full-domain average), so the comparison is apples-to-apples.

| domain | draft | γ | n | teacher τ (same seqs) | real τ | ratio |
|---|---|---:|---:|---:|---:|---:|
| math500 | 2B | 8 | 10 | 6.48 | **6.79** | 1.05 |
| math500 | 27B | 8 | 10 | 7.56 | **7.66** | 1.01 |
| math500 | 2B | 16 | 5 | 10.81 | **10.94** | 1.01 |
| math500 | 27B | 16 | 5 | 13.18 | **12.64** | 0.96 |
| wikitext | 2B | 8 | 10 | 2.86 | **4.39** | 1.53 |
| wikitext | 27B | 8 | 10 | 4.13 | **5.35** | 1.29 |

- **MATH: paired real/teacher ratios 0.96–1.02**, with 95% prompt-level bootstrap CIs including 1.0.
  Teacher forcing is an accurate mean approximation on this deployment distribution. The original
  "within 3%" claim is too strong for the raw-OOD cells and has been removed.
- **WikiText: ratios 1.29–1.49** (2B 1.49 [1.31, 1.66]; 27B 1.29 [1.18, 1.40]). The teacher-forced
  sequences are raw human text; the real loop generates from the target, so its continuations are
  model-generated and easier. Treat the raw-text q as a **conservative/harder** OOD distribution.
  These are pilot-scale cells (5–10 prompts), so the CIs are wide.
  Generated-OOD WikiText and FineWeb sets (target continuations from raw prefixes) are now included
  so the teacher-forced and real-loop OOD distributions can be compared.
- The per-depth shape has two effects: **block-start bias** (the first proposal in every block is
  harder; 2B MATH γ=16: 0.945 marginal vs 0.873 block-start) and **selection** (conditional `p_k`
  rises after the first position). The mean τ is still within 1–5%. The block-start penalty is
  smaller for the stronger draft (27B γ=16 first-proposal acceptance 0.929 vs 2B 0.873), so the
  teacher-forcing gap shrinks with capacity.
- **Finite-γ check of `τ_inf = 1/q`:** at γ=64, `τ(64)/(1/q)` is 1.00 on raw WikiText/FineWeb,
  0.88–0.98 on MATH/GSM8K, 0.84–0.99 on generated OOD, and 0.65–0.81 on HumanEval. The identity is
  asymptotic; low-q, clustered domains need much larger γ to approach `1/q`. The fitted floor caps
  the asymptotic τ, not the finite-γ value.

## 7. Quantization control

The capacity series is **BF16**, so the floor is not a quantization artifact. To separate the
quantization term, quantize the 9B draft (weight-only RTN, group 128, lm_head skipped) and reuse
the same target cache:

| draft | quant | math500 q | math500 E[TV] | math500 tau_inf | wikitext q |
|---|---|---:|---:|---:|---:|
| 9B | BF16 | 0.0470 | 0.0477 | 21.3 | 0.2350 |
| 9B | W8 | 0.0469 | 0.0478 | 21.3 | 0.2357 |
| 9B | W4 | 0.0649 | 0.0677 | 15.4 | 0.2660 |

- **W8 is free** for acceptance (Δq ≈ 0.0001 on MATH, 0.0007 on WikiText).
- **W4 adds an additive floor**: Δq ≈ +0.018 on MATH-500 and +0.031 on WikiText, independent of
  capacity; `tau_inf` drops 21.3→15.4 (MATH) and 4.25→3.76 (WikiText).
- Implication: the old report's `35B-W4` `q=0.17` on WikiText mixes capacity and quantization.
  Our BF16 capacity series is the clean test; the capacity-axis residual is not a W4 artifact.

## 8. LM quality (partial)

`lm-eval` WikiText word perplexity (HF backend, 50 samples for 4B/9B, 20 for 27B):

| model | word PPL |
|---|---:|
| Qwen3.5-4B | 10.91 |
| Qwen3.5-9B | 9.17 |
| Qwen3.5-27B | 7.53 |

This orders with capacity and is consistent with the acceptance floor (stronger draft = lower q).

**GSM8K is not usable yet.** Qwen3.5 thinks by default; `lm-eval --apply_chat_template` does not
pass `enable_thinking=False`, so generation is truncated and exact-match is 0.0–0.02. Even with a
no-think template (empty `<think>` block), the model still emits a long "Thinking Process" and
`lm-eval`'s `####` extractor returns `[invalid]`; this is a **task-format / answer-extraction**
issue, not a model-capability result. Do not report GSM8K until a custom extractor or a direct-answer
prompt is used. WikiText PPL is the usable quality signal for now.

## 9. Generated-OOD: WikiText zero, FineWeb small positive

To match the real-loop distribution, we generated 100 target continuations from raw WikiText
prefixes (`wikitext_gen_target.jsonl`) and reran the 5-draft scan:

**WikiText generated:**

| draft | q | E[TV] | tau_inf |
|---|---:|---:|---:|
| 0.8B | 0.1497 | 0.1783 | 6.68 |
| 2B | 0.1348 | 0.1570 | 7.42 |
| 4B | 0.1430 | 0.1572 | 6.99 |
| 9B | 0.1510 | 0.1596 | 6.62 |
| 27B | 0.1037 | 0.1147 | 9.64 |

- The floor model is **not supported** here: R²=0.43, `q_inf=0`, `Delta`BIC = -1.82 (floor loses to
  the best alternative), log-linear also poor.
- Paired per-sequence comparisons: the 2B->4B difference is -0.0082 [-0.0177, +0.0009] and the
  4B->9B difference is -0.0080 [-0.0160, -0.0010], so the 0.8B–9B group is statistically
  indistinguishable or slightly non-monotone. 27B is clearly better (+0.0473 [+0.0380, +0.0572]).
- This is a **boundary condition on the main claim**, not a nuisance: the fitted intercept depends on
  the measurement distribution. The paper reports both raw and generated OOD and does not claim an
  irreducible floor.
- Interpretation: model-generated OOD prose is easier and more homogeneous; small drafts plateau,
  and pretraining-data differences dominate in the 0.8B–9B range. The off-the-shelf capacity axis is
  **not controlled** at this granularity.
- This does **not** invalidate the raw-text point estimate (which is monotone and has a positive
  `q_inf`); it shows that the residual is distribution-dependent.

**FineWeb generated:** 100 target continuations from raw FineWeb prefixes, same 5-draft scan:

| draft | q | E[TV] | tau_inf |
|---|---:|---:|---:|
| 0.8B | 0.0870 | 0.103 | 11.50 |
| 2B | 0.0750 | 0.090 | 13.33 |
| 4B | 0.0619 | 0.071 | 16.15 |
| 9B | 0.0534 | 0.059 | 18.73 |
| 27B | 0.0492 | 0.057 | 20.34 |

- Floor model: `q_inf=0.0330` (bootstrap [0.0000, 0.0522]; profile [0.0000, 0.0492]), R²=0.983,
  `Delta`BIC = +0.97 (barely wins), LOOCV 0.0119 vs 0.0039 (floor loses). The intercept is small
  and not reliably distinguishable from zero.
- All adjacent improvements are positive: 0.8->2 +0.0119, 2->4 +0.0131, 4->9 +0.0085,
  9->27 +0.0042. The curve is shallow but monotone, unlike generated WikiText.
- This is a **corpus-dependent boundary condition**: target-generated text reduces the residual
  substantially (0.13 raw -> 0.00–0.03 generated), but does not always eliminate it. The paper
  reports both generated controls and does not claim an irreducible floor.

## 10. Cross-family replication (Qwen3.6-35B-A3B target)

Same tokenizer (248077), different target family. 200 MATH prompts, 100 raw WikiText chunks,
100 raw FineWeb chunks.

**MATH-500:**

| draft | q | tau_inf |
|---|---:|---:|
| 0.8B | 0.0990 | 10.1 |
| 2B | 0.0776 | 12.9 |
| 4B | 0.0606 | 16.5 |
| 9B | 0.0558 | 17.9 |
| 27B | 0.0497 | 20.1 |

Floor model: `q_inf=0.0428` (95% CI 0.0331–0.0485), R²=0.989, BIC wins over power/log-linear.
The Qwen3.6 target is harder than Qwen3.5 (`q_inf=0.0287` on Qwen3.5). The point estimate is
positive, the bootstrap CI excludes zero, and the profile CI is [0.0231, 0.0517]. The capacity
ordering replicates.

**Raw WikiText (OOD):**

| draft | q | tau_inf |
|---|---:|---:|
| 0.8B | 0.4023 | 2.49 |
| 2B | 0.3371 | 2.97 |
| 4B | 0.2816 | 3.55 |
| 9B | 0.2588 | 3.86 |
| 27B | 0.2184 | 4.58 |

Floor model: `q_inf=0.1476` (95% CI 0.0974–0.1756), R²=0.993, BIC wins over power/log-linear.
The raw-text OOD ordering replicates across target families (Qwen3.5:
0.3805 / 0.3174 / 0.2581 / 0.2350 / 0.1998; Qwen3.6:
0.4023 / 0.3371 / 0.2816 / 0.2588 / 0.2184). The Qwen3.6 target is slightly harder.

**Raw FineWeb (OOD):**

| draft | q | tau_inf |
|---|---:|---:|
| 0.8B | 0.3457 | 2.89 |
| 2B | 0.2899 | 3.45 |
| 4B | 0.2333 | 4.29 |
| 9B | 0.2057 | 4.86 |
| 27B | 0.1851 | 5.40 |

Floor model: `q_inf=0.1348` (bootstrap [0.1002, 0.1557]; profile [0.0000, 0.1854]), R²=0.987,
BIC wins by +2.50. The FineWeb ordering and the positive point estimate replicate with the
Qwen3.6 target; Qwen3.6 is slightly harder than Qwen3.5 (0.346 vs 0.333 at 0.8B).

**The point estimates agree:** Qwen3.5 `q_inf=0.1368` [0.0967, 0.1615] vs Qwen3.6
`q_inf=0.1476` [0.0974, 0.1756] on WikiText; Qwen3.5 `q_inf=0.1256` [0.0941, 0.1431] vs Qwen3.6
`q_inf=0.1348` [0.1002, 0.1557] on FineWeb. Bootstrap CIs overlap. The per-set profile CIs still
include zero for both WikiText and FineWeb. The pooled four-set shared-exponent fit gives
WikiText-Qwen3.5 0.149 [0.101, 0.178], FineWeb-Qwen3.5 0.109 [0.061, 0.138], WikiText-Qwen3.6
0.170 [0.122, 0.200], FineWeb-Qwen3.6 0.123 [0.075, 0.152], with shared A=0.210, alpha=0.411,
R²=0.988.

**Bottom line:** the single-family objection is addressed for the **raw-text OOD point estimate**:
both target families give `q_inf≈0.13–0.15` on both raw corpora, with overlapping bootstrap CIs.
This is not a claim of an irreducible floor. The generated-OOD non-monotonicity (section 9) is a
boundary condition; the controlled distillation arm (2B/4B, 10–20M tokens, same corpus) is a bounded
intervention, not a resolution of the pretraining-data confound.

## 11. FineWeb-Edu: a second OOD corpus

Streamed FineWeb-Edu (`sample-10BT`) from ModelScope, built 100 raw 1024-token continuations, and
reran the 5-draft scan against the Qwen3.5-35B-A3B cache:

| draft | q | E[TV] | tau_inf |
|---|---:|---:|---:|
| 0.8B | 0.3328 | 0.320 | 3.01 |
| 2B | 0.2769 | 0.262 | 3.61 |
| 4B | 0.2171 | 0.202 | 4.61 |
| 9B | 0.1897 | 0.174 | 5.27 |
| 27B | 0.1721 | 0.156 | 5.81 |

Floor model: `q_inf=0.1256` (95% CI 0.0941–0.1431), R²=0.982, BIC wins over power/log-linear.

**Combined OOD floor table:**

| corpus / domain | q_inf | 95% CI | R² |
|---|---:|---|---:|
| WikiText (Qwen3.5 target) | 0.1368 | [0.0967, 0.1615] | 0.991 |
| WikiText (Qwen3.6 target) | 0.1476 | [0.0974, 0.1756] | 0.993 |
| FineWeb-Edu (Qwen3.5 target) | 0.1256 | [0.0941, 0.1431] | 0.982 |
| FineWeb-Edu (Qwen3.6 target) | 0.1348 | [0.1002, 0.1557] | 0.987 |
| MT-Bench (Qwen3.5 target) | 0.0858 | [0.0000, 0.1238] | 0.985 |
| MATH-500 | 0.0287 | [0.0135, 0.0334] | 0.987 |
| GSM8K | 0.0293 | [0.0212, 0.0325] | 0.984 |
| HumanEval | 0.0165 | [0.0065, 0.0205] | 0.987 |

The raw-OOD **point estimates** are ~0.13 across two corpora and two target families, while in-domain
math is ~0.03 and code ~0.017. The raw-text residual is not WikiText-specific. The profile CIs are
wide (and include zero for the individual Wiki/FineWeb sets), so this is a point-estimate pattern,
not a proof of a common floor. Generated-OOD FineWeb gives a small positive point estimate
(0.033) with a CI including zero; generated-OOD WikiText gives zero.

## 12. Controlled distillation (2B and 4B): specialization hurts OOD

To separate capacity from pretraining data, we distilled 2B and 4B on a fixed 3.13M-token
target-generated corpus (Qwen3.6 + Qwen3.5 outputs). Checkpoints: 2B@10M, 4B@20M total
(4 GPUs × 5M per rank; the 4B run hit an NCCL error at shutdown after saving). Evaluated on the
same caches as the off-the-shelf series:

| model | training | math500 q | WikiText raw q | WikiText gen q | FineWeb q |
|---|---|---:|---:|---:|---:|
| 2B | off-the-shelf | 0.0728 | 0.3174 | 0.1348 | 0.2769 |
| 2B | narrow, 10M | **0.00935** | **0.4668** | **0.2019** | **0.4299** |
| 2B | narrow, 20M | **0.00993** | **0.60195** | **0.2792** | **0.5409** |
| 2B | mixed FineWeb, 10M | 0.0280 | 0.4831 | 0.2251 | 0.4022 |
| 2B | **target-aligned FineWeb, 10M** | **0.0189** | **0.5086** | **0.2273** | **0.4157** |
| 4B | off-the-shelf | 0.0525 | 0.2581 | 0.1430 | 0.2171 |
| 4B | narrow, 20M | **0.00899** | **0.4361** | **0.1791** | **0.3971** |

**This is the key controlled result at this scale:** training on the target's in-domain data makes
both drafts dramatically better in-domain (`q` 0.073/0.053 -> 0.009, a 5.8–7.8x improvement) but
**worse on every OOD corpus** (`q` 1.25–2.07x higher). More narrow training makes it worse: 2B raw `0.467@10M -> 0.602@20M`,
generated `0.202 -> 0.279`, FineWeb `0.430 -> 0.541`. At the same 20M budget, 4B is less damaged
than 2B (raw 0.436 vs 0.602) but still far worse than off-the-shelf (0.258). The draft specializes;
it does not generalize, and capacity only partially mitigates the damage.

**Raw FineWeb coverage is not enough.** Mixing 2.05M FineWeb-Edu self-supervised tokens into the
3.13M target corpus (2B, 10M training tokens) reduces specialization (MATH q 0.009 -> 0.028) but
does **not** improve OOD: WikiText raw 0.467 -> 0.483, generated 0.202 -> 0.225, FineWeb 0.430 ->
0.402. Raw web text teaches general language modeling, not the target's next-token choices.

**Target-aligned FineWeb is also not enough at this scale.** We generated 1.02M target tokens on
2000 FineWeb prompts and mixed them with the 3.13M math corpus; the 2B@10M result is MATH 0.019,
WikiText raw 0.509, generated 0.227, FineWeb 0.416. It is slightly less specialized than the
narrow-trained 2B (MATH 0.009) but still far worse OOD than off-the-shelf (raw 0.317). At this
scale, **all distillation variants are worse OOD than off-the-shelf**; the OOD gap is not removed
by capacity, raw web text, or 1M target-aligned tokens in this regime. A generalist/coverage-scale
intervention (orders of magnitude more target-aligned OOD data) is the plausible lever, but it is
not tested here.

Implications:

1. The off-the-shelf generated-OOD non-monotonicity (section 9) is a **data-coverage** effect, not
   a capacity effect: different pretraining corpora cover the OOD distribution differently.
2. Capacity scaling on narrow data does not remove the OOD gap at 10–20M tokens; **coverage is the
   plausible binding constraint**. This is a bounded intervention, not a general causal claim.
3. The in-domain residual can be driven very low (`tau_inf≈107–111`) by training on the target's own
   distribution; the OOD residual is a different object.

Caveats: the narrow corpus is a mix of Qwen3.6/Qwen3.5 target data; the 4B run hit an NCCL error at
shutdown after saving (checkpoint is valid); 4B used 20M total tokens vs 2B 10M. The
coverage-controlled 2B (FineWeb mixed, 10M) tests whether general web text recovers OOD: it does not
(raw 0.483 vs. 0.467 narrow, generated 0.225 vs. 0.202, FineWeb 0.402 vs. 0.430).

## 12.1 27B full fine-tune (2026-11-04): still worse on every corpus

We ported the 27B run to DeepSpeed ZeRO-3 and completed 20.1M tokens on the mixed corpus
(3.13M target + 2.05M raw FineWeb; 3.9 epochs; lr 5e-5 cosine; single seed). Checkpoints at
5/10/15/20M; full setup in `docs/TRAIN_27B.md`, raw numbers in
`docs/TRAIN_27B_RESULT.md`.

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

Every corpus is worse after training, including the raw corpora present in the mix. The damage
peaks at 10M and partially recovers by 20M (WikiText raw 0.290 → 0.268; MATH 0.101 → 0.067), which
looks like the usual early-distillation generalization damage rather than monotone specialization.
The 27B dense draft has more effective capacity than the 35B-total/3B-active target, so with more
tokens it should fit in-domain very well; the 20M point is a lower bound on fitting ability, not a
capacity limit. A 10M lr 1e-5 control was trained but not evaluated (deprioritised). The
checkpoints are valid (coherent generations; weights differ from base with cosine 0.999/0.998).
The paper therefore treats this as a bounded early-training control; the floor claim rests on the
off-the-shelf series, not on this run.

## 13. What this means for Route A (revised)

- **H2 is partially supported.** On raw WikiText and in-domain sets the floor model wins in-sample
  BIC and the point estimate is positive, but the profile CI includes zero on most individual sets
  and LOOCV does not always favor the floor. On generated OOD the intercept is zero on WikiText and
  0.033 (CI includes 0) on FineWeb. The paper therefore reports a conditional capacity-axis
  residual, not an irreducible floor.
- The **ordering of point estimates** is
  `humaneval (0.016) < math500 (0.029) ≈ gsm8k (0.029) < mtbench (0.086) < wikitext (0.137)`.
  MT-Bench is underpowered and its CI includes zero; treat the ordering as descriptive.
- The **capacity scaling is tail-dominated**: going 0.8B→27B reduces q by 2.5× on math but only
  1.9× on raw WikiText, and WikiText's absolute q stays at 0.20. The 27B draft is still 1.5× above
  the fitted raw-text intercept.
- **Standard SD is not a different story.** `q_s=E[TV]` is nearly the same risk as `q_greedy`
  (Pearson 0.99), and its local exponent is 0.20--0.32 on task/raw sets. The real loop confirms the
  ordering and the shallow exponent.
- **The alignment/coverage axis is real but small at this scale.** Gentle 2B adaptation reduces OOD
  `q` by 1.8--5.2% in point estimates (only raw WikiText is individually significant); the aggressive
  recipe overfits. The capacity conclusion is not replaced by a proven coverage recipe.
- The **practical corollary**: on OOD prose, a 27B draft gives `tau_inf≈5` but costs ~9× the
  target's active compute, so it is not a deployable cheaper draft. A smaller generalist with better
  coverage may be preferable; the paper does not run a latency-matched comparison, so this remains
  a hypothesis rather than a measured speedup.

## 14. Caveats

- MATH/GSM8K target outputs are truncated at 512 tokens (MATH 386/500, GSM8K 52/500 near the cap).
  A 30-prompt MATH regeneration at 768 tokens gives q=0.0761 vs 0.0803 (2B) and 0.0407 vs 0.0434
  (27B); the 512-token values are slightly conservative and the qualitative conclusion is unchanged.
  The acceptance measurement is valid on the generated tokens.
- Off-the-shelf models differ in pretraining data; the capacity axis is not perfectly controlled.
  The Qwen3.5 family/tokenizer is shared, but pretraining corpora may differ.
- Top-256 target cache; residual mass included, but tail approximations remain.
- WikiText is raw prose (no chat template); MT-Bench uses first-turn prompts only.
- **WikiText teacher forcing vs real loop use different distributions.** Teacher forcing scores the
  draft on raw human WikiText; the real loop generates from the target, so its continuations are
  model-generated and easier. For 2B, `tau(8)` is 3.02 on raw text but 4.39 in the real loop.
  Treat the raw-text q as a **conservative/harder** OOD distribution; the deployment-like residual
  may be lower. The paper reports both and uses the generated-OOD WikiText and FineWeb controls.
- The real-loop per-depth shape is domain-dependent: MATH-500 conditional `p_k` is ~flat, while
  WikiText `p_k` rises with depth (selection on easy segments). Do not assume a universal shape.
- `q` is greedy disagreement; the sampling-mode statement `q = E[TV]` is separate (and holds
  numerically: `E[TV] ≈ q` in the table).
- The floor fit uses 5 points; the bootstrap CI reflects sequence sampling, not model-family
  uncertainty. Profile-likelihood CIs include 0 on most individual sets, and LOOCV does not always
  favor the floor model. Replication with Qwen3.6/Qwen3.8 is E8.
- Generated-OOD covers WikiText and FineWeb, but not the Qwen3.6 target. WikiText gives a zero
  intercept; FineWeb gives 0.033 with a CI including zero. This corpus dependence is a boundary
  condition on the main claim.
- The 27B dense draft is not a cheaper draft relative to the target's ~3B active compute (9x).
  Only 0.8B/2B are clearly cheaper. Real-loop timings: draft/target time 0.75–0.84 for 2B and
  2.0–2.2 for 27B.
- The real-loop sample is 5–10 prompts per cell; report prompt-level CIs and treat as pilot-scale.

## 15. Optional follow-ups

**Done since the first revision:**

1. **Generated-OOD FineWeb**: 100 target continuations + 5-draft scan. Result: `q_inf=0.033`
   (bootstrap [0, 0.052]; profile [0, 0.049]), shallow but monotone. This shows the zero-intercept
   result is WikiText-specific, not universal.
2. **Qwen3.6 FineWeb replication**: 100 raw FineWeb chunks + 5-draft scan. Result:
   `q_inf=0.1348` [0.1002, 0.1557], `Delta`BIC +2.50. This completes the cross-family story for the
   second raw OOD corpus.

**Done since the second revision (2026-11-04):**

- **Standard SD check**: exact `q_s=E[TV]` from caches for 11 domains × 5 drafts, plus a real
  standard-SD loop on 15 prompts/cell for MATH-500 and raw WikiText. See §16.
- **Positive control**: gentle 2B adaptation on held-out target-generated FineWeb improves OOD q by
  1.8–5.2%; aggressive adaptation overfits. See §17 and `docs/POSITIVE_CONTROL.md`.
- **MATH 768-token length control**: 30 prompts; q 0.0803 -> 0.0761 (2B), 0.0434 -> 0.0407 (27B), so
  the 512-token truncation is small and slightly conservative.

**Still optional / not run:**

3. **Distillation seeds and learning curves**: 2–3 seeds per 2B/4B cell and 1M/5M/10M/20M checkpoints,
   plus experience-replay/regularization baselines. This would turn the bounded training result into
   a proper scaling study.
4. **More real-loop prompts**: 50–100 prompts per cell and γ=32. This would tighten the prompt-level
   CIs and quantify the block-start bias as γ grows.
5. **Regenerate MATH/GSM8K at `max_new=768`** to remove the 512-token truncation (MATH 386/500 near
   cap). **Done for 30 MATH prompts**: q 0.0803 -> 0.0761 (2B), 0.0434 -> 0.0407 (27B); the 512-token
   values are slightly conservative. GSM8K remains optional.
6. **lm-eval no-think**: build a text-only checkpoint with `enable_thinking=False` in the chat
   template, rerun GSM8K. WikiText PPL is already usable.
7. **35B-W4 control** (optional): separate the quantization term at the target scale.

## 16. Standard speculative sampling (2026-11-04)

The greedy scan is complemented by the standard stochastic rule. At a target-generated prefix the
expected acceptance probability of a draft token is `1 - TV(P,Q)`, so the cached per-token TV gives the
exact sampling rejection rate `q_s = E[TV]` (up to the top-256 residual). A Monte Carlo block
simulation with per-token acceptance probabilities gives `tau_sim(gamma)`. We also ran a real
standard-SD loop (draft samples from Q; accept with `min(1,P/Q)`; residual sampling on rejection;
bonus token from P) on 15 prompts per cell for MATH-500 and raw WikiText, `gamma=4`, 96 new tokens.

| domain | q_greedy alpha | q_s alpha | D1/2(q_s) | tau_sim(g=4) 0.8B -> 27B |
|---|---:|---:|---:|---:|
| MATH-500 | 0.264 | 0.316 | 9.0x | 3.9 -> 4.6 |
| GSM8K | 0.255 | 0.311 | 9.3x | 3.9 -> 4.6 |
| HumanEval | 0.195 | 0.264 | 13.8x | 4.4 -> 4.7 |
| MT-Bench | 0.209 | 0.262 | 14.0x | 3.0 -> 4.0 |
| WikiText raw | 0.184 | 0.202 | 30.8x | 2.4 -> 3.4 |
| FineWeb raw | 0.195 | 0.213 | 25.9x | 2.7 -> 3.6 |
| WikiText gen | 0.079 | 0.106 | n.m. | 3.4 -> 3.9 |
| FineWeb gen | 0.171 | 0.182 | 45.2x | 4.0 -> 4.4 |

Across 55 `(domain, draft)` cells, `q_greedy` and `q_s` are nearly the same quantity in this regime:
Pearson 0.9915, Spearman 0.9893, mean ratio `q_greedy/q_s = 0.909`, maximum absolute gap 0.0323.
The within-domain standardized correlation is 0.9926. The sampling exponent is slightly larger than
the greedy exponent, i.e. standard SD is if anything more favorable to capacity, but it is still
shallow: halving `q_s` needs 9--31x more capacity on task/raw sets. The greedy rejection rate is also
strongly anti-correlated with the simulated accepted length: Pearson -0.9835 across cells and -0.9905
after within-domain standardization, so it is a good proxy for `tau`.

Real-loop validation (`out/std_sd/loop_summary.md`): accepted length agrees with `tau_sim` within
0.9--1.1x; the loop's `E[TV]` is within 10--30% of the cache value. The gap is a decoding-distribution
effect (the cache uses target-greedy continuations; standard SD samples from the target and visits
higher-entropy prefixes, especially on MATH), not an estimator error. The ordering across drafts and
the shallow exponent are unchanged. Reproduce with:

```bash
# cache-based full curve
python -m infra.std_sd_stats
# real standard-SD loop (one GPU pair per process)
GPUS="0,1" DOMAINS="math500 wikitext" LIMIT=15 GAMMA=4 bash scripts/21_standard_sd.sh
python -m infra.std_sd_loop_stats
```

Raw outputs: `out/std_sd/stats.json`, `out/std_sd/loop_summary.json`, figures `paper/figures/fig5_std_sd.png`.

## 17. Positive control: gentle deployment adaptation (2026-11-04)

The review asked for a positive alignment/coverage baseline. We continue-train 2B on 1.8k **held-out**
target-generated FineWeb sequences (0.92M output tokens; first 100 FineWeb prompts reserved for
evaluation). Two recipes:

| run | tokens | FineWeb raw | FineWeb gen | WikiText raw | WikiText gen | MATH-500 |
|---|---:|---:|---:|---:|---:|---:|
| off-the-shelf 2B | -- | 0.2769 | 0.0750 | 0.3174 | 0.1348 | 0.0728 |
| gentle, lr 5e-6 | 0.5M | **0.2718** | **0.0733** | **0.3008** | **0.1297** | 0.0731 |
| aggressive, lr 2e-5 | 0.5M | 0.2966 | 0.0822 | 0.3284 | 0.1449 | 0.0797 |
| aggressive, lr 2e-5 | 1M | 0.3110 | 0.0892 | 0.3525 | 0.1530 | 0.0825 |
| aggressive, lr 2e-5 | 1.5M | 0.3103 | 0.0900 | 0.3517 | 0.1550 | 0.0828 |
| aggressive, lr 2e-5 | 2M | 0.3131 | 0.0903 | 0.3533 | 0.1557 | 0.0830 |

**Result.** Gentle adaptation improves every OOD set by 1.8--5.2% in point estimates and leaves MATH
unchanged. Only the raw WikiText change is individually significant (delta -0.0166, 95% CI
[-0.0272, -0.0055]); the others have CIs including zero. Aggressive adaptation overfits and worsens
every set, with damage growing with tokens. This is a real but small
positive control for the alignment/coverage axis. It does not remove the residual, and the contrast
confounds learning rate with token budget, so it is a recipe-sensitivity check, not a controlled LR
study. It also shows that "more target data" is not automatically better.

Reproduce: `bash scripts/22_positive_control.sh` (aggressive 2M run) plus the gentle 500k run described
in `docs/POSITIVE_CONTROL.md`. Raw results: `out/accept/positive/`.

## 18. Reproduce

```bash
# target generation (35B)
TARGET=qwen3.5-35b-a3b DM=balanced BS=8 bash scripts/03_target.sh

# capacity scan (5 drafts × 5 domains)
TARGET=qwen3.5-35b-a3b TARGET_DM=balanced DRAFT_DM=cuda:0 bash scripts/04_capacity_scan.sh

# fit
bash scripts/05_fit.sh

# mechanism: certificate + survival/beta
python -m infra.analyze_accept --glob 'out/accept/*.npz' --out out/analysis
```

Raw results: `out/accept/*.json`, `out/target_cache/*.npz`, `out/fit/capacity/fit.json`,
`out/fit/capacity/capacity_wall.png`.
