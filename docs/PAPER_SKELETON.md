> **Note (2026-11-06):** the current paper is the compression-floor rewrite in
> `paper/main.tex` / `docs/COMPRESSION_FLOOR.md`. This file documents the previous
> off-the-shelf capacity-scan revision and is kept for provenance.

# Paper Skeleton (Route A, revised 2026-11-04)

> Working title: **The Limits of Draft Scaling in Speculative Decoding: Diminishing Returns and
> Distribution Shift**
>
> Author: Lixuan Wei (Institute of Automation, Chinese Academy of Sciences; work done during an
> internship; undergraduate institution omitted).
>
> Status: revised draft with the current results. The original "irreducible floor" claim has been
> narrowed to a **capacity-axis asymptotic residual**. See `paper/main.tex` and
> `out/paper_stats/summary.md`.

## Abstract (current)

Speculative decoding accelerates LLM inference by pairing a target with a cheaper draft. Prior work
reports log-linear acceptance scaling with draft capacity. We study whether this scaling saturates
under distribution shift. Sweeping a same-family capacity series (Qwen3.5-0.8B/2B/4B/9B/27B) against
a 35B MoE target on six evaluation sets, including raw and target-generated OOD prose, we fit
`q(D) = q_inf + A D^-alpha`. On raw human OOD text the fitted intercept is ~0.13 and the
sequence-bootstrap CI excludes zero, but with five capacity points the profile-likelihood CI includes
zero and leave-one-out prediction does not favor the floor model; a pooled shared-exponent fit gives
0.11–0.17. On target-generated OOD text the intercept is much smaller and not reliably
distinguishable from zero: WikiText gives 0.00 (0.8B→2B improves, 2B–9B are flat or slightly
non-monotone, and only 27B improves further), while FineWeb gives 0.033 with a CI including zero and
a shallow but monotone curve. We therefore do not claim an irreducible floor; `q_inf` is a
capacity-axis asymptotic residual that depends on the measurement distribution.
W8 is acceptance-free and W4 adds a separate additive term. Real loops agree with teacher forcing
within 5% on MATH, but raw WikiText teacher forcing is conservative: the real loop gives 1.29–1.49x
higher `tau` than teacher forcing. Controlled distillation improves in-domain `q` by 5.8–7.8x but worsens OOD
`q` by 1.25–2.07x at 10–20M tokens; raw FineWeb and 1M target-aligned FineWeb do not fix it at this
scale. The 27B dense draft costs ~9x the target's active parameters, so it is a capacity probe, not
a deployable cheaper draft. On OOD text, coverage and target alignment—not raw capacity alone—are
the binding constraints.

## Contributions

1. **A controlled capacity series and measurement protocol** (same family/tokenizer, 0.8B–27B, six
   evaluation sets, raw and generated OOD, open code/caches).
2. **A capacity-axis fit with explicit uncertainty**: BIC/AIC, `Delta`BIC, sequence bootstrap,
   profile likelihood, leave-one-capacity-out prediction, and a pooled shared-exponent sensitivity
   analysis. The intercept is positive on raw OOD text and small or zero on generated OOD text; it
   is not an irreducible lower bound.
3. **A separation of capacity from quantization**: W8 is free for acceptance, W4 adds an additive
   term; the capacity-axis residual survives both.
4. **Diagnostics, not a mechanism proof**: TV/margin certificate (loose), survival shape, top-k
   recall, and entropy stratification.
5. **A bounded training study**: 2B/4B distillation at 10–20M tokens shows specialization; in-domain
   improves 5.8–7.8x, OOD worsens 1.25–2.07x. Raw and 1M target-aligned FineWeb do not fix OOD at
   this scale. Single seed; do not generalize beyond this regime.

## Method

- Target: Qwen3.5-35B-A3B (35B total / ~3B active, MoE).
- Drafts: Qwen3.5-0.8B / 2B / 4B / 9B / 27B (dense, shared tokenizer).
- Evaluation sets: MATH-500, GSM8K, HumanEval, MT-Bench, WikiText-2 raw, FineWeb-Edu raw; plus
  generated-OOD WikiText and FineWeb controls (100 target continuations from raw prefixes each).
- Metrics: greedy `q`, TV/KL/JS, margin `delta`, `rho`, `q_h`, block simulation, survival, clustering.
- Fits: floor vs pure power vs log-linear vs inverse-log; BIC/AIC; sequence bootstrap; profile CI;
  LOOCV; pooled raw-OOD shared-exponent fit.
- Controls: Qwen3.6 cross-family (MATH + raw WikiText), 9B BF16/W8/W4, real loops (5–10 prompts),
  2B/4B distillation (10–20M tokens, single seed).

## Key numbers

### Raw sets (Qwen3.5 target)

| set | q_inf | seq. bootstrap | profile | Delta BIC | LOO floor / best alt |
|---|---:|---|---|---:|---|
| MATH-500 | 0.0287 | [0.0136, 0.0332] | [0.0000, 0.0416] | +2.36 | 0.0166 / 0.0093 |
| GSM8K | 0.0293 | [0.0214, 0.0326] | [0.0000, 0.0443] | +2.09 | 0.0208 / 0.0063 |
| HumanEval | 0.0165 | [0.0057, 0.0206] | [0.0092, 0.0199] | +0.66 | 0.0083 / 0.0024 |
| MT-Bench | 0.0858 | [0.0000, 0.1237] | [0.0000, 0.1256] | +1.85 | 0.0367 / 0.0130 |
| WikiText raw | 0.1368 | [0.0983, 0.1615] | [0.0000, 0.1938] | +2.41 | 0.0270 / 0.0153 |
| FineWeb raw | 0.1256 | [0.0977, 0.1437] | [0.0000, 0.1778] | +1.98 | 0.0622 / 0.0190 |
| WikiText gen. | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.1503] | -1.82 | 0.0238 / 0.0247 |

Pooled raw-OOD (shared A=0.215, alpha=0.404): WikiText 0.145 [0.084, 0.179], FineWeb 0.104
[0.043, 0.138], Qwen3.6 WikiText 0.166 [0.105, 0.200].

### Cross-family (Qwen3.6 target)

MATH: `q = 0.0990/0.0776/0.0606/0.0558/0.0497`, `q_inf = 0.0428` [0.0340, 0.0483]
(profile [0.0231, 0.0517]), `R^2 = 0.989`, `Delta`BIC +1.55.
Raw WikiText: `q = 0.402/0.337/0.282/0.259/0.218`, `q_inf = 0.1476` [0.0987, 0.1771]
(profile [0.0000, 0.2057]), `R^2 = 0.993`, `Delta`BIC +2.82.
Raw FineWeb: `q = 0.346/0.290/0.233/0.206/0.185`, `q_inf = 0.1348` [0.1002, 0.1557]
(profile [0.0000, 0.1854]), `R^2 = 0.987`, `Delta`BIC +2.50.
No generated-OOD replication across target families.

### Real loop (pilot)

| domain | draft | gamma | n | teacher tau | real tau | ratio [95% CI] |
|---|---|---:|---:|---:|---:|---|
| MATH | 2B | 8 | 10 | 6.73 | 6.79 | 1.02 [0.95, 1.10] |
| MATH | 27B | 8 | 10 | 7.67 | 7.66 | 1.01 [0.96, 1.06] |
| MATH | 2B | 16 | 5 | 11.01 | 10.94 | 1.01 [0.87, 1.15] |
| MATH | 27B | 16 | 5 | 13.14 | 12.64 | 0.96 [0.89, 1.03] |
| WikiText | 2B | 8 | 10 | 2.92 | 4.39 | 1.49 [1.31, 1.66] |
| WikiText | 27B | 8 | 10 | 4.15 | 5.35 | 1.29 [1.18, 1.40] |

### Controlled distillation (2B/4B, 10–20M tokens)

| model | training | MATH | WikiText raw | Gen-OOD | FineWeb raw |
|---|---|---:|---:|---:|---:|
| 2B | off | 0.0728 | 0.3174 | 0.1348 | 0.2769 |
| 2B | narrow 10M | 0.00935 | 0.4668 | 0.2019 | 0.4299 |
| 2B | narrow 20M | 0.00993 | 0.6020 | 0.2792 | 0.5409 |
| 2B | mixed raw FineWeb 10M | 0.0280 | 0.4831 | 0.2251 | 0.4022 |
| 2B | target-aligned FineWeb 10M | 0.0189 | 0.5086 | 0.2273 | 0.4157 |
| 4B | off | 0.0525 | 0.2581 | 0.1430 | 0.2171 |
| 4B | narrow 20M | 0.00899 | 0.4361 | 0.1791 | 0.3971 |

In-domain improvement: 5.8–7.8x. OOD worsening: 1.25–2.07x. Single seed; 4B uses 20M vs 2B 10M in
the narrow comparison; no learning curves.

## Diagnostics

- Margin certificate: violation rate <= 1.7e-4; `q_h` is a loose upper bound (WikiText 27B:
  `q = 0.200`, `q_h = 0.490`). Diagnostic only.
- Survival beta: WikiText ~0.96 (near exponential); chat/code 0.67–0.75 (clustered); math 0.80–0.88.
- Top-k recall among rejected tokens (27B): WikiText top-5 82%, top-10 93%; MATH top-5 97%,
  top-10 99.9%. OOD errors are mostly near-misses just outside the draft's top-5.
- Entropy quintiles (27B): WikiText 0.017/0.067/0.178/0.291/0.446; MATH 0/0/0/0.001/0.193.
  The raw-OOD residual is spread across confident and uncertain tokens.

## Related work positioning

- `tau_inf = 1/q` is standard under stationarity/regeneration (Leviathan 2023; Lookahead 2025);
  present as a lemma with explicit conditions.
- TV/margin certificates: *When Is a Draft Accepted?* (arXiv 2606.30265) — cite; our `rho` is an
  interpretable corollary, not the main theorem.
- Capacity scaling: Scylla (arXiv 2505.07858) — our residual is the boundary condition under domain
  shift.
- Acceptance != speed: *Performance or Illusion?* (MLSys 2026).
- Domain shift: OSD (2310.07177), domain drafts (2503.07807).
- Dynamic depth: SVIP (EMNLP 2025), PACE-dLLM.
- Training objectives: VSD (ICML 2026), Spec-AUF (ICML 2026), LK losses.

## Figures

1. **Fig. 1**: q(D) per set with fitted asymptotic residuals, raw + generated OOD.
2. **Fig. 2**: q_inf with sequence-bootstrap and profile CIs, plus pooled OOD.
3. **Fig. 3**: controlled 2B distillation.
4. **Fig. 4**: real loop vs teacher forcing with prompt-level CIs.

## Limitations

- Off-the-shelf drafts differ in pretraining data; capacity is not perfectly controlled.
- Single primary target; Qwen3.6 replication covers MATH, raw WikiText, and raw FineWeb only.
- Five capacity points; profile CIs include zero on most individual sets; LOOCV does not always
  favor the floor model; BIC margins are 2–3 points on raw OOD.
- Generated-OOD covers WikiText and FineWeb, but not the Qwen3.6 target. WikiText gives zero;
  FineWeb gives 0.033 with a CI including zero.
- MATH/GSM8K target outputs truncated at 512 tokens (MATH 386/500 near cap).
- Top-256 target cache; residual included but tail approximations remain.
- Real-loop cells are 5–10 prompts; pilot-scale.
- Distillation is single-seed, 10–20M tokens; do not generalize.
- 27B dense draft is not cheaper than the target's active compute; capacity probe only.
- The full capacity scan is greedy; standard speculative sampling is measured from the same caches
  and validated with a 15-prompt-per-cell real loop. Top-k/tree verification not tested.
- Positive control is a single-seed 2B intervention; the gentle-vs-aggressive contrast confounds
  learning rate with token budget.

## Venue / timeline

- Main target: NeurIPS 2027 (abstract 2027-05-14, paper 2027-05-21).
- Stretch: ACL 2027 ARR (2027-01-04) if the optional follow-ups are done by mid-December.
- Backup: ICML 2027 (2027-01-22) / AAAI 2028.
