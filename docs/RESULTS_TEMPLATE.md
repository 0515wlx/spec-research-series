# Results Template (fill from `out/`)

> Copy the numbers from `out/accept/*.json` and `out/fit/capacity/fit.json` into these tables.
> Keep the raw JSON paths; do not hand-edit numbers without re-running the pipeline.

## 1. Capacity scan: q(D) per domain

| draft | D (B) | math500 | gsm8k | humaneval | wikitext | mtbench |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5-0.8b | 0.8 | | | | | |
| qwen3.5-2b | 2.0 | | | | | |
| qwen3.5-4b | 4.0 | | | | | |
| qwen3.5-9b | 9.0 | | | | | |
| qwen3.5-27b | 27.0 | | | | | |

Source: `out/accept/{domain}_qwen3.5-{size}.json` → `q`.

## 2. E[TV] per domain

| draft | math500 | gsm8k | humaneval | wikitext | mtbench |
|---|---:|---:|---:|---:|---:|
| 0.8B | | | | | |
| 2B | | | | | |
| 4B | | | | | |
| 9B | | | | | |
| 27B | | | | | |

Source: `E_TV`.

## 3. τ∞ = 1/q per domain

| draft | math500 | gsm8k | humaneval | wikitext | mtbench |
|---|---:|---:|---:|---:|---:|
| 0.8B | | | | | |
| 2B | | | | | |
| 4B | | | | | |
| 9B | | | | | |
| 27B | | | | | |

Source: `tau_inf`. Also record `tau(gamma)` for gamma = 8/16/32/64 from `per_gamma`.

## 4. Floor fit per domain

| domain | q_inf (fit) | q_inf 95% CI | A | alpha | R² | BIC(floor) | BIC(power) | BIC(loglin) | best |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| math500 | | | | | | | | | |
| gsm8k | | | | | | | | | |
| humaneval | | | | | | | | | |
| wikitext | | | | | | | | | |
| mtbench | | | | | | | | | |

Source: `out/fit/capacity/fit.json` → `fits.floor`, `bootstrap_q_inf`, `fits._best_by_bic`.

**H2 supported if:** floor wins BIC on >= 3 OOD domains and `q_inf` CI excludes 0. Otherwise record
the falsification and pivot (see `docs/ROUTE_A_PLAN.md` §5).

## 5. Margin / certificate diagnostics

| domain | mean margin δ | mean ρ | q_h = P(ρ≥1) | violations P(ρ<1, disagree) |
|---|---:|---:|---:|---:|
| math500 | | | | |
| wikitext | | | | |

Source: `E_margin`, `E_rho`, `q_h`. Violations should be ~0 (top-k truncation caveat).

## 6. Real speculative loop

| draft | domain | γ | α | τ | draft s | target s | vs teacher-forced τ |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | math500 | 8 | | | | | |
| 27B | math500 | 8 | | | | | |
| 2B | wikitext | 8 | | | | | |

Source: `out/loop/{domain}_{draft}_g{gamma}.json`.

## 7. Quantization control

| draft | quant | math500 q | wikitext q | E[TV] math | E[TV] wiki |
|---|---|---:|---:|---:|---:|
| 9B | BF16 | | | | |
| 9B | W8 | | | | |
| 9B | W4 | | | | |
| 35B | W4 | | | | |

Source: `out/accept/quant_*`.

## 8. Training arm (optional)

| student | tokens | math500 q | wikitext q | val token acc | val soft accept |
|---|---:|---:|---:|---:|---:|
| 2B | 50M | | | | |
| 2B | 100M | | | | |
| 4B | 50M | | | | |
| 4B | 100M | | | | |
| 4B | 200M | | | | |

Source: `out/train/*.jsonl`, `out/accept/*distill*`.

## 9. Figures

- [ ] `capacity_wall.png` panel (a): q vs D per domain
- [ ] panel (b): τ∞ vs D
- [ ] panel (c): q_inf per domain with bootstrap CI
- [ ] survival S(k) for 2B vs 27B on math500 vs wikitext
- [ ] α(γ) for 2B vs 27B on math500 vs wikitext
- [ ] quantization control q vs D (BF16/W8/W4)
- [ ] training arm q vs tokens

## 10. Limitations to state in the paper

- Target-generated sequences truncated at 512 tokens (record the truncation rate per domain).
- Off-the-shelf models differ in pretraining data; the capacity axis is not perfectly controlled.
- Top-256 target cache; residual mass included but tail approximations remain.
- WikiText is raw prose without chat formatting; MT-Bench is first-turn only.
- Single target family; replication with Qwen3.6/Qwen3.8 is in E8.
- Teacher-forcing overestimates real-loop τ at long γ (report both).
