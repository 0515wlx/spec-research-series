# Literature Map and Positioning

> Compiled 2026-11-02. Focus: what is already known, and what is left for Route A.
> Always re-check publication status before citing; several are arXiv preprints.

## 1. Direct threats (must cite and differentiate)

| # | Work | Venue / date | What it already does | Our delta |
|---|---|---|---|---|
| 1 | **When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding** — arXiv 2606.30265 | arXiv 2026-06 | Exact KL certificate for greedy/relaxed/top-m/tree acceptance; margin condition `gamma_p > sqrt(2 eps)`; Qwen3 experiments | We use the TV/margin corollary only as a diagnostic; the paper's contribution is the **capacity/OOD scaling law**, not the certificate. Cite explicitly. |
| 2 | **Scaling Laws for Speculative Decoding (Scylla)** — arXiv 2505.07858 | arXiv 2025-05 | Log-linear acceptance vs pretrain tokens, draft capacity, batch size | We test **when the log-linear law breaks under domain shift** and fit a capacity-axis asymptotic residual `q_inf(pi)` with explicit uncertainty. We do not claim it is irreducible. |
| 3 | **Speculative Decoding: Performance or Illusion?** — arXiv 2601.11580 | MLSys 2026 Oral | Real vLLM study; verification dominates; acceptance length varies; theoretical upper bound | We agree that acceptance != speed; our paper is about **why acceptance saturates on OOD**, not about speed. |
| 4 | **Variational Speculative Decoding** — arXiv/ICML 2026 | ICML 2026 | Sequence-level acceptance training objective (variational) | Our training arm is a control, not a new objective; cite as the stronger training method. |
| 5 | **Spec-AUF: Accept-Until-Fail Training** — arXiv 2607.01893 | ICML 2026 | Train-inference misalignment; accept-until-fail | Same: cite; our training arm uses CE/acceptance surrogate only to test the floor. |
| 6 | **Draft Model Knows When to Stop (SVIP)** — EMNLP 2025 | EMNLP 2025 | Draft entropy predicts acceptance; dynamic length | We treat entropy/confidence as a covariate, not a method contribution. |
| 7 | **UniVer: A Unified Perspective for Multi-step and Multi-draft SD** — arXiv 2605.04543 | NeurIPS 2026 | OT unification of multi-step/multi-draft; prefix acceptance | Our theory is a lemma + empirical scaling law, not a new verification algorithm. |
| 8 | **Training Domain Draft Models** — arXiv 2503.07807 | arXiv 2025-03 | Domain shift drops acceptance 3.7%–38.2%; offline vs online distillation | We quantify the **capacity dependence** of the residual OOD gap, not just the average drop. |
| 9 | **Online Speculative Decoding (OSD)** — arXiv 2310.07177 | 2023 | Adapt draft to query distribution; acceptance +0.1–0.65 | We explain why adaptation helps: it reduces `q_inf`, not just `q_in`. |
| 10 | **Flatter Tokens are More Valuable** — arXiv 2601.18902 | arXiv 2026-01 | Target flatness/entropy for draft data selection | We use margin/flatness as a diagnostic for `q`; no data-selection method claimed. |
| 11 | **LK Losses: Direct Acceptance Rate Optimization** — arXiv 2602.23881 | arXiv 2026-02 | Direct acceptance optimization; TV gradient | Cite for the TV/acceptance identity and the training objective. |
| 12 | **Out-of-Vocabulary Sampling / RDK** — arXiv 2506.03206 | arXiv 2025-06 | `acceptance = 1 - TV`; vocabulary pruning | Standard identity; cite. |
| 13 | **FLy: Training-Free Loosely Speculative Decoding** — arXiv 2511.22972 | ICLR 2026 | OOD robustness via relaxed/semantic acceptance; beats EAGLE-3 on OOD | Our paper is about **exact greedy acceptance** and capacity; FLy is a different acceptance rule. Related but not overlapping. |
| 14 | **Judge Decoding** — arXiv 2501.19309 | 2025 | Even GPT-4o/human text has low acceptance under exact verification; judge head improves | Supports the "exact match is the bottleneck" framing; cite in related work. |
| 15 | **A Theoretical Perspective for Speculative Decoding** — NeurIPS 2024 | NeurIPS 2024 | TV/Markov-chain analysis; expected rejection; batch algorithms | Foundational; our renewal lemma is a special case. Cite. |

## 2. Other relevant work

- **Lookahead Reasoning** (NeurIPS 2025): uses `g(gamma) = (1-alpha^{gamma+1})/((1-alpha)(1+c gamma))`
  and the `1/(1-alpha)` upper bound. Confirms `tau_inf = 1/q` is standard.
- **FastMTP** (arXiv 2509.18362): vanilla MTP acceptance collapses beyond the first draft step;
  fine-tuning fixes it. Relevant to the "generalist draft" arm.
- **PACE-dLLM** (arXiv 2609.26249): confidence-cliff dynamic block decoding. Relevant to finite-γ.
- **SpecTr / SpecTr++** (NeurIPS 2023 / 2024): optimal transport for multi-draft SD.
- **Medusa** (ICML 2024), **EAGLE / EAGLE-2 / EAGLE-3** (ICML/NeurIPS 2024–2025): draft architectures
  and tree verification; standard baselines.
- **NOT-A-BANDIT** (ICLR 2026): no-regret drafter selection; relevant to "which draft to use".
- **RepSpec** (ICLR 2027), **PRISM** (MLSys 2026), **SpecDiff-2** (MLSys 2026): recent draft
  architecture improvements; baselines if we add a method arm.
- **Survey**: *Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding*
  (arXiv 2411.13157); *From Draft-Then-Verify to Systems Co-Design* (2026 survey).

## 3. Positioning statement (one paragraph)

Prior work establishes that (i) per-token acceptance is `1 - TV` in speculative sampling and can be
certified by KL/margin conditions, (ii) acceptance improves log-linearly with draft capacity in-domain,
and (iii) domain shift lowers acceptance. What is missing is a **controlled capacity series across
domains** that separates the reducible in-domain gap from the capacity-axis residual on OOD text, and
a test of whether the residual survives quantization and training controls. We provide that, and show
that `tau_inf = 1/q` saturates at `1/q_inf(pi)` within the measured range — so "just make the draft
bigger" has a distribution-dependent ceiling. The residual is positive on raw human OOD text, zero on
target-generated OOD text, and only weakly identified with five capacity points; the practical
corollary is that coverage/generalization, not raw capacity alone, is the plausible binding constraint.

## 4. What we must not claim

- That `tau_inf = 1/q` is new.
- That the TV/margin certificate is new (arXiv 2606.30265).
- That acceptance length alone implies speedup (MLSys 2026 Oral).
- That a small generalist always beats a large specialist (it is domain- and cost-dependent).
- That W4 quantization is free (it adds a `q_q` floor; measure it).

## 5. Citation keys (BibTeX-ready)

```
@article{sharma2026draftaccepted,
  title={When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding},
  author={Sharma, Aaryam},
  journal={arXiv preprint arXiv:2606.30265},
  year={2026}
}
@article{liao2025scylla,
  title={Scaling Laws for Speculative Decoding},
  author={Liao, Xiang and others},
  journal={arXiv preprint arXiv:2505.07858},
  year={2025}
}
@inproceedings{liu2026illusion,
  title={Speculative Decoding: Performance or Illusion?},
  author={Liu, Xiaoxuan and Yu, Jiaxiang and Park, Jongseok and Stoica, Ion and Cheung, Alvin},
  booktitle={MLSys},
  year={2026}
}
@inproceedings{yin2024theoretical,
  title={A Theoretical Perspective for Speculative Decoding Algorithm},
  author={Yin, Ming and others},
  booktitle={NeurIPS},
  year={2024}
}
```

## 6. Third-revision additions (2026-11-06)

Post-hoc compression frameworks and MoE-specific pruning:

| Work | What it provides | Our use |
|---|---|---|
| SliceGPT (Ashkboos et al., ICLR 2024) | rotation-based row/column deletion | non-coordinate operator; support-selection principle explains why it mixes bases |
| ShortGPT (Men et al., NeurIPS 2024) | layer redundancy / depth pruning | depth-reduction control |
| Sheared LLaMA (Xia et al., ICLR 2024) | structured pruning with continued training | distinguishes post-hoc from trained compression |
| REAP (Lasby et al., 2025) | one-shot MoE expert pruning | MoE expert-deletion control |
| MoE-Pruner (Xie et al., 2024) | expert pruning with router-aware importance | MoE pruning context |
| T2D (Nahin et al., 2026, arXiv 2610.34103) | target-to-draft pruning + distillation, latency-aware | trained/adapted regime; orthogonal to our no-training post-hoc result |
| Zhu et al. (2024) | LLM compression survey | broad positioning |

The key distinction for this paper: expert deletion at fixed top-$k$ is memory-only, while
active-expert reduction ($k\to k'$) is the compute-reducing MoE operator. The certificate and cost
test apply to both, but only the latter enters the FLOP frontier.

## 7. Fifth-revision additions (2026-11-07)

### Self-speculative / early-exit / retrieval drafts

| Work | What it provides | Our use |
|---|---|---|
| Draft & Verify (Zhang et al., ACL 2024) | static layer-skip self-draft, no training | trained/layer-selection regime; outside post-hoc width claim |
| LayerSkip (Elhoushi et al., ACL 2024) | early-exit + cache reuse self-draft | same; the trained/adapted alternative |
| Kangaroo (Liu et al., NeurIPS 2024) | shallow sub-network + adapter | adapter is trained; outside post-hoc |
| SWIFT (Xia et al., ICLR 2025) | on-the-fly layer-set search | layer-selection, not weight compression |
| BiLD (Kim et al., NeurIPS 2023) | small model + fallback/rollback | different acceptance rule; cite |
| REST (He et al., NAACL 2024) | retrieval datastore draft | rho -> 0 limit; no target weights to compress |
| Prompt lookup (Saxena, 2023) | n-gram/context draft | rho -> 0 limit; verification-bound |

These methods are mostly in the trained/adapted or layer-selection regime, so
they do not contradict the post-hoc no-go; the paper now classifies them
explicitly instead of leaving the impression that self-speculation was untested.

### Quantization axis

| Work | What it provides | Our use |
|---|---|---|
| QSpec (Zhao et al., EMNLP 2025) | W4A4 draft + W4A16 verify, up to 1.64x | quantization+SD is a successful orthogonal axis |
| QuantSpec (Tiwari et al., ICML 2025) | 4-bit weights + hierarchical 4-bit KV, >90% acceptance, up to 2.5x | same; long-context positive result |
| ML-SpecQD (Georganas et al., 2025) | MXFP4 WOQ draft, 80-90% acceptance, up to 2.72x | same; consistent with our W4 86-96% top-1 acceptance |
| Quasar (Huang & Wen, 2026) | W8A8 quantized verifier; acceptance preserved | consistent with our near-lossless W8; orthogonal to drafting |
| SD Meets Quantization (Zhang et al., 2025) | W4A16 hierarchical framework, 2.78x on W4 Llama-3-70B | same; quantization is a successful path |
| Why Does PTQ Work? (Chen et al., 2026) | top-1 flip rate Flip@1; LM-head geometry | W4 top-1 flip benchmark; connects to our q |
| QEP (Arai & Ichikawa, NeurIPS 2025) | Lipschitz output-error propagation bound | gives a training-valid bound on quantization-induced margin perturbation |
| QGen (AskariHemmat et al., 2024) | QAT generalization bound | loss/generalization, not top-1 agreement with a fixed teacher |
| Compute-Optimal QAT (Dremov et al., ICLR 2026) | QAT loss scaling law | QAT is a trained operator; outside post-hoc |

**Decision:** weight-only quantization is a memory-only axis. It passes the
memory-roofline break-even (W4 q=0.039-0.156 vs q*_mem=0.31-0.38) but our packed
INT4/INT8 kernels are not faster than bf16 at batch 1 (rho_1=0.99-1.01 for W4,
2.11 for W8). This is a kernel-dependent cost-only result, not a structural
no-go. The paper excludes it from the compute-reducing claim and reports it as
a boundary.
