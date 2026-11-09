# S01 — Rejection-Layer No-Go for Post-Hoc FFN Width Deletion

**Status:** `drafting` · **Version:** 0.1.0 · **Updated:** 2026-11-09

**DOI:** [10.5281/zenodo.24084215](https://doi.org/10.5281/zenodo.24084215)

[Draft PDF](draft.pdf)

## Summary

A conditional rejection no-go for activation-aware uniform FFN width deletion with a ridge readout, used as the target's own speculative draft on measured raw English prose. The rejection layer carries the conclusion; the cost layer is a secondary diagnostic.

## Abstract

We audit a no-training baseline for speculative decoding: activation-aware uniform FFN width deletion with a ridge readout, used as the target's own draft, on the evaluated raw English token population under an in-sample top-2 audit at c=1. On that scope the baseline is a conditional rejection no-go. Retaining 99% of FFN width still flips 6.5-6.8% of top-1 decisions on WikiText/FineWeb (27-28x the c=1 FLOP break-even), and the near-identity regime has a decision-boundary resolution floor (1.2-1.4% at 99.99% width; 0.31% in a clean float32 rebuild). Confirmatory evidence is 36 Qwen3.5-9B raw cells plus 9 independent Llama/K2 cells. The cost layer is an implementation-conditional diagnostic for the padded kernel, not a production-kernel claim.

## Claims

The public claim table is [claims.md](claims.md).

## Reproduce

The measurement and training pipeline is in `infra/`; the quickstart is in the repository
README. Large raw artifacts and checkpoints are maintained in the private lab repository.
