# S02 — Measured Cost of Post-Hoc FFN Width Deletion

**Status:** `preparation` · **Version:** 0.0.0 · **Updated:** 2026-11-09

No public draft yet. This study is in preparation.

## Summary

The cost layer of the FFN width-deletion audit: roofline bound, padded narrow-MLP implementation, zero-FFN control, quantization, and the gap to a production kernel.

## Abstract

A separate study of the measured cost layer. The current evidence is implementation-conditional: the padded narrow-MLP point estimate is an upper bound, the zero-FFN control can fall below the c=1 frontier, and several K-cell labels are point estimates. This study is not closed.

## Claims

The public claim table is [claims.md](claims.md).

## Reproduce

The measurement and training pipeline is in `infra/`; the quickstart is in the repository
README. Large raw artifacts and checkpoints are maintained in the private lab repository.
