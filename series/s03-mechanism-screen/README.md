# S03 — Decision-Boundary Mechanism and Margin-Mass Screening

**Status:** `preparation` · **Version:** 0.0.0 · **Updated:** 2026-11-09

No public draft yet. This study is in preparation.

## Summary

The mechanism layer: near-boundary resolution floors, bf16 tie atoms, float32 controls, and the margin-mass screen as a triage diagnostic rather than a certificate.

## Abstract

A separate study of why the rejection floor exists and whether a cheap screen can predict it. The mechanism is strong enough for a diagnostic; the screen has asymmetric false negatives and is not a certificate. This study is not closed.

## Claims

The public claim table is [claims.md](claims.md).

## Reproduce

The measurement and training pipeline is in `infra/`; the quickstart is in the repository
README. Large raw artifacts and checkpoints are maintained in the private lab repository.
