# S04 claims

| ID | Layer | Strength | Status | Claim | Condition |
|---|---|---|---|---|---|
| `C-quant` | scope | literature + measured | supported | Memory-only quantization. | Different memory-only axis; $\rho_{\rm FLOP}=1$, so the FLOP-axis audit does not cover it. |
| `C-zero-ffn` | cost | measured | conditional | Zero-MLP realized-cost audit. | Batch 1, Qwen3.5-9B, padded narrow-MLP implementation under the clean kernel-time protocol; separate from the any-batch roofline bound. The free-FFN counterfactual is a diagnostic, not a width-only kernel and not a reopening of the operator. |
| `C-serving` | cost | measured | conditional | Fair serving check. | Batch 1, 10 prompts, 64 tokens; not a wall-clock guarantee for every engine. |
| `C-matched` | scope | measured | conditional | Matched-FLOP comparison. | Matched nominal FLOP, not matched measured latency; 4B is also batch-1 cost-only no-go. |
