# S02 claims

| ID | Layer | Strength | Status | Claim | Condition |
|---|---|---|---|---|---|
| `C-cost-roofline` | cost | theorem | supported | Relative cost is at least the roofline ratio, uniformly in batch size and hardware. | Same-kernel fair comparison, equal efficiency, $\tau_B\ge1$, nominal FLOP/byte ratios. The bound is hardware-invariant because it uses model ratios, not hardware peaks. |
| `C-cost-taxonomy` | cost | theorem + measured | conditional | Cost verdicts are ordered: W, K, G, R, with R-free as a stronger sub-label. | Measured clean kernel-time cost for W/K; nominal FLOP and byte ratios for G/R; measured time share for the free-FFN counterfactual. The cost layer is load-bearing only for K and G; raw 98--99% cells are R by 13--20x. |
| `C-quant` | scope | literature + measured | supported | Memory-only quantization. | Different memory-only axis; $\rho_{\rm FLOP}=1$, so the FLOP-axis audit does not cover it. |
| `C-serving` | cost | measured | conditional | Fair serving check. | Batch 1, 10 prompts, 64 tokens; not a wall-clock guarantee for every engine. |
| `C-matched` | scope | measured | conditional | Matched-FLOP comparison. | Matched nominal FLOP, not matched measured latency; 4B is also batch-1 cost-only no-go. |
