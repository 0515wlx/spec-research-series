# S05 claims

| ID | Layer | Strength | Status | Claim | Condition |
|---|---|---|---|---|---|
| `C-rej-greedy` | rejection | theorem + measured | supported | Top-2/top-$K$ rejection lower bound. | Post-hoc empirical bound; requires evaluation-token logits; not a pre-deployment predictor. |
| `C-rej-sampling` | rejection | theorem + measured | supported | TV floor for exact token-level protocols. | Exact SD rule using the draft distribution; tree verification is outside. |
| `C-cost-roofline` | cost | theorem | supported | Relative cost is at least the roofline ratio, uniformly in batch size and hardware. | Same-kernel fair comparison, equal efficiency, $\tau_B\ge1$, nominal FLOP/byte ratios. The bound is hardware-invariant because it uses model ratios, not hardware peaks. |
| `C-mechanism-screen` | scope | exploratory | exploratory | Low-margin mechanism and deployment screen. | One target forward pass on a deployment prefix; triage only, with a mandatory direct audit below threshold. Not a distribution-free certificate. |
