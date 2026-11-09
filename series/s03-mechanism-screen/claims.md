# S03 claims

| ID | Layer | Strength | Status | Claim | Condition |
|---|---|---|---|---|---|
| `C-highwidth-floor` | rejection | measured | conditional | The near-identity resolution floor is a measured decision-boundary effect, not a tie-order or channel-ranking artifact. | Qwen3.5-9B and Qwen3-4B, WikiText raw, one deleted channel per layer (99.99% width), activation-aware ranking, bf16 target plus a float32 control. It is a mechanism diagnostic, not a deployable configuration. |
| `C-mechanism-screen` | scope | exploratory | exploratory | Low-margin mechanism and deployment screen. | One target forward pass on a deployment prefix; triage only, with a mandatory direct audit below threshold. Not a distribution-free certificate. |
| `C-coverage` | theory | theorem | supported | Deployment coverage bound. | Data-only learner, uncovered deployment component; post-hoc operators see target weights. |
| `C-global-search` | rejection | measured, exploratory | exploratory | Global post-hoc stress test. | Qwen3.5-9B, measured WikiText/FineWeb cells, 30/50/90/98/99% average width; search rows are pruning-only, with a ridge-readout spot check; best-of-budget heuristic, not a proof of global optimality. Reuses the same evaluation cells; not a new confirmatory family. |
