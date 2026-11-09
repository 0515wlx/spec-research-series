# S01 claims

| ID | Layer | Strength | Status | Claim | Condition |
|---|---|---|---|---|---|
| `C-exact-selection` | theory | theorem | supported | Exact all-readout FFN width reduction is coordinate selection. | Equality for all inputs and all block weights; fixed-target redundancy is Lemma. |
| `C-fixed-weight` | theory | theorem | supported | Fixed-weight exactness is characterized by an exact identity. | Pure deletion: $W_{d,D}a_D=0$. Output-inactive channels or a functional redundancy are nongeneric escapes; we do not formalize a generic-weight quantifier. |
| `C-refit` | theory | identity | supported | Deleted channels perturb the block output, and the refitted readout has an exact decomposition. | Pure deletion is the first term; residual ridge adds the readout-correction term. The deployed certificate is measured on the composed map. |
| `C-rej-greedy` | rejection | theorem + measured | supported | Top-2/top-$K$ rejection lower bound. | Post-hoc empirical bound; requires evaluation-token logits; not a pre-deployment predictor. |
| `C-rej-sampling` | rejection | theorem + measured | supported | TV floor for exact token-level protocols. | Exact SD rule using the draft distribution; tree verification is outside. |
| `C-highwidth-floor` | rejection | measured | conditional | The near-identity resolution floor is a measured decision-boundary effect, not a tie-order or channel-ranking artifact. | Qwen3.5-9B and Qwen3-4B, WikiText raw, one deleted channel per layer (99.99% width), activation-aware ranking, bf16 target plus a float32 control. It is a mechanism diagnostic, not a deployable configuration. |
| `C-raw-no-go` | rejection | measured, confirmatory | conditional | Raw-text rejection no-go. | Nominal $\phi_{\rm MLP}$ (sensitivity in Appendix); measured distributions. Confirmatory families are 36 Qwen3.5-9B raw cells at $\alpha=0.05/36$ and 9 Llama/K2 raw cells at $\alpha=0.05/9$; the joint union bound is at most $0.10$. The 200/218/55/99/135 counts are descriptive. |
