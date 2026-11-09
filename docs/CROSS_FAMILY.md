# Cross-draft-family replication: Qwen3 target/drafts

Setup: Qwen3-32B target, Qwen3-0.6B/1.7B/4B/8B/14B drafts (same Qwen3 tokenizer,
different from Qwen3.5/3.6). Raw WikiText and raw FineWeb, 100 chunks each.

## Results

| dataset | 0.6B | 1.7B | 4B | 8B | 14B | floor q_inf | bootstrap | BIC winner |
|---|---:|---:|---:|---:|---:|---:|---|---|
| WikiText raw | 0.4313 | 0.3495 | 0.2893 | 0.2617 | 0.2272 | 0.0327 | [0.0000, 0.1209] | power (BIC -53.40 vs floor -52.02) |
| FineWeb raw | 0.4068 | 0.3310 | 0.2796 | 0.2430 | 0.2007 | ~0 | [~0, ~0] | log-linear (BIC -51.99 vs floor -46.47) |

## Interpretation

The Qwen3 family does **not** replicate a positive, statistically supported floor over the
0.6B--14B range. The WikiText point estimate is 0.033 but the bootstrap CI includes zero and
the pure power law wins BIC; FineWeb is even closer to a pure power/log-linear law.

This is an important boundary condition:

- The floor is supported for the Qwen3.5/3.6 target/draft families (same tokenizer, 0.8B--27B
  drafts against 35B MoE targets).
- It is **not** universal across draft families and capacity ranges. A Qwen3-32B target with
  drafts up to 14B may simply not be close enough to the target for a floor to appear; the
  data are consistent with a pure power law.
- The paper should report this as a failed replication / scope limitation, not hide it.

If we want a stronger cross-family test, we would need a Qwen3 draft closer to the 32B target
(e.g., a 32B draft is the target itself, so not useful) or a larger Qwen3 target. The current
Qwen3 family cannot provide that capacity range.
