# Extra generated-OOD corpora: C4, Gutenberg, code

All scans use Qwen3.5-35B-A3B target, Qwen3.5-0.8B/2B/4B/9B/27B drafts,
100 target continuations per corpus (512 scored tokens each). ModelScope sources:
`allenai/c4`, `common-pile/project_gutenberg`, `bigcode/the-stack-smol` (Python).

| corpus | q(0.8B) | q(2B) | q(4B) | q(9B) | q(27B) | q_inf | bootstrap | BIC winner |
|---|---:|---:|---:|---:|---:|---:|---|---|
| C4 (generated) | 0.0865 | 0.0751 | 0.0656 | 0.0562 | 0.0529 | 0.0373 | [0.0000, 0.0634] | floor (+1.72 over power) |
| Gutenberg (generated) | 0.1716 | 0.1413 | 0.1098 | 0.0940 | 0.0830 | 0.0542 | [0.0000, 0.0947] | floor (+1.37) |
| Code/Python (generated) | 0.0759 | 0.0617 | 0.0489 | 0.0437 | 0.0405 | 0.0337 | [0.0000, 0.0436] | floor (+4.69) |

Combined with the existing generated-OOD results:

| generated corpus | q_inf | bootstrap | BIC winner |
|---|---:|---|---|
| WikiText | 0.0000 | [0, 0] | power/log-linear |
| FineWeb | 0.0330 | [0, 0.052] | floor (+0.97) |
| C4 | 0.0373 | [0, 0.063] | floor (+1.72) |
| Gutenberg | 0.0542 | [0, 0.095] | floor (+1.37) |
| Code | 0.0337 | [0, 0.044] | floor (+4.69) |

**Interpretation.** Generated-OOD is not uniformly zero: four of five generated corpora
have a positive point estimate (0.033--0.054), and the floor model wins BIC on all four.
The bootstrap CIs include zero because there are only five capacity points, but the
qualitative pattern is not WikiText/FineWeb-specific. Generated WikiText remains the
exception: its 0.8B--9B curve is flat and the floor model loses BIC.

This strengthens the cross-corpus evidence for a small generated-OOD residual while
keeping the raw-vs-generated distinction: raw human text gives ~0.13, target-generated
text gives ~0.03--0.05.
