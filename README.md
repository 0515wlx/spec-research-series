# Speculative Decoding Audit Series

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24084215.svg)](https://doi.org/10.5281/zenodo.24084215)

A public series of reproducible audits on speculative decoding: where the acceptance floor comes
from, which compression operators clear it, and what the measured serving cost actually is.

This repository is the **display layer** of an ongoing research series. It publishes neutral draft
PDFs, public claims, code, and sanitized results. The LaTeX source, the review harness, and the
submission workflow are maintained privately.

## Studies

| ID | Study | Status | Version | Draft | DOI |
|---|---|---|---|---|---|
| S01 | Rejection-layer no-go for post-hoc FFN width deletion | `drafting` | 0.1.0 | [PDF](series/s01-rejection-no-go/draft.pdf) | [10.5281/zenodo.24084215](https://doi.org/10.5281/zenodo.24084215) |
| S02 | Measured cost of post-hoc FFN width deletion | `preparation` | — | — | — |
| S03 | Decision-boundary mechanism and margin-mass screening | `preparation` | — | — | — |
| S04 | Positive operators, adaptation, matched-latency drafts | `preparation` | — | — | — |
| S05 | Reproducible acceptance/cost benchmark and leaderboard | `preparation` | — | — | — |

The full index and status vocabulary are in [`series/README.md`](series/README.md).

### Status vocabulary

- **preparation** — scope and claims are still being established.
- **drafting** — core claim is closed; manuscript is in progress.
- **stable** — frozen public draft.
- **revision** — revised draft in progress.
- **archived** — retired or superseded.

## Latest draft

S01: *Rejection-Layer No-Go for Post-Hoc FFN Width Deletion* —
[PDF](series/s01-rejection-no-go/draft.pdf) · [claims](series/s01-rejection-no-go/claims.md)

The rejection layer carries the conclusion: activation-aware uniform FFN width deletion with a
ridge readout does not clear the speculative-decoding acceptance frontier on the evaluated raw
English token population under an in-sample top-2 audit at `c=1`. The cost layer is a secondary
implementation diagnostic, not a production-kernel claim.

## What is here

- `series/` — study index, neutral draft PDFs, public claims.
- `infra/` — reproducible acceptance / distribution-shift / cost pipeline.
- `code/` — legacy exploratory scripts, kept for provenance.
- `configs/` — model and experiment configs.
- `docs/` — experiment plans, infrastructure notes, results, literature map.
- `artifacts/` — small result summaries and leaderboard data.
- `site/` — static project page.

## Reproduce

```bash
python -m infra.env_check

python -m infra.download --models qwen3.5-0.8b qwen3.5-2b qwen3.5-27b qwen3.5-35b-a3b

python -m infra.generate_target --target qwen3.5-35b-a3b --domain math500 \
  --limit 500 --max-new 512 --out data/processed/math500_target.jsonl

python -m infra.accept_eval --target qwen3.5-35b-a3b --draft qwen3.5-2b \
  --data data/processed/math500_target.jsonl --out out/accept/math500_qwen3.5-2b.json
```

Large artifacts (`out/`, `ckpt/`, `logs/`, raw data) are not in this repository. The public/private
split is documented in `docs/REPO_ARCHITECTURE.md`.

## Citation

If you use this code or the audit results, cite the series and the specific study. See
`CITATION.cff`.

- Version DOI (`v0.1.0`): [10.5281/zenodo.24084215](https://doi.org/10.5281/zenodo.24084215)
- Concept DOI (all versions): [10.5281/zenodo.24084214](https://doi.org/10.5281/zenodo.24084214)

Use the version DOI when the claim depends on this exact snapshot; use the concept DOI when
referring to the series across versions.

## License

- Code: Apache-2.0 (`LICENSE`)
- Docs and draft PDFs: CC-BY-4.0 (`LICENSE-CC-BY-4.0`)

## Contact

Lixuan Wei, Institute of Automation, Chinese Academy of Sciences.
