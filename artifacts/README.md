# Artifacts

This directory holds **small, citable result summaries** that are safe to publish. It is not a
mirror of `out/`.

- `leaderboard.json` — aggregate acceptance / cost / certification results used by the project page.
- `manifest.json` — the artifact index: source script, config hash, model revision, date, claim IDs.
- `summaries/` — per-experiment JSON with the fields needed to reproduce a table or figure.

Rules:

- No checkpoints, no raw tensors, no full evaluation dumps, no corpora.
- Every file must name the generator script and the claim IDs it supports.
- Keep files small enough to fit in a Zenodo release without a special transfer.
- The authoritative raw results stay in the private lab repository under `out/`.

The public leaderboard is intentionally a summary. If a claim needs the full raw data to be audited,
the release should include a manifest and a request path, not a 10 GB upload.
