# Priority and public timestamps

The goal of the public release is to make priority **verifiable and citable**, not merely visible.
GitHub commit timestamps are useful for the lab notebook but are weak evidence in an academic
priority dispute. A Zenodo DOI on an immutable release is the citable object.

## What counts as a priority timestamp

1. **Zenodo version DOI** for an immutable release tag. This is the primary timestamp.
2. **Zenodo concept DOI** for the project as a whole (all versions).
3. **GitHub Release** page for the same tag, with the PDFs attached.
4. **Software Heritage** archive (optional) for a content-addressed SWHID.
5. **Public paper PDF** hosted in the release, with a version and date on the title page.

A commit, a branch, or a README edit is not a priority timestamp by itself.

## Release rules

- One release per meaningful result bundle, not per commit.
- Tag format: `vMAJOR.MINOR.PATCH` for the repository; `paper-<venue>-vX` for a frozen paper
  snapshot; `bench-vX` for a leaderboard/data release.
- Releases are immutable. Correct a mistake with a new patch release, not by moving a tag.
- Every release gets a row in `docs/PUBLICATION_LEDGER.md`.
- The release must be buildable from the tag alone: `make -C paper check` must pass.

## What to put in a release

- final PDFs,
- the exact source tree used to build them,
- `paper/meta/claims.yaml` and `evidence.yaml`,
- small result summaries and leaderboard JSON,
- a `CHANGELOG.md` entry describing what changed and what did not.

## What not to put in a release

- raw results, checkpoints, logs, or processed corpora,
- internal review rounds or adversarial notes,
- credentials, local paths, or third-party data without a clear license.

## Citing a release

Use the **version DOI** when the claim depends on the exact snapshot. Use the **concept DOI** when
referring to the project across versions. Record both in `CITATION.cff` and the paper's artifact
statement after acceptance.
