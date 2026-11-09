# Public / private repository architecture

This project separates the public display layer from the private research harness.

## Public repository: `spec-research`

The public repository is the **series display and artifact surface**:

- study index and neutral draft PDFs,
- public claim tables,
- reproducible pipeline (`infra/`),
- legacy exploratory code (`code/`, provenance),
- sanitized experiment plans, infrastructure notes, and results,
- small result summaries,
- static project page.

It does **not** contain:

- LaTeX source or paper build manifests,
- venue manifests or submission status,
- mock-review personas, rubrics, or rounds,
- raw results, checkpoints, logs, or processed corpora.

The public tree is generated from the private lab repository by a whitelist export, not by a direct
`git push --all`. The export also runs a public-tree check that fails if a venue name, submission
term, or private registry key appears.

## Private repository: `spec-research-lab`

The private repository is the **lab notebook and harness**:

- full paper source and all venue builds,
- the study registry with target venues and submission status,
- the mock-review harness and review rounds,
- raw results, checkpoints, logs, and processed data,
- internal notes and review action lists.

Large directories remain gitignored even in the private repository; the private remote is a backup
of the source and small artifacts, not of the checkpoint tree.

## Release flow

1. Build the neutral draft in the private repository.
2. Generate the public series artifacts.
3. Export the whitelisted public tree.
4. Run the public-tree leak check.
5. Commit, tag, and release the public repository.
6. Zenodo archives the tagged release and mints a version DOI.

## Why two repositories

- A single public repository would leak venue targeting, review harness material, and submission
  status.
- A single private repository would forfeit the priority timestamp, the citable DOI, and the
  reproducibility surface that readers and reviewers expect.
- The split keeps the public artifact small, buildable, and safe to archive, while the lab keeps the
  full experimental record.
