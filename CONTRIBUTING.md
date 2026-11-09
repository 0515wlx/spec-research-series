# Contributing

This repository is a public research display layer. Issues and pull requests are welcome when they
improve reproducibility, documentation, or the audit harness.

## Ground rules

- Do not add LaTeX source, venue manifests, submission status, or review-harness material. Those
  stay in the private lab repository.
- Keep claims tied to evidence. A public claim must map to an entry in the study's `claims.md`.
- Do not add venue names or submission language anywhere in this repository.
- Large artifacts (checkpoints, raw results, processed corpora) do not belong here. See
  `docs/REPO_ARCHITECTURE.md`.
- Run `python scripts/check_public.py .` before opening a pull request. It is the public-tree gate.

## Reporting a problem

Open an issue with the command, config hash, model revision, and the smallest result file that
reproduces the problem. If the issue is about a study claim, cite the claim ID from the study's
`claims.md`.
