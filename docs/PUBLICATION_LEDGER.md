# Publication ledger

One row per claim or artifact that is public. This is the human-readable index that answers:
*what did we claim, on what evidence, in which public artifact, and when?*

Release `v0.1.0`: [10.5281/zenodo.24084215](https://doi.org/10.5281/zenodo.24084215)
Concept DOI: [10.5281/zenodo.24084214](https://doi.org/10.5281/zenodo.24084214)

Status values: `internal`, `public`, `submitted`, `accepted`, `superseded`.

| Claim ID | Layer | Status | Public artifact | DOI | Notes |
|---|---|---|---|---|---|
| `C-exact-selection` | theory | public | `series/S01` | 10.5281/zenodo.24084215 | Lemma `lem:select`; all-readout coordinate selection |
| `C-fixed-weight` | theory | public | `series/S01` | 10.5281/zenodo.24084215 | Lemma `lem:fixed`; fixed-weight exact identity |
| `C-refit` | theory | public | `series/S01` | 10.5281/zenodo.24084215 | Lemmas `lem:perturb`, `lem:refit`; exact decomposition |
| `C-rej-greedy` | rejection | public | `series/S01` | 10.5281/zenodo.24084215 | Top-2/top-K measured lower bound |
| `C-rej-sampling` | rejection | public | `series/S01` | 10.5281/zenodo.24084215 | Standard SD acceptance; measured `q_s` |
| `C-highwidth-floor` | mechanism | public | `series/S01` | 10.5281/zenodo.24084215 | bf16 tie floor; float32 control |
| `C-raw-no-go` | rejection | public | `series/S01` | 10.5281/zenodo.24084215 | Conditional raw-prose no-go; 36+9 confirmatory cells |
| `C-cost-roofline` | cost | public | `series/S02` | 10.5281/zenodo.24084215 | Batch-free roofline bound; preparation |
| `C-cost-taxonomy` | cost | public | `series/S02` | 10.5281/zenodo.24084215 | W/K/G/R/R-free taxonomy; preparation |
| `C-quant` | operator | public | `series/S04` | 10.5281/zenodo.24084215 | Width + INT4 combined operator; preparation |
| `C-zero-ffn` | operator | public | `series/S04` | 10.5281/zenodo.24084215 | Zero-FFN control; preparation |
| `C-serving` | serving | public | `series/S02` | 10.5281/zenodo.24084215 | Real-loop serving measurements; preparation |
| `C-matched` | cost | public | `series/S02` | 10.5281/zenodo.24084215 | Matched-latency comparison; preparation |
| `C-coverage` | deployment | public | `series/S03` | 10.5281/zenodo.24084215 | Coverage / distribution-shift diagnostic; preparation |
| `C-mechanism-screen` | mechanism | public | `series/S03` | 10.5281/zenodo.24084215 | Margin-mass screen; preparation |
| `C-global-search` | mechanism | public | `series/S03` | 10.5281/zenodo.24084215 | Best-of-budget global post-hoc stress test; preparation |

The generated claim tables in the public study pages are the authoritative wording. This ledger
tracks public availability and provenance, not the final venue decision.
