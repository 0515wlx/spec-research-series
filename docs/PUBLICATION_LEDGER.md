# Publication ledger

One row per claim or artifact that is public. This is the human-readable index that answers:
*what did we claim, on what evidence, in which public artifact, and when?*

Release: pending the new public series snapshot.

Status values: `internal`, `public`, `submitted`, `accepted`, `superseded`.

| Claim ID | Layer | Status | Public artifact | DOI | Notes |
|---|---|---|---|---|---|
| `C-exact-selection` | theory | public | `series/S01` | pending | Lemma `lem:select`; all-readout coordinate selection |
| `C-fixed-weight` | theory | public | `series/S01` | pending | Lemma `lem:fixed`; fixed-weight exact identity |
| `C-refit` | theory | public | `series/S01` | pending | Lemmas `lem:perturb`, `lem:refit`; exact decomposition |
| `C-rej-greedy` | rejection | public | `series/S01` | pending | Top-2/top-K measured lower bound |
| `C-rej-sampling` | rejection | public | `series/S01` | pending | Standard SD acceptance; measured `q_s` |
| `C-highwidth-floor` | mechanism | public | `series/S01` | pending | bf16 tie floor; float32 control |
| `C-raw-no-go` | rejection | public | `series/S01` | pending | Conditional raw-prose no-go; 36+9 confirmatory cells |
| `C-cost-roofline` | cost | public | `series/S02` | pending | Batch-free roofline bound; preparation |
| `C-cost-taxonomy` | cost | public | `series/S02` | pending | W/K/G/R/R-free taxonomy; preparation |
| `C-quant` | operator | public | `series/S04` | pending | Width + INT4 combined operator; preparation |
| `C-zero-ffn` | operator | public | `series/S04` | pending | Zero-FFN control; preparation |
| `C-serving` | serving | public | `series/S02` | pending | Real-loop serving measurements; preparation |
| `C-matched` | cost | public | `series/S02` | pending | Matched-latency comparison; preparation |
| `C-coverage` | deployment | public | `series/S03` | pending | Coverage / distribution-shift diagnostic; preparation |
| `C-mechanism-screen` | mechanism | public | `series/S03` | pending | Margin-mass screen; preparation |
| `C-global-search` | mechanism | public | `series/S03` | pending | Best-of-budget global post-hoc stress test; preparation |

The generated claim tables in the public study pages are the authoritative wording. This ledger
tracks public availability and provenance, not the final venue decision.
