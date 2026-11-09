"""Family statistics with non-overlapping confirmatory families.

The previous audit pooled the 36 Qwen3.5-9B raw cells into a 45-test family
that also contained the 9 Llama/K2 cells.  The Qwen cells are also the main
36-test family, so the pooled family double-counts them and the Bonferroni
level is not interpretable as an independent confirmation.

This script defines two non-overlapping confirmatory families:

* raw36: 12 compressed widths x {WikiText, FineWeb, C4} on Qwen3.5-9B;
* xfam9: 6 Llama-3.1-8B raw cells (0.95/0.98/0.99 x 2 corpora) and 3
  K2-Horizon-3.7B WikiText cells (0.95/0.98/0.99).

For each family we report (i) the per-cell sequence-cluster lower bound at
alpha/2, (ii) the minimum margin over cells, and (iii) a two-level bootstrap
that resamples corpora/models as clusters and sequences within each cell.  The
two-level interval is a dependence-robust check on the *minimum* margin; a
positive lower endpoint means every cell in the family clears q* under that
resampling scheme.

Reads existing result NPZ files only.
"""
from __future__ import annotations

import json
import os

import numpy as np

MLP_SHARE = 0.6088
PHI = {"qwen": 0.6088, "wikitext": 0.6088, "fineweb": 0.6088, "c4": 0.6088,
       "llama": 0.7511, "k2": 0.6410}
NINE_WIDTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
HIGH = [0.95, 0.98, 0.99]
QWEN = [("wikitext", "out/rank_scan/qwen3.5-9b_actwidth"),
        ("fineweb", "out/rank_scan/qwen3.5-9b_actwidth"),
        ("c4", "out/rank_scan/qwen3.5-9b_actwidth_c4")]
XFAM = [("wikitext_llama", "out/rank_scan/llama31-8b_actwidth", "llama"),
        ("fineweb_llama", "out/rank_scan/llama31-8b_actwidth", "llama"),
        ("wikitext", "out/rank_scan/k2-horizon-3.7b_actwidth_100", "k2")]


def q_star(rho: float, gamma: int = 4) -> float:
    target = 1.0 + gamma * rho

    def accepted(alpha):
        return gamma + 1 if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)

    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if accepted(1.0 - mid) >= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _boot_ci(x, seq_id, n_seq, q=0.025, B=5000, seed=0):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, x.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    vals = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))


def load_cell(setname, scan, frac, group):
    z = np.load(os.path.join(scan, f"{setname}_frac{frac:g}.npz"))
    m = z["margin_log"].astype(np.float64)
    d = z["d_margin_log"].astype(np.float64)
    match = z["match"].astype(bool)
    seq_id = z["seq_id"].astype(np.int64)
    n_seq = int(seq_id.max()) + 1
    flip = (m + d) < 0
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, flip.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    phi = PHI.get(group, MLP_SHARE)
    rho = 1.0 - phi * (1.0 - frac)
    qs = q_star(rho)
    return {
        "group": group, "set": setname, "frac": frac, "q_star": qs,
        "q": 1.0 - float(match.mean()), "q_lb_top2": float(flip.mean()),
        "n_seq": n_seq, "tokens": int(len(match)),
        "_per": per, "_cnt": cnt, "_flip": flip, "_seq_id": seq_id,
    }


def q_from_per(per, cnt, seq_idx=None, seq_w=None):
    if seq_idx is None:
        num = per.sum()
        den = cnt.sum()
    else:
        w = np.ones(len(seq_idx)) if seq_w is None else seq_w
        num = float((per[seq_idx] * w).sum())
        den = float((cnt[seq_idx] * w).sum())
    return num / max(den, 1.0)


def certify(cells, alpha_each):
    for c in cells:
        lo, hi = _boot_ci(c["_flip"], c["_seq_id"], c["n_seq"], alpha_each / 2)
        c["q_lb_lower_adj"] = lo
        c["q_lb_upper_adj"] = hi
        c["margin"] = lo - c["q_star"]
        c["certified"] = bool(lo > c["q_star"])
        for k in ("_flip", "_seq_id"):
            del c[k]
    return cells


def hierarchical_min_margin(cells, clusters, B=4000, seed=0):
    """Two-level bootstrap of the minimum margin over cells.

    clusters: list of cluster names; each cell has ``group``.  We resample
    clusters with replacement, then sequences within each sampled cell, and
    recompute q_lb for every cell.  Widths are fixed.
    """
    by_group = {}
    for c in cells:
        by_group.setdefault(c["group"], []).append(c)
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(B):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        mins_b = []
        for g in draw:
            for c in by_group.get(g, []):
                n = c["n_seq"]
                idx = rng.integers(0, n, size=n)
                q = q_from_per(c["_per"], c["_cnt"], idx)
                mins_b.append(q - c["q_star"])
        if mins_b:
            mins.append(min(mins_b))
    return {
        "min_margin_mean": float(np.mean(mins)),
        "min_margin_lower_95": float(np.percentile(mins, 2.5)),
        "min_margin_upper_95": float(np.percentile(mins, 97.5)),
        "prob_all_above": float(np.mean(np.asarray(mins) > 0)),
        "B": B,
    }


def summarize(cells, name, family_size, clusters):
    alpha = 0.05 / family_size
    certify(cells, alpha)
    for c in cells:
        c.pop("_per", None)
        c.pop("_cnt", None)
    compressed = cells
    out = {
        "name": name, "family_size": family_size, "alpha": alpha,
        "n_certified": int(sum(c["certified"] for c in cells)),
        "n_bad": int(sum((not c["certified"]) and c["q"] > c["q_star"] for c in cells)),
        "min_margin": float(min(c["margin"] for c in cells)),
        "min_margin_cell": min(cells, key=lambda c: c["margin"])["set"]
        + f"@{min(cells, key=lambda c: c['margin'])['frac']:g}",
        "cells": cells,
    }
    return out


def main():
    raw = []
    for s, scan in QWEN:
        for w in NINE_WIDTHS:
            raw.append(load_cell(s, scan, w, "qwen"))
    # group by corpus, not one pooled cluster: the two-level check resamples
    # the three corpora and the sequences within each cell.
    raw_cells = []
    for s, scan in QWEN:
        group = s
        cells = [c for c in raw if c["set"] == s]
        raw_cells.extend(cells)
    raw_out = summarize(raw_cells, "raw36", 36, ["wikitext", "fineweb", "c4"])
    # hierarchical bootstrap needs the private arrays; rebuild and run.
    raw_boot = []
    for s, scan in QWEN:
        for w in NINE_WIDTHS:
            raw_boot.append(load_cell(s, scan, w, s))
    raw_out["hierarchical"] = hierarchical_min_margin(raw_boot, ["wikitext", "fineweb", "c4"])

    xf = []
    for s, scan, group in XFAM:
        hs = HIGH
        for w in hs:
            xf.append(load_cell(s, scan, w, group))
    xfam_out = summarize(xf, "xfam9", 9, ["llama", "k2"])
    xf_boot = []
    for s, scan, group in XFAM:
        for w in HIGH:
            xf_boot.append(load_cell(s, scan, w, group))
    xfam_out["hierarchical"] = hierarchical_min_margin(xf_boot, ["llama", "k2"])

    out = {"raw36": raw_out, "xfam9": xfam_out,
           "note": "Non-overlapping families. raw36 is the main confirmatory family; "
                   "xfam9 is an independent cross-family confirmation. The two-level "
                   "bootstrap resamples corpora/models and sequences."}
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/family_stats.json", "w") as f:
        json.dump(out, f, indent=1)
    for k in ("raw36", "xfam9"):
        d = out[k]
        print(f"{k}: tests={d['family_size']} certified={d['n_certified']} "
              f"bad={d['n_bad']} alpha={d['alpha']:.5f} min_margin={d['min_margin']:+.4f} "
              f"({d['min_margin_cell']}) hier_lower={d['hierarchical']['min_margin_lower_95']:+.4f} "
              f"P(all>0)={d['hierarchical']['prob_all_above']:.3f}")


if __name__ == "__main__":
    main()
