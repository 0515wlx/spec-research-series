"""Cross-family confirmatory family with a single Bonferroni correction.

The review asks whether the no-go survives if a non-Qwen family is promoted from
exploratory to confirmatory status.  We pre-specify one family:

* Qwen3.5-9B English raw text: 12 compressed widths x 3 corpora = 36 cells;
* Llama-3.1-8B raw text: 3 high widths (0.95/0.98/0.99) x 2 corpora = 6 cells;
* K2-Horizon-3.7B raw text: 3 high widths x WikiText = 3 cells;

for 45 tests.  The adjusted two-sided level is 0.05/45.  We compute the
sequence-cluster bootstrap interval for q_lb = Pr(m + Delta_m < 0) directly from
the per-token NPZ files and compare its lower endpoint with the FLOP break-even.

This script reads existing result files only.
"""
from __future__ import annotations

import json
import os

import numpy as np

NINE = [
    ("wikitext", "out/rank_scan/qwen3.5-9b_actwidth", "certificate_actwidth_9b.json"),
    ("fineweb", "out/rank_scan/qwen3.5-9b_actwidth", "certificate_actwidth_9b.json"),
    ("c4", "out/rank_scan/qwen3.5-9b_actwidth_c4", "certificate_c4_9b.json"),
]
LLAMA = [
    ("wikitext_llama", "out/rank_scan/llama31-8b_actwidth"),
    ("fineweb_llama", "out/rank_scan/llama31-8b_actwidth"),
]
K2 = [
    ("wikitext", "out/rank_scan/k2-horizon-3.7b_actwidth_100"),
]
NINE_WIDTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
HIGH_WIDTHS = [0.95, 0.98, 0.99]


def _boot_ci(x, seq_id, n_seq, q=0.025, B=4000, seed=0):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, x.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    vals = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))


def _cert_index(path):
    rows = json.load(open(path))
    return {(r["set"], float(r["frac"])): r for r in rows}


def _cell(setname, scan, frac, q_star):
    z = np.load(os.path.join(scan, f"{setname}_frac{frac:g}.npz"))
    m = z["margin_log"].astype(np.float64)
    d = z["d_margin_log"].astype(np.float64)
    match = z["match"].astype(bool)
    seq_id = z["seq_id"]
    n_seq = int(seq_id.max()) + 1
    flip = (m + d) < 0
    q = 1.0 - float(match.mean())
    q_lb = float(flip.mean())
    lo, hi = _boot_ci(flip, seq_id, n_seq)
    return {
        "set": setname, "frac": frac, "seqs": n_seq, "scan": scan,
        "tokens": int(len(match)), "q": q, "q_lb_top2": q_lb,
        "q_lb_ci_family": [lo, hi], "q_star": q_star,
        "certified_family": bool(lo > q_star),
    }


def main():
    cells = []
    cert9 = {p: _cert_index(os.path.join("out/rank_scan", p)) for p in
             {"certificate_actwidth_9b.json", "certificate_c4_9b.json"}}
    for setname, scan, certfile in NINE:
        cert = cert9[certfile]
        for w in NINE_WIDTHS:
            key = (setname, w)
            if key not in cert:
                # C4 key is "c4"; the other two are wikitext/fineweb
                continue
            cells.append(_cell(setname, scan, w, cert[key]["q_star"]))
    cert_llama = _cert_index("out/rank_scan/certificate_llama31_8b.json")
    for setname, scan in LLAMA:
        for w in HIGH_WIDTHS:
            key = (setname, w)
            if key in cert_llama:
                cells.append(_cell(setname, scan, w, cert_llama[key]["q_star"]))
    cert_k2 = _cert_index("out/rank_scan/certificate_k2_100.json")
    for setname, scan in K2:
        for w in HIGH_WIDTHS:
            key = (setname, w)
            if key in cert_k2:
                cells.append(_cell(setname, scan, w, cert_k2[key]["q_star"]))

    family_size = len(cells)
    alpha = 0.05 / max(family_size, 1)
    # Recompute the CI at the family-adjusted level and certify.
    for c in cells:
        z = np.load(os.path.join(c["scan"], f"{c['set']}_frac{c['frac']:g}.npz"))
        m = z["margin_log"].astype(np.float64)
        d = z["d_margin_log"].astype(np.float64)
        seq_id = z["seq_id"]
        n_seq = int(seq_id.max()) + 1
        flip = (m + d) < 0
        lo, hi = _boot_ci(flip, seq_id, n_seq, alpha / 2)
        c["q_lb_ci_bonferroni"] = [lo, hi]
        c["certified_bonferroni"] = bool(lo > c["q_star"])
        del c["scan"]

    out = {
        "family_size": family_size,
        "bonferroni_alpha_two_sided": alpha,
        "bonferroni_level_per_interval": 1.0 - alpha,
        "cells": cells,
        "n_certified": int(sum(c["certified_bonferroni"] for c in cells)),
    }
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/cross_family_confirm.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"family_size={family_size} alpha={alpha:.6f} "
          f"level={1-alpha:.6f}")
    for c in cells:
        print(f"{c['set']:<15} {c['frac']:<5g} q={c['q']:.4f} "
              f"lb={c['q_lb_top2']:.4f} bonf_lo={c['q_lb_ci_bonferroni'][0]:.4f} "
              f"q*={c['q_star']:.4f} cert={c['certified_bonferroni']}")
    print(f"certified {out['n_certified']}/{family_size}")


if __name__ == "__main__":
    main()
