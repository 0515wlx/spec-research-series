"""Recompute family-wise certificate audits with the correct family composition.

The fourth-round audit grouped the 9B scan by directory, which mixed MATH into
the ``English raw'' family.  This script defines the families explicitly:

* raw39: 13 widths x {WikiText, FineWeb, C4}, one Bonferroni family;
* math13: 13 widths x MATH-500, a separate in-domain family;
* xfam45: 36 compressed 9B raw cells + 6 Llama + 3 K2, the cross-family family.

For every cell we recompute the sequence-cluster bootstrap interval at the
family-adjusted level and report certified / bad / identity counts.
"""
from __future__ import annotations

import json
import os

import numpy as np

WIDTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
HIGH = [0.95, 0.98, 0.99]
MLP_SHARE = 0.6088


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


def _boot_ci(x, seq_id, n_seq, q=0.025, B=4000, seed=0):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, x.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    vals = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))


def cell(setname, scan, frac, qs):
    z = np.load(os.path.join(scan, f"{setname}_frac{frac:g}.npz"))
    m = z["margin_log"].astype(np.float64)
    d = z["d_margin_log"].astype(np.float64)
    match = z["match"].astype(bool)
    seq_id = z["seq_id"]
    n_seq = int(seq_id.max()) + 1
    flip = (m + d) < 0
    return {
        "set": setname, "scan": scan, "frac": frac, "q_star": qs,
        "q": 1.0 - float(match.mean()), "q_lb_top2": float(flip.mean()),
        "tokens": int(len(match)), "seqs": n_seq,
        "_flip": flip, "_seq_id": seq_id, "_n_seq": n_seq,
    }


def finalize(cells, family_size, identity_widths=(1.0,)):
    alpha = 0.05 / max(family_size, 1)
    for c in cells:
        lo, _ = _boot_ci(c["_flip"], c["_seq_id"], c["_n_seq"], alpha / 2)
        c["q_lb_adj_lower"] = lo
        c["certified"] = bool(lo > c["q_star"])
        c["identity"] = bool(c["frac"] in identity_widths)
        c["bad"] = bool((not c["identity"]) and (c["q"] > c["q_star"])
                        and (lo <= c["q_star"]))
        del c["_flip"], c["_seq_id"], c["_n_seq"]
    compressed = [c for c in cells if not c["identity"]]
    return {
        "family_size": family_size,
        "alpha": alpha,
        "cells": cells,
        "n_cells": len(cells),
        "n_compressed": len(compressed),
        "n_certified": int(sum(c["certified"] for c in compressed)),
        "n_bad": int(sum(c["bad"] for c in compressed)),
        "n_identity": int(sum(c["identity"] for c in cells)),
    }


def main():
    raw = []
    for s, scan in [("wikitext", "out/rank_scan/qwen3.5-9b_actwidth"),
                    ("fineweb", "out/rank_scan/qwen3.5-9b_actwidth"),
                    ("c4", "out/rank_scan/qwen3.5-9b_actwidth_c4")]:
        for w in WIDTHS:
            rho = 1.0 - MLP_SHARE * (1.0 - w)
            raw.append(cell(s, scan, w, q_star(rho)))
    raw39 = finalize(raw, 39)

    math = []
    for w in WIDTHS:
        rho = 1.0 - MLP_SHARE * (1.0 - w)
        math.append(cell("math9b", "out/rank_scan/qwen3.5-9b_actwidth",
                         w, q_star(rho)))
    math13 = finalize(math, 13)

    cert_llama = {r["set"] + f"_{r['frac']:g}": r for r in
                  json.load(open("out/rank_scan/certificate_llama31_8b.json"))}
    cert_k2 = {r["set"] + f"_{r['frac']:g}": r for r in
               json.load(open("out/rank_scan/certificate_k2_100.json"))}
    xfam = []
    for s, scan in [("wikitext", "out/rank_scan/qwen3.5-9b_actwidth"),
                    ("fineweb", "out/rank_scan/qwen3.5-9b_actwidth"),
                    ("c4", "out/rank_scan/qwen3.5-9b_actwidth_c4")]:
        for w in WIDTHS[:-1]:
            rho = 1.0 - MLP_SHARE * (1.0 - w)
            xfam.append(cell(s, scan, w, q_star(rho)))
    for s, scan in [("wikitext_llama", "out/rank_scan/llama31-8b_actwidth"),
                    ("fineweb_llama", "out/rank_scan/llama31-8b_actwidth")]:
        for w in HIGH:
            key = s + f"_{w:g}"
            xfam.append(cell(s, scan, w, cert_llama[key]["q_star"]))
    for w in HIGH:
        key = "wikitext_" + f"{w:g}"
        xfam.append(cell("wikitext", "out/rank_scan/k2-horizon-3.7b_actwidth_100",
                         w, cert_k2[key]["q_star"]))
    xfam45 = finalize(xfam, len(xfam), identity_widths=())

    out = {"raw39": raw39, "math13": math13, "xfam45": xfam45}
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/confirm_family.json", "w") as f:
        json.dump(out, f, indent=1)
    for name, d in out.items():
        print(f"{name}: cells={d['n_cells']} compressed={d['n_compressed']} "
              f"certified={d['n_certified']} bad={d['n_bad']} "
              f"identity={d['n_identity']} alpha={d['alpha']:.5f}")


if __name__ == "__main__":
    main()
