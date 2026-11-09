"""Mechanism analysis: target margin, rejection, and perturbation by margin strata.

For a compressed-draft result NPZ we compute, per evaluation set:

* target top-1/top-2 log-margin quantiles;
* the fraction of tokens below absolute margin thresholds;
* rejection rate and mean |Delta_m| in the lowest-margin quintile;
* q_lb / q, the fraction of rejections explained by the top-2 event.

The same script can compare a pruned result with a reconstructed result; the latter
has the same NPZ schema (``margin_log``, ``d_margin_log``, ``match``).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def analyze(path: str) -> dict:
    z = np.load(path)
    m = z["margin_log"].astype(np.float64)
    d = z["d_margin_log"].astype(np.float64)
    match = z["match"].astype(bool)
    q = 1.0 - float(match.mean())
    flip = (m + d) < 0
    q_lb = float(flip.mean())
    qs = np.quantile(m, [0.2, 0.4, 0.6, 0.8])
    low = m <= qs[0]
    # five margin quintiles (lowest first)
    bins = np.digitize(m, qs)
    strata = []
    for b in range(5):
        mask = bins == b
        if mask.sum() == 0:
            continue
        strata.append({
            "quintile": b + 1,
            "n": int(mask.sum()),
            "margin_mean": float(m[mask].mean()),
            "reject": float(1.0 - match[mask].mean()),
            "mean_abs_d": float(np.abs(d[mask]).mean()),
        })
    return {
        "path": path,
        "tokens": int(len(m)),
        "q": q,
        "q_lb": q_lb,
        "q_lb_over_q": (q_lb / q) if q > 0 else None,
        "margin_median": float(np.median(m)),
        "margin_q10": float(np.quantile(m, 0.1)),
        "margin_q25": float(np.quantile(m, 0.25)),
        "frac_m_lt_0.1": float((m < 0.1).mean()),
        "frac_m_lt_0.5": float((m < 0.5).mean()),
        "frac_m_lt_1.0": float((m < 1.0).mean()),
        "low_quintile_reject": float(1.0 - match[low].mean()),
        "low_quintile_abs_d": float(np.abs(d[low]).mean()),
        "overall_abs_d": float(np.abs(d).mean()),
        "strata": strata,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="label:set:path entries; path is a result .npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.spec:
        label, setname, path = s.split(":", 2)
        r = analyze(path)
        r.update({"label": label, "set": setname})
        rows.append(r)
        print(f"{label:<10} {setname:<10} q={r['q']:.4f} q_lb={r['q_lb']:.4f} "
              f"median_m={r['margin_median']:.3f} lowq_rej={r['low_quintile_reject']:.4f} "
              f"lowq_|d|={r['low_quintile_abs_d']:.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
