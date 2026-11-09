"""Audit the top-2 certificate: FWER-adjusted intervals and near-threshold cells.

Reads the certificate JSON produced by ``infra.compression_certificate`` and the
per-token NPZ files in the scan directory.  For a pre-specified family of tests
(one per width x raw corpus), it computes a Bonferroni-adjusted sequence-cluster
bootstrap interval for both q and q_lb and reports cells where the point estimate
is above the FLOP frontier but the adjusted lower bound is not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _boot_ci(x, seq_id, n_seq, q=0.025, B=4000, seed=0):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, x.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    vals = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", nargs="+", required=True)
    ap.add_argument("--scan", nargs="+", required=True)
    ap.add_argument("--tests", type=int, default=26,
                    help="number of tests in the pre-specified family")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    alpha = 0.05 / max(a.tests, 1)
    rows = []
    for cert_path, scan_dir in zip(a.cert, a.scan):
        cert = json.load(open(cert_path))
        for r in cert:
            stem = os.path.join(scan_dir, f"{r['set']}_frac{r['frac']:g}.npz")
            if not os.path.exists(stem):
                continue
            z = np.load(stem)
            m = z["margin_log"].astype(np.float64)
            d = z["d_margin_log"].astype(np.float64)
            match = z["match"].astype(bool)
            flip = (m + d) < 0
            seq_id = z["seq_id"]
            n_seq = int(seq_id.max()) + 1
            q_lo, q_hi = _boot_ci(1.0 - match, seq_id, n_seq, alpha / 2)
            l_lo, l_hi = _boot_ci(flip, seq_id, n_seq, alpha / 2)
            rows.append({
                **r,
                "family": os.path.basename(scan_dir),
                "q_fwer_ci": [q_lo, q_hi],
                "q_lb_fwer_ci": [l_lo, l_hi],
                "certified_fwer": bool(l_lo > r["q_star"]),
                "point_above_but_lb_below": bool(r["q"] > r["q_star"] and l_lo <= r["q_star"]),
                "ratio_q_lb_q": (r["q_lb_top2"] / r["q"]) if r["q"] > 0 else None,
            })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"alpha per test = {alpha:.5f} (Bonferroni over {a.tests} tests)")
    for r in rows:
        print(f"{r['family']:<28} {r['set']:<12} frac={r['frac']:<5g} "
              f"q={r['q']:.4f} q_lb={r['q_lb_top2']:.4f} q*={r['q_star']:.4f} "
              f"fwer_lb={r['q_lb_fwer_ci'][0]:.4f} certified={r['certified_fwer']}")
    n_point = sum(r["point_above_but_lb_below"] for r in rows)
    print(f"cells point-above-but-FWER-LB-below: {n_point}/{len(rows)}")


if __name__ == "__main__":
    main()
