"""Numerical rejection certificate for a compressed draft.

For each token we know the target top-1/top-2 log-margin ``m`` and the
compressed draft's perturbation of that margin ``d`` (from the cache; the
log-normalisers cancel in the difference).  If ``m + d < 0`` the draft prefers
the target's second token over its first, so this event is a *subset* of the
greedy rejection event.  Hence

    q >= q_lb := Pr(m + d < 0).

This is a deterministic, non-vacuous lower bound computed from the same cache
as the headline q.  We compare q_lb with the cost-model break-even threshold
q_star(rho, gamma).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load_npz(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "q_per_seq"}


def percentile_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    vals = x[idx].mean(1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="directory with *_frac*.npz")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--mlp-share", type=float, default=0.6088)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from infra.cost_frontier import q_star
    rows = []
    for path in sorted(glob.glob(os.path.join(a.scan, "*_frac*.npz"))):
        name = os.path.basename(path)
        if name.startswith("compress_"):
            continue
        z = load_npz(path)
        m = z["margin_log"].astype(np.float64)
        d = z["d_margin_log"].astype(np.float64)
        match = z["match"].astype(bool)
        q = 1.0 - match.mean()
        flip_top2 = (m + d) < 0
        q_lb = float(flip_top2.mean())
        # bootstrap the lower bound by sequence
        seq_id = z["seq_id"]
        n_seq = int(seq_id.max()) + 1
        per = np.zeros(n_seq)
        cnt = np.zeros(n_seq)
        np.add.at(per, seq_id, flip_top2.astype(np.float64))
        np.add.at(cnt, seq_id, 1)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n_seq, size=(2000, n_seq))
        boot = (per[idx].sum(1)) / np.maximum(cnt[idx].sum(1), 1)
        lb_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        frac = float(name.split("_frac")[-1].replace(".npz", ""))
        rho = 1.0 - a.mlp_share * (1.0 - frac)
        qs = q_star(rho, a.gamma)
        rows.append({
            "set": name.split("_frac")[0], "frac": frac, "rho": rho,
            "q": float(q), "q_lb_top2": q_lb, "q_lb_ci": lb_ci,
            "q_star": qs, "no_go_point": bool(q > qs),
            "no_go_lower_bound": bool(lb_ci[0] > qs),
            "mean_abs_d": float(np.abs(d).mean()),
            "frac_sign_flip": float(((m + d) < 0).mean()),
            "tokens": int(len(m)),
        })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    for r in sorted(rows, key=lambda x: (x["set"], x["frac"])):
        print(f"{r['set']:<12} frac={r['frac']:<5g} rho={r['rho']:.3f} "
              f"q={r['q']:.4f} q_lb={r['q_lb_top2']:.4f} q*={r['q_star']:.4f} "
              f"no-go={r['no_go_point']} certified={r['no_go_lower_bound']}")


if __name__ == "__main__":
    main()
