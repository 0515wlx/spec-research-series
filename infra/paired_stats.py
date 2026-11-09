"""Paired sequence bootstrap and TOST for two drafts on the same tokens.

Both inputs are ``.npz`` files written by ``infra.accept_eval`` / ``infra.rank_scan``.
They must contain ``match`` and ``seq_id`` for the same sequences in the same order.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def load(path):
    z = np.load(path)
    return z["match"].astype(bool), z["seq_id"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="TOST equivalence margin on the q difference")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ma, sa = load(a.a)
    mb, sb = load(a.b)
    if len(ma) != len(mb) or not np.array_equal(sa, sb):
        raise SystemExit("the two files do not describe the same token stream")
    n_seq = int(sa.max()) + 1
    # per-sequence rejection counts
    ca = np.zeros(n_seq); cb = np.zeros(n_seq); cn = np.zeros(n_seq)
    np.add.at(ca, sa, (~ma).astype(np.float64))
    np.add.at(cb, sb, (~mb).astype(np.float64))
    np.add.at(cn, sa, 1.0)
    qa = ca.sum() / cn.sum()
    qb = cb.sum() / cn.sum()
    d = qa - qb
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_seq, size=(a.bootstrap, n_seq))
    boot = (ca[idx].sum(1) - cb[idx].sum(1)) / np.maximum(cn[idx].sum(1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # TOST: two one-sided tests against +/- margin, normal approximation on the
    # bootstrap distribution.
    se = boot.std(ddof=1)
    z = 1.96
    p_lower = 1.0 - 0.5 * (1 + math.erf(((d + a.margin) / max(se, 1e-12)) / np.sqrt(2)))
    p_upper = 0.5 * (1 + math.erf(((d - a.margin) / max(se, 1e-12)) / np.sqrt(2)))
    p_tost = max(p_lower, p_upper)
    out = {
        "label_a": a.label_a, "label_b": a.label_b,
        "q_a": float(qa), "q_b": float(qb), "delta_a_minus_b": float(d),
        "delta_ci95": [float(lo), float(hi)],
        "tost_margin": a.margin, "tost_p": float(p_tost),
        "equivalent": bool(p_tost < 0.05),
        "tokens": int(len(ma)), "seqs": n_seq,
    }
    print(json.dumps(out, indent=1))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
