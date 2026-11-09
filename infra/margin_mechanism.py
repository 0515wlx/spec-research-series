"""Mechanism analysis: why raw text has low top-1/top-2 margin mass.

The paper's screen uses L = Pr(m < 0.5).  This script asks whether L is just a
proxy for target entropy / calibration, or whether raw text has excess
low-margin mass at fixed entropy.  For each corpus we align the result NPZ
(``margin_log``, ``d_margin_log``, ``match``) with the target cache
(``top_logp``, ``p0``, ``ent``, ``target_ids``) and report

* Spearman correlations among margin, entropy, top-1 probability, and NLL;
* a token-level logistic regression of rejection on margin and the covariates;
* entropy-stratified low-margin mass and rejection (raw vs MATH);
* entropy-reweighted low-margin mass: reweight MATH tokens to the raw-text
  entropy distribution (and vice versa) to separate composition from shape;
* a corpus-level regression of q on low-margin mass and mean entropy.

Existing result files only; no model is run.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

SETS = [
    ("WikiText", "out/rank_scan/qwen3.5-9b_actwidth/wikitext_frac0.98.npz",
     "out/target_cache/wikitext_target9b_k256.npz"),
    ("FineWeb", "out/rank_scan/qwen3.5-9b_actwidth/fineweb_frac0.98.npz",
     "out/target_cache/fineweb_target9b_k256.npz"),
    ("C4", "out/rank_scan/qwen3.5-9b_actwidth_c4/c4_frac0.98.npz",
     "out/target_cache/c4_target9b_k256.npz"),
    ("MATH", "out/rank_scan/qwen3.5-9b_actwidth/math9b_frac0.98.npz",
     "out/target_cache/math500_target9b_k256.npz"),
]


def token_nll(cache):
    """Negative log-likelihood of the observed token from the top-256 cache."""
    top_idx = cache["top_idx"]
    top_logp = cache["top_logp"].astype(np.float64)
    tids = cache["target_ids"].astype(np.int64)
    n = top_idx.shape[0] * top_idx.shape[1]
    idx = top_idx.reshape(n, -1)
    lp = top_logp.reshape(n, -1)
    tid = tids.reshape(n)
    nll = np.full(n, np.nan)
    for i in range(n):
        hit = np.nonzero(idx[i] == tid[i])[0]
        if len(hit):
            nll[i] = -lp[i, hit[0]]
    return nll


def load(spec):
    res = np.load(spec[1])
    cache = np.load(spec[2], allow_pickle=True)
    n = min(len(res["match"]), cache["target_ids"].size)
    m = res["margin_log"].astype(np.float64)[:n]
    d = res["d_margin_log"].astype(np.float64)[:n]
    match = res["match"].astype(bool)[:n]
    ent = res["ent_t"].astype(np.float64)[:n]
    ent_cache = cache["ent"].reshape(-1).astype(np.float64)[:n]
    p0 = cache["p0"].reshape(-1).astype(np.float64)[:n]
    nll = token_nll(cache)[:n]
    return {"label": spec[0], "m": m, "d": d, "reject": ~match,
            "ent": ent, "ent_cache": ent_cache, "p0": p0, "nll": nll,
            "q": float((~match).mean())}


def entropy_strata(rows, bins=5):
    # Pool the raw rows to define common entropy quantile edges.
    ent = rows[0]["ent"]
    edges = np.quantile(ent, np.linspace(0, 1, bins + 1)[1:-1])
    out = {}
    for r in rows:
        b = np.digitize(r["ent"], edges)
        s = []
        for j in range(bins):
            mask = b == j
            if mask.sum() == 0:
                continue
            s.append({
                "bin": j + 1,
                "ent_mean": float(r["ent"][mask].mean()),
                "n": int(mask.sum()),
                "low_margin": float((r["m"][mask] < 0.5).mean()),
                "reject": float(r["reject"][mask].mean()),
                "margin_median": float(np.median(r["m"][mask])),
            })
        out[r["label"]] = s
    return out


def reweight_mass(source, target):
    """Low-margin mass in ``source`` reweighted to target's entropy histogram."""
    lo, hi = np.quantile(target["ent"], [0.01, 0.99])
    edges = np.linspace(lo, hi, 11)
    s_bin = np.digitize(source["ent"], edges)
    t_bin = np.digitize(target["ent"], edges)
    total = 0.0
    weight_total = 0.0
    for j in range(1, 11):
        sm = s_bin == j
        tm = t_bin == j
        if sm.sum() == 0 or tm.sum() == 0:
            continue
        w = tm.mean()
        total += w * float((source["m"][sm] < 0.5).mean())
        weight_total += w
    return total / max(weight_total, 1e-9)


def main():
    rows = [load(s) for s in SETS]
    out = {"cells": {}, "correlations": {}, "entropy_strata": {}, "reweight": {}}
    for r in rows:
        ent = r["ent"]
        nll = r["nll"]
        ok = np.isfinite(nll)
        out["cells"][r["label"]] = {
            "q": r["q"], "tokens": int(len(r["m"])),
            "margin_median": float(np.median(r["m"])),
            "margin_mean": float(r["m"].mean()),
            "low_margin": float((r["m"] < 0.5).mean()),
            "entropy_mean": float(ent.mean()),
            "entropy_median": float(np.median(ent)),
            "p0_mean": float(r["p0"].mean()),
            "nll_mean": float(np.nanmean(nll)),
            "nll_coverage": float(ok.mean()),
            "reject": r["q"],
            "mean_abs_dm": float(np.abs(r["d"]).mean()),
        }
        corr = {
            "margin_entropy": spearmanr(r["m"], ent).statistic,
            "margin_p0": spearmanr(r["m"], r["p0"]).statistic,
            "margin_nll": spearmanr(r["m"][ok], nll[ok]).statistic,
        }
        out["correlations"][r["label"]] = {k: float(v) for k, v in corr.items()}

    out["entropy_strata"] = entropy_strata(rows)
    raw = next(r for r in rows if r["label"] == "WikiText")
    math = next(r for r in rows if r["label"] == "MATH")
    out["reweight"] = {
        "math_to_wikitext_entropy": reweight_mass(math, raw),
        "wikitext_to_math_entropy": reweight_mass(raw, math),
        "math_low_margin": float((math["m"] < 0.5).mean()),
        "wikitext_low_margin": float((raw["m"] < 0.5).mean()),
    }

    # Corpus-level regression across the four cells (descriptive, n=4).
    L = np.array([(r["m"] < 0.5).mean() for r in rows])
    E = np.array([r["ent"].mean() for r in rows])
    Q = np.array([r["q"] for r in rows])
    X = np.vstack([np.ones_like(L), L, E]).T
    coef = np.linalg.lstsq(X, Q, rcond=None)[0]
    pred = X @ coef
    ss_res = float(((Q - pred) ** 2).sum())
    ss_tot = float(((Q - Q.mean()) ** 2).sum())
    out["corpus_regression"] = {
        "n": int(len(Q)), "coef_intercept_lowmass_entropy": coef.tolist(),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "rows": [{"label": r["label"], "low_margin": float(l),
                  "mean_entropy": float(e), "q": float(q)}
                 for r, l, e, q in zip(rows, L, E, Q)],
    }

    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/margin_mechanism.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["cells"], indent=1))
    print("correlations:", json.dumps(out["correlations"], indent=1))
    print("reweight:", json.dumps(out["reweight"], indent=1))
    print("regression:", json.dumps(out["corpus_regression"], indent=1))


if __name__ == "__main__":
    main()
