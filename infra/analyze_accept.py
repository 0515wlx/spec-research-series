"""Post-hoc analysis of accept_eval .npz files: margin certificate, survival, beta."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

PAT = re.compile(r"([a-z0-9]+)_qwen3\.5-([0-9.]+)b\.npz$")


def fit_stretched(match: np.ndarray, ks: list[int]) -> dict:
    from scipy.optimize import curve_fit

    S, K = [], []
    for k in ks:
        if len(match) < k:
            continue
        ok = np.convolve(match.astype(np.int8), np.ones(k, dtype=np.int8), "valid") == k
        s = float(ok.mean())
        if s > 1e-8:
            S.append(s); K.append(k)
    if len(S) < 4:
        return {}
    K = np.asarray(K, float); S = np.asarray(S, float)

    def f(k, lam, beta):
        return np.exp(-np.power(k / lam, beta))

    try:
        popt, _ = curve_fit(f, K, S, p0=[10.0, 0.8], bounds=([0.1, 0.05], [1e6, 2.0]), maxfev=50000)
        yhat = f(K, *popt)
        r2 = 1 - np.sum((np.log(S) - np.log(yhat)) ** 2) / np.sum((np.log(S) - np.log(S).mean()) ** 2)
        return {"lambda": float(popt[0]), "beta": float(popt[1]), "r2_log": float(r2),
                "S": {str(k): float(s) for k, s in zip(K.astype(int), S)}}
    except Exception as e:
        return {"error": repr(e)}


def analyze(path: str) -> dict:
    d = np.load(path)
    match = d["match"].astype(bool)
    tv = d["tv"].astype(float)
    rho = d["rho"].astype(float)
    margin = d["margin"].astype(float)
    n = len(match)
    q = 1 - float(match.mean())
    viol = float(((rho < 1) & (~match)).mean())
    q_h = float((rho >= 1).mean())
    # per-sequence q not needed here
    return {
        "file": path,
        "tokens": int(n),
        "q": q,
        "E_TV": float(tv.mean()),
        "E_margin": float(margin.mean()),
        "E_rho": float(rho.mean()),
        "q_h": q_h,
        "safe_frac": float((rho < 1).mean()),
        "certificate_violation": viol,
        "stretched": fit_stretched(match, [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/accept/*.npz")
    ap.add_argument("--out", default="out/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = {}
    for p in sorted(glob.glob(a.glob)):
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        dom, size = m.group(1), float(m.group(2))
        rows.setdefault(dom, {})[size] = analyze(p)

    print(f"{'domain':10s} {'D':>5s} {'q':>7s} {'E[TV]':>7s} {'q_h':>7s} {'safe':>7s} {'viol':>8s} {'beta':>6s} {'lambda':>7s}")
    for dom in sorted(rows):
        for D in sorted(rows[dom]):
            r = rows[dom][D]
            st = r.get("stretched", {})
            print(f"{dom:10s} {D:5.1f} {r['q']:7.4f} {r['E_TV']:7.4f} {r['q_h']:7.4f} "
                  f"{r['safe_frac']:7.4f} {r['certificate_violation']:8.2e} "
                  f"{st.get('beta', float('nan')):6.3f} {st.get('lambda', float('nan')):7.2f}")
    with open(os.path.join(a.out, "accept_analysis.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", os.path.join(a.out, "accept_analysis.json"))


if __name__ == "__main__":
    main()
