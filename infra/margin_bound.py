"""Numerically evaluate the margin--TV bound (Lemma 1 / Eq. 8) from existing caches.

For each evaluation set we

  * estimate the target margin CDF  M(eps) = Pr(delta <= 2 eps)  directly,
  * fit the power-law tail          M(eps) <= c eps^gamma  on a small-margin window,
  * read E[TV] for every draft from the cached .npz files,
  * evaluate the *empirical* bound  min_eps [ M(eps) + E[TV]/eps ]
    and the *parametric* bound       K * E[TV]^(gamma/(gamma+1)),
  * compare both with the observed greedy rejection rate q,
  * report the exponent predicted by the square-root law
    alpha_pred = beta * gamma/(gamma+1), where E[TV] ~ D^-beta.

This is a numerical check of the certificate, not a new theorem.  It is CPU-only
and reads the same caches as the paper.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

OUT = "out/margin_bound"
os.makedirs(OUT, exist_ok=True)

DRAFTS = [
    ("qwen3.5-0.8b", 0.8),
    ("qwen3.5-2b", 2.0),
    ("qwen3.5-4b", 4.0),
    ("qwen3.5-9b", 9.0),
    ("qwen3.5-27b", 27.0),
]
# (label, accept prefix, accept directory)
SETS = [
    ("MATH-500", "math500_", "out/accept"),
    ("GSM8K", "gsm8k_", "out/accept"),
    ("HumanEval", "humaneval_", "out/accept"),
    ("MT-Bench", "mtbench_", "out/accept"),
    ("WikiText raw", "wikitext_", "out/accept"),
    ("FineWeb raw", "fineweb_", "out/accept"),
    ("WikiText gen", "wikitext_gen_", "out/accept"),
    ("FineWeb gen", "fineweb_gen_", "out/accept"),
]


def load_domain(prefix, adir):
    rows = {}
    for name, D in DRAFTS:
        jp = os.path.join(adir, f"{prefix}{name}.json")
        zp = jp.replace(".json", ".npz")
        if not (os.path.exists(jp) and os.path.exists(zp)):
            continue
        j = json.load(open(jp))
        z = np.load(zp, allow_pickle=True)
        rows[name] = {
            "D": D,
            "q": float(j["q"]),
            "E_TV": float(j["E_TV"]),
            "margin": z["margin"].astype(np.float64),
            "tv": z["tv"].astype(np.float64),
        }
    return rows


def fit_margin_tail(margin, lo_mass=0.005, hi_mass=0.25):
    """Fit log M(eps) = log c + gamma log eps on a small-margin mass window.

    ``margin`` is delta; the bound uses eps = delta/2.
    """
    x = np.sort(margin) / 2.0
    n = len(x)
    # Empirical CDF mass points inside the tail window.
    q = np.arange(1, n + 1, dtype=float) / n
    keep = (q >= lo_mass) & (q <= hi_mass) & (x > 0)
    xk, qk = x[keep], q[keep]
    if len(xk) < 8:
        return None
    A = np.column_stack([np.ones_like(xk), np.log(xk)])
    coef, *_ = np.linalg.lstsq(A, np.log(qk), rcond=None)
    gamma = float(coef[1])
    c = float(np.exp(coef[0]))
    # R^2 in log space
    pred = A @ coef
    ss_res = float(np.sum((np.log(qk) - pred) ** 2))
    ss_tot = float(np.sum((np.log(qk) - np.log(qk).mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    # Smallest eps at which the empirical bound can still be evaluated.
    return {"c": c, "gamma": gamma, "r2": r2,
            "eps_min": float(xk[0]), "eps_max": float(xk[-1]),
            "mass_lo": float(lo_mass), "mass_hi": float(hi_mass),
            "n_fit": int(len(xk))}


def empirical_bound(margin, tv_mean, n_grid=400):
    """min_eps [ M(eps) + E[TV]/eps ] using the empirical margin CDF."""
    x = np.sort(margin) / 2.0
    n = len(x)
    # Candidate eps: quantiles of margin/2, plus a geometric range.
    qs = np.linspace(1.0 / n, 0.5, n_grid)
    cand = np.quantile(x, qs)
    cand = cand[cand > 0]
    # Include points between order statistics so M is flat there.
    if len(x) > 1:
        mids = np.sqrt(x[:-1] * x[1:])
        cand = np.unique(np.concatenate([cand, mids]))
    M = np.searchsorted(x, cand, side="right") / n
    vals = M + tv_mean / cand
    k = int(np.argmin(vals))
    return {
        "bound": float(vals[k]),
        "eps_star": float(cand[k]),
        "M_at_star": float(M[k]),
        "tv_over_eps": float(tv_mean / cand[k]),
        "nontrivial_frac": float(np.mean(vals < 1.0)),
        "min_bound_lt_1": bool(vals[k] < 1.0),
    }


def power_bound(c, gamma, tv_mean):
    """Parametric optimum of c eps^gamma + eps_D/eps."""
    # d/deps: c gamma eps^(gamma-1) - eps_D eps^-2 = 0
    eps_star = (tv_mean / (c * gamma)) ** (1.0 / (gamma + 1.0))
    bound = c * eps_star ** gamma + tv_mean / eps_star
    K = bound / max(tv_mean ** (gamma / (gamma + 1.0)), 1e-30)
    return {"bound": float(bound), "eps_star": float(eps_star), "K": float(K)}


def fit_power_law(D, y):
    D = np.asarray(D, float)
    y = np.asarray(y, float)
    ok = y > 0
    coef = np.polyfit(np.log(D[ok]), np.log(y[ok]), 1)
    return {"A": float(np.exp(coef[1])), "beta": float(-coef[0])}


def main():
    summary = {}
    for label, prefix, adir in SETS:
        rows = load_domain(prefix, adir)
        if len(rows) < 3:
            print(f"[skip] {label}: {len(rows)} drafts")
            continue
        # Margin is target-only; pool to reduce cache noise.
        margin = np.concatenate([r["margin"] for r in rows.values()])
        margin = margin[margin > 0]
        tail = fit_margin_tail(margin)
        D = [rows[n]["D"] for n in rows]
        q = [rows[n]["q"] for n in rows]
        tv = [rows[n]["E_TV"] for n in rows]
        beta_fit = fit_power_law(D, tv)
        alpha_fit = fit_power_law(D, q)
        alpha_pred = beta_fit["beta"] * tail["gamma"] / (tail["gamma"] + 1.0)

        per_draft = []
        for name, r in rows.items():
            eb = empirical_bound(margin, r["E_TV"])
            pb = power_bound(tail["c"], tail["gamma"], r["E_TV"])
            per_draft.append({
                "draft": name, "D": r["D"], "q": r["q"], "E_TV": r["E_TV"],
                "empirical_bound": eb["bound"], "empirical_eps_star": eb["eps_star"],
                "empirical_M_at_star": eb["M_at_star"],
                "empirical_tv_over_eps": eb["tv_over_eps"],
                "nontrivial_eps_frac": eb["nontrivial_frac"],
                "parametric_bound": pb["bound"], "parametric_eps_star": pb["eps_star"],
                "bound_over_q": eb["bound"] / r["q"],
            })
        summary[label] = {
            "tail": tail,
            "beta_TV": beta_fit, "alpha_q": alpha_fit,
            "alpha_pred_sqrt": alpha_pred,
            "per_draft": per_draft,
        }
        print(f"\n== {label} ==")
        print(f"  margin tail: gamma={tail['gamma']:.3f} c={tail['c']:.3g} "
              f"R2={tail['r2']:.3f} window=[{tail['mass_lo']},{tail['mass_hi']}]")
        print(f"  beta_TV={beta_fit['beta']:.3f} alpha_q={alpha_fit['beta']:.3f} "
              f"alpha_pred(gamma/(gamma+1)*beta)={alpha_pred:.3f}")
        for r in per_draft:
            print(f"  D={r['D']:5.1f} q={r['q']:.4f} TV={r['E_TV']:.4f} "
                  f"emp_bound={r['empirical_bound']:.4f} (x{r['bound_over_q']:.2f} q) "
                  f"eps*={r['empirical_eps_star']:.3g} "
                  f"param={r['parametric_bound']:.4f} nontrivial={r['nontrivial_eps_frac']:.2f}")

    with open(os.path.join(OUT, "margin_bound.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # Markdown table for the paper/rebuttal.
    lines = ["# Numerical evaluation of the margin--TV bound", "",
             "Empirical bound is `min_eps M(eps) + E[TV]/eps` with `M` the empirical",
             "target margin CDF; parametric bound uses the fitted power-law tail.",
             "",
             "| set | gamma | c | beta(TV) | alpha(q) | alpha_pred |", "|---|---:|---:|---:|---:|---:|"]
    for label, s in summary.items():
        t = s["tail"]
        lines.append(f"| {label} | {t['gamma']:.3f} | {t['c']:.3g} | "
                     f"{s['beta_TV']['beta']:.3f} | {s['alpha_q']['beta']:.3f} | "
                     f"{s['alpha_pred_sqrt']:.3f} |")
    lines += ["", "| set | D | q | E[TV] | empirical bound | bound/q | eps* | nontrivial eps | parametric bound |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, s in summary.items():
        for r in s["per_draft"]:
            lines.append(
                f"| {label} | {r['D']:.1f}B | {r['q']:.4f} | {r['E_TV']:.4f} | "
                f"{r['empirical_bound']:.4f} | {r['bound_over_q']:.2f} | "
                f"{r['empirical_eps_star']:.3g} | {r['nontrivial_eps_frac']:.2f} | "
                f"{r['parametric_bound']:.4f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved", os.path.join(OUT, "margin_bound.json"), "and summary.md")


if __name__ == "__main__":
    main()
