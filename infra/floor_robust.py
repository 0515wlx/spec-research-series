"""Robust floor fits: softplus/half-normal intercepts and q_s = E[TV] curves.

This script uses existing acceptance caches only.  It refits the saturating model
    q(D) = q_inf + A D^{-alpha}
with three parameterisations of q_inf:
  * bounded   : q_inf in [0, 1] (the original fit)
  * softplus  : q_inf = softplus(theta), strictly positive and smooth
  * halfnormal: q_inf = |theta|, non-negative with a boundary at zero
and repeats the analysis for q_s = E[TV] instead of greedy disagreement q.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

OUT = "out/floor_robust"
os.makedirs(OUT, exist_ok=True)

D = np.array([0.8, 2.0, 4.0, 9.0, 27.0])

# label -> (domain, accept dir, filename prefix)
SETS = {
    "MATH-500": ("math500", "out/accept", "math500_"),
    "GSM8K": ("gsm8k", "out/accept", "gsm8k_"),
    "HumanEval": ("humaneval", "out/accept", "humaneval_"),
    "MT-Bench": ("mtbench", "out/accept", "mtbench_"),
    "WikiText raw": ("wikitext", "out/accept", "wikitext_"),
    "FineWeb raw": ("fineweb", "out/accept", "fineweb_"),
    "WikiText gen": ("wikitext_gen", "out/accept", "wikitext_gen_"),
    "FineWeb gen": ("fineweb_gen", "out/accept", "fineweb_gen_"),
    "MATH Qwen3.6": ("math500", "out/accept/repl", "repl_math500_"),
    "WikiText Qwen3.6": ("wikitext", "out/accept/repl", "repl_wikitext_"),
    "FineWeb Qwen3.6": ("fineweb", "out/accept/repl", "repl_fineweb_"),
}
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_softplus(y):
    y = np.maximum(y, 1e-8)
    return np.log(np.expm1(y))


def floor_bounded(D, p):
    qinf, A, alpha = p
    return qinf + A * np.power(D, -alpha)


def floor_softplus(D, p):
    qinf = softplus(p[0])
    A = softplus(p[1])
    alpha = softplus(p[2]) + 0.02
    return qinf + A * np.power(D, -alpha)


def floor_halfnormal(D, p):
    qinf = np.abs(p[0])
    A = softplus(p[1])
    alpha = softplus(p[2]) + 0.02
    return qinf + A * np.power(D, -alpha)


def power(D, p):
    A, alpha = p
    return A * np.power(D, -alpha)


def loglin(D, p):
    a, b = p
    return a + b * np.log10(D)


def invlog(D, p):
    a, b = p
    return a + b / np.log10(D + 1.0)


def fit_model(name, D, y):
    D = np.asarray(D, float)
    y = np.asarray(y, float)
    if name == "bounded":
        res = least_squares(lambda p: floor_bounded(D, p) - y,
                            [max(y.min() * 0.5, 1e-4), max(y.max(), 1e-3), 0.3],
                            bounds=([0, 0, 0], [1, 10, 10]), max_nfev=100000)
        qinf = float(res.x[0])
    elif name == "softplus":
        res = least_squares(lambda p: floor_softplus(D, p) - y,
                            [inv_softplus(max(y.min() * 0.5, 1e-4)),
                             inv_softplus(max(y.max(), 1e-3)), inv_softplus(0.3)],
                            bounds=([-30, -30, -30], [10, 10, 10]), max_nfev=100000)
        qinf = float(softplus(res.x[0]))
    elif name == "halfnormal":
        res = least_squares(lambda p: floor_halfnormal(D, p) - y,
                            [max(y.min() * 0.5, 1e-4),
                             inv_softplus(max(y.max(), 1e-3)), inv_softplus(0.3)],
                            bounds=([-10, -30, -30], [10, 10, 10]), max_nfev=100000)
        qinf = float(abs(res.x[0]))
    elif name == "power":
        res = least_squares(lambda p: power(D, p) - y,
                            [max(y.max(), 1e-3), 0.3],
                            bounds=([0, 0], [10, 10]), max_nfev=100000)
        qinf = None
    elif name == "loglinear":
        res = least_squares(lambda p: loglin(D, p) - y,
                            [y.mean(), 0.0], max_nfev=100000)
        qinf = None
    elif name == "invlog":
        res = least_squares(lambda p: invlog(D, p) - y,
                            [y.mean(), 0.0], max_nfev=100000)
        qinf = None
    else:
        raise ValueError(name)
    pred = {
        "bounded": floor_bounded, "softplus": floor_softplus,
        "halfnormal": floor_halfnormal, "power": power,
        "loglinear": loglin, "invlog": invlog,
    }[name](D, res.x)
    rss = float(np.sum((y - pred) ** 2))
    n, k = len(y), len(res.x)
    return {
        "params": [float(x) for x in res.x],
        "q_inf": qinf,
        "pred": pred.tolist(),
        "rss": rss,
        "r2": float(1 - rss / np.sum((y - y.mean()) ** 2)),
        "aic": float(n * np.log(max(rss / n, 1e-18)) + 2 * k),
        "bic": float(n * np.log(max(rss / n, 1e-18)) + k * np.log(n)),
    }


def profile_ci(name, D, y, grid=400, level=0.95):
    best = fit_model(name, D, y)
    rss_min = best["rss"]
    n = len(y)
    sigma2 = rss_min / max(n - 3, 1)
    thresh = rss_min + chi2.ppf(level, 1) * sigma2
    y = np.asarray(y, float)
    if name == "bounded":
        grid_vals = np.linspace(0.0, max(y.max() * 1.3, 0.2), grid)
        def resid(qinf, p):
            return qinf + p[0] * D ** (-p[1]) - y
        p0 = [max(y.max(), 1e-3), 0.3]
        bounds = ([0, 0], [10, 10])
        qs = grid_vals
    elif name == "softplus":
        theta = np.linspace(-12, 0, grid)
        qs = softplus(theta)
        def resid(qinf, p):
            A, alpha = p
            return qinf + A * D ** (-alpha) - y
        p0 = [max(y.max(), 1e-3), 0.3]
        bounds = ([0, 0], [10, 10])
        grid_vals = theta
    elif name == "halfnormal":
        qs = np.linspace(0.0, max(y.max() * 1.3, 0.2), grid)
        def resid(qinf, p):
            return qinf + p[0] * D ** (-p[1]) - y
        p0 = [max(y.max(), 1e-3), 0.3]
        bounds = ([0, 0], [10, 10])
        grid_vals = qs
    else:
        return [None, None]
    rss = []
    for g in grid_vals:
        qinf = softplus(g) if name == "softplus" else g
        try:
            r = least_squares(lambda p: resid(qinf, p), p0, bounds=bounds, max_nfev=20000)
            rss.append(float(np.sum(resid(qinf, r.x) ** 2)))
        except Exception:
            rss.append(np.inf)
    ok = qs[np.asarray(rss) <= thresh]
    return [float(ok.min()), float(ok.max())] if len(ok) else [None, None]


def loocv(name, D, y):
    errs = []
    for i in range(len(D)):
        mask = np.ones(len(D), bool)
        mask[i] = False
        try:
            f = fit_model(name, D[mask], y[mask])
            pred = {
                "bounded": floor_bounded, "softplus": floor_softplus,
                "halfnormal": floor_halfnormal,
            }[name](D[i:i + 1], np.asarray(f["params"]))[0]
            errs.append(float(pred - y[i]))
        except Exception:
            errs.append(np.nan)
    errs = np.asarray(errs, float)
    return {"rmse": float(np.sqrt(np.nanmean(errs ** 2))), "errors": errs.tolist()}


def bootstrap_qinf(name, D, per_seq, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ys = []
        for arr in per_seq:
            arr = np.asarray(arr, float)
            ys.append(arr[rng.integers(0, len(arr), len(arr))].mean())
        try:
            vals.append(fit_model(name, D, np.asarray(ys))["q_inf"])
        except Exception:
            pass
    if not vals:
        return {"mean": None, "ci95": [None, None], "n": 0}
    vals = np.asarray(vals, float)
    return {"mean": float(vals.mean()),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "n": int(len(vals))}


def load_series(domain, accept_dir, prefix):
    qs, tvs = [], []
    for model in MODELS:
        jp = os.path.join(accept_dir, f"{prefix}{model}.json")
        npz = os.path.join(accept_dir, f"{prefix}{model}.npz")
        if not (os.path.exists(jp) and os.path.exists(npz)):
            return None, None
        d = json.load(open(jp))
        qs.append(np.asarray(d["q_per_seq"], float))
        z = np.load(npz, allow_pickle=True)
        tv, sid = z["tv"], z["seq_id"]
        tvs.append(np.asarray([tv[sid == i].mean() for i in range(int(sid.max()) + 1)], float))
    return qs, tvs


def main():
    out = {"sets": {}}
    for label, (domain, accept_dir, prefix) in SETS.items():
        qseq, tseq = load_series(domain, accept_dir, prefix)
        if qseq is None:
            print("skip missing", label)
            continue
        entry = {"domain": domain, "metrics": {}}
        for metric, per_seq in [("q", qseq), ("q_s", tseq)]:
            y = np.asarray([a.mean() for a in per_seq], float)
            m = {"values": y.tolist(), "fits": {}}
            for name in ["bounded", "softplus", "halfnormal", "power", "loglinear", "invlog"]:
                f = fit_model(name, D, y)
                if name in ["bounded", "softplus", "halfnormal"]:
                    f["profile_ci95"] = profile_ci(name, D, y)
                    f["loocv"] = loocv(name, D, y)
                    f["bootstrap"] = bootstrap_qinf(name, D, per_seq)
                m["fits"][name] = f
            best_alt = min((m["fits"][n]["bic"], n) for n in ["power", "loglinear", "invlog"])
            m["best_alt"] = best_alt[1]
            m["delta_bic_softplus"] = float(best_alt[0] - m["fits"]["softplus"]["bic"])
            m["delta_bic_halfnormal"] = float(best_alt[0] - m["fits"]["halfnormal"]["bic"])
            entry["metrics"][metric] = m
            print(f"{label:18s} {metric:3s} "
                  f"softplus={m['fits']['softplus']['q_inf']:.4f} "
                  f"halfnorm={m['fits']['halfnormal']['q_inf']:.4f} "
                  f"bounded={m['fits']['bounded']['q_inf']:.4f}")
        out["sets"][label] = entry
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Robust floor fits (softplus / half-normal) and q_s = E[TV]", ""]
    lines += ["| set | metric | q_inf softplus | boot 95% | profile 95% | q_inf half-normal | q_inf bounded | Delta BIC softplus |",
              "|---|---|---:|---|---|---:|---:|---:|"]
    for label, e in out["sets"].items():
        for metric, m in e["metrics"].items():
            f = m["fits"]["softplus"]
            b = f["bootstrap"]
            p = f["profile_ci95"]
            lines.append(
                f"| {label} | {metric} | {f['q_inf']:.4f} | "
                f"[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | "
                f"[{p[0]:.4f}, {p[1]:.4f}] | "
                f"{m['fits']['halfnormal']['q_inf']:.4f} | "
                f"{m['fits']['bounded']['q_inf']:.4f} | "
                f"{m['delta_bic_softplus']:+.2f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", OUT)


if __name__ == "__main__":
    main()
