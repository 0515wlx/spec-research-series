"""Leave-one-corpus-out validation of the margin-mass diagnostic.

The review asks whether the empirical relation between low-margin mass and
rejection (Figure ``fig_margin_mass``) extrapolates to an unseen corpus, and
whether it can be used as a pre-deployment diagnostic.

For every retained width we fit

    q = a_w + b_w * L,          L = fraction of target tokens with margin < 0.5

on the eleven English/multilingual/code raw corpora, then predict each held-out
corpus from the remaining ten.  We report MAE/RMSE/max error, rank correlation,
and whether the predicted rejection is still above the FLOP break-even.  We also
apply the same fit to two deployment-like held-out suites (MT-Bench chat and
SWE-bench issue-to-patch) and simulate the operational recipe: draw a small
prefix sample, estimate L, predict q, compare with q*.

Only existing result NPZ files are read; no model is run.
"""
from __future__ import annotations

import glob
import json
import os
from itertools import combinations

import numpy as np

ROOT = "."
RAW = [
    ("WikiText", "out/rank_scan/qwen3.5-9b_actwidth", "wikitext"),
    ("FineWeb", "out/rank_scan/qwen3.5-9b_actwidth", "fineweb"),
    ("C4", "out/rank_scan/qwen3.5-9b_actwidth_c4", "c4"),
    ("Wiki-zh", "out/rank_scan/qwen3.5-9b_actwidth_multilingual", "wikipedia_zh"),
    ("Wiki-de", "out/rank_scan/qwen3.5-9b_actwidth_multilingual", "wikipedia_de"),
    ("Wiki-fr", "out/rank_scan/qwen3.5-9b_actwidth_multilingual", "wikipedia_fr"),
    ("Wiki-ja", "out/rank_scan/qwen3.5-9b_actwidth_multilingual", "wikipedia_ja"),
    ("FineWeb2-zh", "out/rank_scan/qwen3.5-9b_actwidth_raw2", "fineweb2_zh"),
    ("Code", "out/rank_scan/qwen3.5-9b_actwidth_raw2", "code_raw"),
    ("PG-19", "out/rank_scan/qwen3.5-9b_actwidth_raw2", "pg19"),
    ("arXiv", "out/rank_scan/qwen3.5-9b_actwidth_raw2", "arxiv"),
]
EXTERNAL = [
    ("MT-Bench", "out/rank_scan/qwen3.5-9b_actwidth_mtbench", "mtbench9b"),
    ("SWE-bench", "out/rank_scan/qwen3.5-9b_actwidth_swe", "swe"),
    ("MATH", "out/rank_scan/qwen3.5-9b_actwidth", "math9b"),
]
WIDTHS = [0.5, 0.9, 0.95, 0.98, 0.99]
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


def features(path: str) -> dict:
    z = np.load(path)
    m = z["margin_log"].astype(np.float64)
    d = z["d_margin_log"].astype(np.float64)
    match = z["match"].astype(bool)
    return {
        "q": float(1.0 - match.mean()),
        "q_lb": float(((m + d) < 0).mean()),
        "low_mass": float((m < 0.5).mean()),
        "low_mass_02": float((m < 0.2).mean()),
        "median_m": float(np.median(m)),
        "mean_abs_d": float(np.abs(d).mean()),
        "tokens": int(len(m)),
    }


def load_cells():
    cells = {}
    for label, d, s in RAW + EXTERNAL:
        for w in WIDTHS:
            p = os.path.join(d, f"{s}_frac{w:g}.npz")
            if os.path.exists(p):
                cells[(label, w)] = features(p)
    return cells


def fit_predict(x, y, xt):
    A = np.vstack([np.ones_like(x), x]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(coef[0] + coef[1] * xt), coef


def fit_predict2(X, y, xt):
    A = np.hstack([np.ones((len(X), 1)), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.r_[1.0, xt] @ coef), coef


def loo_one_width(cells, w):
    rows = []
    for label, _, _ in RAW:
        c = cells.get((label, w))
        if c is None:
            continue
        rows.append((label, c["low_mass"], c["mean_abs_d"], c["q"]))
    labels = [r[0] for r in rows]
    L = np.asarray([r[1] for r in rows], float)
    D = np.asarray([r[2] for r in rows], float)
    Q = np.asarray([r[3] for r in rows], float)
    rho = 1.0 - MLP_SHARE * (1.0 - w)
    qs = q_star(rho)
    preds = np.zeros_like(Q)
    preds2 = np.zeros_like(Q)
    fold = []
    for i in range(len(Q)):
        mask = np.ones(len(Q), bool)
        mask[i] = False
        pred, _ = fit_predict(L[mask], Q[mask], L[i])
        pred2, _ = fit_predict2(np.vstack([L[mask], D[mask]]).T, Q[mask],
                                np.asarray([L[i], D[i]]))
        preds[i] = pred
        preds2[i] = pred2
        fold.append({
            "corpus": labels[i], "low_mass": float(L[i]),
            "mean_abs_d": float(D[i]), "q": float(Q[i]),
            "q_pred": float(pred), "error": float(pred - Q[i]),
            "q_pred_2feat": float(pred2), "error_2feat": float(pred2 - Q[i]),
            "q_star": qs,
            "actual_above": bool(Q[i] > qs),
            "pred_above": bool(pred > qs),
            "pred_above_2feat": bool(pred2 > qs),
        })
    err = preds - Q
    err2 = preds2 - Q
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(preds)),
                                  np.argsort(np.argsort(Q)))[0, 1])
    return {
        "width": w, "rho_flop": rho, "q_star": qs,
        "n_corpora": int(len(Q)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "pearson": float(np.corrcoef(preds, Q)[0, 1]),
        "spearman": rank_corr,
        "mae_2feat": float(np.mean(np.abs(err2))),
        "rmse_2feat": float(np.sqrt(np.mean(err2 ** 2))),
        "max_abs_2feat": float(np.max(np.abs(err2))),
        "pred_above_count": int(sum(r["pred_above"] for r in fold)),
        "actual_above_count": int(sum(r["actual_above"] for r in fold)),
        "classification_accuracy": float(np.mean(
            [r["pred_above"] == r["actual_above"] for r in fold])),
        "folds": fold,
    }


def external_check(cells, w):
    train = [(c["low_mass"], c["mean_abs_d"], c["q"])
             for (lab, ww), c in cells.items()
             if lab in [x[0] for x in RAW] and ww == w]
    if len(train) < 3:
        return []
    L = np.asarray([t[0] for t in train], float)
    D = np.asarray([t[1] for t in train], float)
    Q = np.asarray([t[2] for t in train], float)
    rho = 1.0 - MLP_SHARE * (1.0 - w)
    qs = q_star(rho)
    out = []
    for label, _, _ in EXTERNAL:
        c = cells.get((label, w))
        if c is None:
            continue
        pred, coef = fit_predict(L, Q, c["low_mass"])
        pred2, _ = fit_predict2(np.vstack([L, D]).T, Q,
                                np.asarray([c["low_mass"], c["mean_abs_d"]]))
        out.append({
            "suite": label, "width": w, "low_mass": c["low_mass"],
            "mean_abs_d": c["mean_abs_d"],
            "q": c["q"], "q_lb": c["q_lb"], "q_pred": float(pred),
            "error": float(pred - c["q"]),
            "q_pred_2feat": float(pred2),
            "error_2feat": float(pred2 - c["q"]),
            "q_star": qs,
            "actual_above": bool(c["q"] > qs),
            "pred_above": bool(pred > qs),
            "pred_above_2feat": bool(pred2 > qs),
        })
    return out


def deploy_simulation(cells, w, n=1024, repeats=200, seed=0):
    """Draw a small token sample, estimate L, predict q, compare with q*."""
    rng = np.random.default_rng(seed)
    rho = 1.0 - MLP_SHARE * (1.0 - w)
    qs = q_star(rho)
    train = [(c["low_mass"], c["mean_abs_d"], c["q"])
             for (lab, ww), c in cells.items()
             if lab in [x[0] for x in RAW] and ww == w]
    if len(train) < 3:
        return []
    L = np.asarray([t[0] for t in train], float)
    D = np.asarray([t[1] for t in train], float)
    Q = np.asarray([t[2] for t in train], float)
    out = []
    for label, d, s in EXTERNAL:
        p = os.path.join(d, f"{s}_frac{w:g}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        m = z["margin_log"].astype(np.float64)
        dm = z["d_margin_log"].astype(np.float64)
        q = 1.0 - z["match"].astype(bool).mean()
        draws = []
        draws2 = []
        for _ in range(repeats):
            idx = rng.integers(0, len(m), n)
            l_hat = float((m[idx] < 0.5).mean())
            pred, _ = fit_predict(L, Q, l_hat)
            draws.append(pred)
            d_hat = float(np.abs(dm[idx]).mean())
            pred2, _ = fit_predict2(np.vstack([L, D]).T, Q,
                                    np.asarray([l_hat, d_hat]))
            draws2.append(pred2)
        draws = np.asarray(draws)
        draws2 = np.asarray(draws2)
        out.append({
            "suite": label, "width": w, "n_sample": n,
            "low_mass_true": float((m < 0.5).mean()),
            "q_true": float(q), "q_star": qs,
            "q_pred_mean": float(draws.mean()),
            "q_pred_ci95": [float(np.percentile(draws, 2.5)),
                            float(np.percentile(draws, 97.5))],
            "pred_above_fraction": float((draws > qs).mean()),
            "q_pred_2feat_mean": float(draws2.mean()),
            "q_pred_2feat_ci95": [float(np.percentile(draws2, 2.5)),
                                  float(np.percentile(draws2, 97.5))],
            "pred_above_2feat_fraction": float((draws2 > qs).mean()),
        })
    return out


def main():
    cells = load_cells()
    out = {"widths": {}, "external": {}, "deploy": {}}
    for w in WIDTHS:
        if not any((lab, w) in cells for lab, _, _ in RAW):
            continue
        out["widths"][str(w)] = loo_one_width(cells, w)
        out["external"][str(w)] = external_check(cells, w)
        out["deploy"][str(w)] = deploy_simulation(cells, w)
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/loco_margin.json", "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Leave-one-corpus-out validation of the margin-mass diagnostic", ""]
    lines += ["Fit: q = a_w + b_w * L, L = fraction of target tokens with margin < 0.5.",
              "Training corpora: the eleven raw-text corpora; each row predicts one held-out corpus.",
              ""]
    lines += ["| width | q* | MAE | RMSE | max | MAE 2-feat | Pearson | Spearman | pred>q* | actual>q* |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w, r in out["widths"].items():
        lines.append(
            f"| {float(w):g} | {r['q_star']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} | "
            f"{r['max_abs']:.4f} | {r['mae_2feat']:.4f} | {r['pearson']:.3f} | {r['spearman']:.3f} | "
            f"{r['pred_above_count']}/{r['n_corpora']} | "
            f"{r['actual_above_count']}/{r['n_corpora']} |")
    lines += ["", "## Held-out folds", ""]
    for w, r in out["widths"].items():
        lines.append(f"### width {float(w):g}")
        lines += ["| corpus | L | q | q_pred | error | pred>q* |",
                  "|---|---:|---:|---:|---:|---|"]
        for f in r["folds"]:
            lines.append(
                f"| {f['corpus']} | {f['low_mass']:.4f} | {f['q']:.4f} | "
                f"{f['q_pred']:.4f} | {f['error']:+.4f} | {f['pred_above']} |")
        lines.append("")
    lines += ["## Deployment-like held-out suites (fit on eleven raw corpora)", "",
              "| width | suite | L | mean_abs_d | q | q_pred | error | q_pred 2-feat | error | q* | pred>q* |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w, rows in out["external"].items():
        for r in rows:
            lines.append(
                f"| {float(w):g} | {r['suite']} | {r['low_mass']:.4f} | "
                f"{r['mean_abs_d']:.3f} | {r['q']:.4f} | "
                f"{r['q_pred']:.4f} | {r['error']:+.4f} | "
                f"{r['q_pred_2feat']:.4f} | {r['error_2feat']:+.4f} | "
                f"{r['q_star']:.4f} | {r['pred_above']} |")
    lines += ["", "## Small-prefix deployment diagnostic (n=1024 tokens, 200 draws)", "",
              "| width | suite | q_true | q_pred mean | 95% pred | frac > q* | q_pred 2-feat | frac > q* |",
              "|---:|---|---:|---:|---|---:|---:|---:|"]
    for w, rows in out["deploy"].items():
        for r in rows:
            lines.append(
                f"| {float(w):g} | {r['suite']} | {r['q_true']:.4f} | "
                f"{r['q_pred_mean']:.4f} | [{r['q_pred_ci95'][0]:.4f}, "
                f"{r['q_pred_ci95'][1]:.4f}] | {r['pred_above_fraction']:.3f} | "
                f"{r['q_pred_2feat_mean']:.4f} | {r['pred_above_2feat_fraction']:.3f} |")
    with open("out/paper_stats/loco_margin.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved out/paper_stats/loco_margin.json")


if __name__ == "__main__":
    main()
