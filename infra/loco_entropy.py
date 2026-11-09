"""Leave-one-corpus-out screen with entropy as a second feature.

The mechanism analysis shows that low-margin mass is largely a monotone readout
of target predictive entropy.  This script asks whether adding corpus mean
entropy to the mass-only screen improves out-of-sample prediction:

    q = a_w + b_w * L                  (mass only)
    q = a_w + b_w * L + c_w * H        (mass + mean target entropy)

It uses the same eleven raw corpora and the same held-out deployment suites as
``infra/loco_margin.py``.  Existing result NPZ files only.
"""
from __future__ import annotations

import json
import os

import numpy as np

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


def features(path):
    z = np.load(path)
    m = z["margin_log"].astype(np.float64)
    ent = z["ent_t"].astype(np.float64)
    match = z["match"].astype(bool)
    return {"q": 1.0 - float(match.mean()),
            "low_mass": float((m < 0.5).mean()),
            "mean_entropy": float(ent.mean())}


def cells_at(w):
    cells = {}
    for label, d, s in RAW + EXTERNAL:
        p = os.path.join(d, f"{s}_frac{w:g}.npz")
        if os.path.exists(p):
            cells[label] = features(p)
    return cells


def fit(X, y, xt):
    A = np.hstack([np.ones((len(X), 1)), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.r_[1.0, xt] @ coef)


def loo(cells, w, with_entropy):
    labels = [l for l, _, _ in RAW if l in cells]
    X = []
    y = []
    for l in labels:
        c = cells[l]
        X.append([c["low_mass"], c["mean_entropy"]] if with_entropy
                 else [c["low_mass"]])
        y.append(c["q"])
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    qs = q_star(1.0 - MLP_SHARE * (1.0 - w))
    preds, actual = [], []
    for i, l in enumerate(labels):
        mask = np.ones(len(labels), bool)
        mask[i] = False
        xt = X[i]
        preds.append(fit(X[mask], y[mask], xt))
        actual.append(y[i])
    preds = np.asarray(preds)
    actual = np.asarray(actual)
    ext = []
    for label, _, _ in EXTERNAL:
        if label not in cells:
            continue
        c = cells[label]
        xt = ([c["low_mass"], c["mean_entropy"]] if with_entropy
              else [c["low_mass"]])
        pred = fit(X, y, np.asarray(xt))
        ext.append({"suite": label, "q": c["q"], "q_pred": pred,
                    "error": pred - c["q"], "q_star": qs,
                    "pred_above": bool(pred > qs),
                    "actual_above": bool(c["q"] > qs)})
    return {
        "width": w, "q_star": qs,
        "mae": float(np.mean(np.abs(preds - actual))),
        "max_abs": float(np.max(np.abs(preds - actual))),
        "fn": int(sum(1 for p, a in zip(preds, actual)
                      if (p <= qs) and (a > qs))),
        "external": ext,
    }


def main():
    out = {}
    for w in WIDTHS:
        cells = cells_at(w)
        if len(cells) < 4:
            continue
        out[str(w)] = {
            "mass_only": loo(cells, w, False),
            "mass_entropy": loo(cells, w, True),
        }
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/loco_entropy.json", "w") as f:
        json.dump(out, f, indent=1)
    for w, r in out.items():
        m, me = r["mass_only"], r["mass_entropy"]
        print(f"w={w}: mass MAE={m['mae']:.4f} max={m['max_abs']:.4f} FN={m['fn']} | "
              f"+entropy MAE={me['mae']:.4f} max={me['max_abs']:.4f} FN={me['fn']}")
        for a, b in zip(m["external"], me["external"]):
            print(f"   {a['suite']:<10} q={a['q']:.4f} mass={a['q_pred']:.4f} "
                  f"+H={b['q_pred']:.4f}")


if __name__ == "__main__":
    main()
