"""Does the raw-OOD floor survive within entropy/margin strata?

If the aggregate q(D) intercept is just a composition effect (OOD text has more
high-entropy tokens), then q(D) within narrow entropy bins should go to zero.
We compute q(D) per entropy/margin quintile for WikiText and FineWeb raw, fit the
same saturating model in each stratum, and report the intercept.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

OUT = "out/entropy_strata"
os.makedirs(OUT, exist_ok=True)
D = np.array([0.8, 2.0, 4.0, 9.0, 27.0])
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_softplus(y):
    y = np.maximum(y, 1e-8)
    return np.log(np.expm1(y))


def fit_softplus(D, y):
    D = np.asarray(D, float)
    y = np.asarray(y, float)
    if np.all(y < 1e-8):
        return {"q_inf": 0.0, "r2": 1.0, "bic": -np.inf}
    res = least_squares(
        lambda p: softplus(p[0]) + softplus(p[1]) * D ** (-(softplus(p[2]) + 0.02)) - y,
        [inv_softplus(max(y.min() * 0.5, 1e-4)), inv_softplus(max(y.max(), 1e-3)),
         inv_softplus(0.3)],
        bounds=([-30, -30, -30], [10, 10, 10]), max_nfev=100000)
    pred = softplus(res.x[0]) + softplus(res.x[1]) * D ** (-(softplus(res.x[2]) + 0.02))
    rss = float(np.sum((y - pred) ** 2))
    n = len(y)
    return {"q_inf": float(softplus(res.x[0])),
            "A": float(softplus(res.x[1])),
            "alpha": float(softplus(res.x[2]) + 0.02),
            "r2": float(1 - rss / np.sum((y - y.mean()) ** 2)),
            "bic": float(n * np.log(max(rss / n, 1e-18)) + 3 * np.log(n))}


def flatten_cache(cache_path):
    z = np.load(cache_path, allow_pickle=True)
    ent = z["ent"]
    margin = z["margin"]
    gl = z["gen_len"]
    return (np.concatenate([ent[i, :gl[i]] for i in range(len(gl))]),
            np.concatenate([margin[i, :gl[i]] for i in range(len(gl))]))


def q_by_bin(domain, cache_path, cov, bins=5):
    x, _ = flatten_cache(cache_path)
    if cov == "margin":
        x, _ = flatten_cache(cache_path)[1], None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    out = {}
    for model in MODELS:
        p = f"out/accept/{domain}_{model}.npz"
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        match = z["match"]
        if len(match) != len(b):
            return None
        out[model] = np.asarray([1 - match[b == k].mean() if (b == k).any() else np.nan
                                 for k in range(len(edges) - 1)])
    return edges, out


def main():
    out = {}
    for domain, cache in [("wikitext", "out/target_cache/wikitext_k256.npz"),
                          ("fineweb", "out/target_cache/fineweb_k256.npz"),
                          ("math500", "out/target_cache/math500_k256.npz"),
                          ("gsm8k", "out/target_cache/gsm8k_k256.npz"),
                          ("humaneval", "out/target_cache/humaneval_k256.npz"),
                          ("mtbench", "out/target_cache/mtbench_k256.npz")]:
        out[domain] = {}
        for cov in ["ent", "margin"]:
            res = q_by_bin(domain, cache, cov)
            if res is None:
                continue
            edges, by_model = res
            strata = []
            for k in range(len(edges) - 1):
                y = np.asarray([by_model[m][k] for m in MODELS], float)
                strata.append({"bin": k, "q": y.tolist(), **fit_softplus(D, y)})
            out[domain][cov] = {"edges": edges.tolist(), "strata": strata}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# q(D) within entropy/margin strata", ""]
    for domain, covs in out.items():
        for cov, e in covs.items():
            lines.append(f"## {domain} / {cov}")
            lines.append("| bin | edge range | q(0.8B) | q(2B) | q(4B) | q(9B) | q(27B) | q_inf | R2 |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
            for s in e["strata"]:
                rng = f"{e['edges'][s['bin']]:.3f}--{e['edges'][s['bin']+1]:.3f}"
                q = s["q"]
                lines.append(f"| {s['bin']} | {rng} | " + " | ".join(f"{v:.4f}" for v in q) +
                             f" | {s['q_inf']:.4f} | {s['r2']:.3f} |")
            lines.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", OUT)


if __name__ == "__main__":
    main()
