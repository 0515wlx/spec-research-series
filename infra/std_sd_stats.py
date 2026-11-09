"""Standard speculative-sampling statistics from existing caches.

The teacher-forced caches give, for every target-generated token, the exact
target/draft TV distance.  Under standard speculative sampling the expected
acceptance probability of a draft token at that prefix is ``1 - TV``.  This
script uses that identity to compute, without any new model runs:

  * the sampling rejection rate ``q_s = E[TV]`` and its local capacity
    exponent / D_1/2;
  * a simulated accepted length ``tau_sim(gamma)`` under the standard
    acceptance rule (independent per-token acceptance approximation);
  * the relation between the greedy rejection rate ``q_greedy`` and the
    standard-SD rejection rate ``q_s``.

The real standard-SD loop in :mod:`infra.spec_sampling` is the validation;
this script provides the full capacity curve.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT = "out/std_sd"
os.makedirs(OUT, exist_ok=True)
DRAFTS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
D = np.array([0.8, 2.0, 4.0, 9.0, 27.0])
DOMAINS = ["math500", "gsm8k", "humaneval", "mtbench", "wikitext", "fineweb",
           "wikitext_gen", "fineweb_gen", "c4_gen", "gutenberg_gen", "code_gen"]


def slope_ci(qs_per_seq, n_boot=1000, seed=0):
    """Local exponent alpha = -slope(log q, log D) with sequence bootstrap."""
    qs = np.array([np.mean(x) for x in qs_per_seq], float)
    x = np.log(D)
    y = np.log(np.clip(qs, 1e-12, None))
    alpha = -float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        qb = np.array([np.mean(x[rng.integers(0, len(x), len(x))])
                       for x in qs_per_seq], float)
        try:
            vals.append(-float(np.polyfit(x, np.log(np.clip(qb, 1e-12, None)), 1)[0]))
        except Exception:
            pass
    vals = np.array(vals)
    return {
        "alpha": alpha,
        "alpha_ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "D_half": float(2 ** (1 / alpha)) if alpha > 0 else None,
        "D_half_ci95": [float(2 ** (1 / np.quantile(vals, 0.975))),
                        float(2 ** (1 / np.quantile(vals, 0.025)))] if len(vals) else None,
    }


def split_by_seq(values, seq_id):
    return [values[seq_id == s] for s in np.unique(seq_id)]


def simulate_tau(tv, gamma, n_rep=30, seed=0):
    """Monte Carlo accepted length under standard acceptance (1-TV)."""
    a = np.clip(1.0 - np.asarray(tv, float), 0.0, 1.0)
    N = len(a)
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_rep):
        t = 0
        steps = emitted = 0
        while t < N:
            acc = 0
            for i in range(gamma):
                if t + i >= N:
                    break
                if rng.random() < a[t + i]:
                    acc += 1
                else:
                    break
            emitted += acc + 1
            steps += 1
            t += acc + 1
        taus.append(emitted / max(steps, 1))
    return float(np.mean(taus))


def tau_ci(tv_per_seq, gamma, n_boot=500, seed=0):
    vals = [simulate_tau(x, gamma) for x in tv_per_seq]
    rng = np.random.default_rng(seed)
    b = [np.mean(np.array(vals)[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    return {"tau": float(np.mean(vals)),
            "tau_ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]}


def main():
    stats = {"cells": {}, "domains": {}}
    for dom in DOMAINS:
        qseq, tvseq, tv_by_seq = [], [], []
        rows = []
        ok = True
        for draft in DRAFTS:
            jp = f"out/accept/{dom}_{draft}.json"
            npz = f"out/accept/{dom}_{draft}.npz"
            if not (os.path.exists(jp) and os.path.exists(npz)):
                ok = False
                break
            j = json.load(open(jp))
            z = np.load(npz, allow_pickle=True)
            tv = z["tv"].astype(float)
            sid = z["seq_id"]
            parts = split_by_seq(tv, sid)
            qseq.append(np.array(j["q_per_seq"], float))
            tvseq.append(np.array([float(x.mean()) for x in parts], float))
            tv_by_seq.append(parts)
            rows.append({
                "draft": draft, "D": float(DRAFTS.index(draft)),
                "q_greedy": float(j["q"]), "q_s": float(j["E_TV"]),
                "ratio_greedy_over_sampling": float(j["q"] / max(j["E_TV"], 1e-12)),
                "tau_inf_greedy": float(j["tau_inf"]),
            })
        if not ok:
            continue
        for r, d in zip(rows, D):
            r["D"] = float(d)
        g = slope_ci(qseq)
        s = slope_ci(tvseq)
        for r, parts in zip(rows, tv_by_seq):
            r["tau_sim_g4"] = tau_ci(parts, 4)
            r["tau_sim_g8"] = tau_ci(parts, 8)
        # Fit-window sensitivity: does the local exponent depend on the range?
        windows = {"0.8-27": [0, 1, 2, 3, 4], "0.8-9": [0, 1, 2, 3], "2-27": [1, 2, 3, 4]}
        sens = {}
        for name, idx in windows.items():
            xx = np.log(D[idx])
            qg = np.array([np.mean(qseq[i]) for i in idx], float)
            qs = np.array([np.mean(tvseq[i]) for i in idx], float)
            sens[name] = {
                "alpha_greedy": float(-np.polyfit(xx, np.log(np.clip(qg, 1e-12, None)), 1)[0]),
                "alpha_sampling": float(-np.polyfit(xx, np.log(np.clip(qs, 1e-12, None)), 1)[0]),
            }
        stats["domains"][dom] = {
            "q_greedy": g, "q_s": s,
            "fit_window": sens,
            "rows": rows,
        }
        for r in rows:
            stats["cells"][f"{dom}/{r['draft']}"] = r

    # Relation between greedy and sampling rejection.
    cells = list(stats["cells"].values())
    qg = np.array([c["q_greedy"] for c in cells])
    qs = np.array([c["q_s"] for c in cells])
    tau = np.array([c["tau_sim_g4"]["tau"] for c in cells], float)
    stats["relation"] = {
        "pearson_all": float(pearsonr(qg, qs)[0]),
        "spearman_all": float(spearmanr(qg, qs)[0]),
        "mean_ratio_greedy_over_sampling": float(np.mean(qg / np.clip(qs, 1e-12, None))),
        "max_abs_gap": float(np.max(np.abs(qg - qs))),
        "pearson_qgreedy_vs_tau": float(pearsonr(qg, tau)[0]),
        "spearman_qgreedy_vs_tau": float(spearmanr(qg, tau)[0]),
        "pearson_qsampling_vs_tau": float(pearsonr(qs, tau)[0]),
        "n_cells": len(cells),
    }

    # Within-domain standardized correlation (removes domain-level difficulty).
    doms = sorted({k.split("/")[0] for k in stats["cells"]})
    zg, zs, zt = [], [], []
    for dom in doms:
        cs = [stats["cells"][f"{dom}/{d}"] for d in DRAFTS if f"{dom}/{d}" in stats["cells"]]
        a = np.array([c["q_greedy"] for c in cs])
        b = np.array([c["q_s"] for c in cs])
        t = np.array([c["tau_sim_g4"]["tau"] for c in cs])
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            continue
        zg.extend(((a - a.mean()) / a.std()).tolist())
        zs.extend(((b - b.mean()) / b.std()).tolist())
        zt.extend(((t - t.mean()) / t.std()).tolist())
    if len(zg) >= 3:
        stats["relation"]["pearson_within_domain_standardized"] = float(pearsonr(zg, zs)[0])
        stats["relation"]["spearman_within_domain_standardized"] = float(spearmanr(zg, zs)[0])
        stats["relation"]["pearson_qgreedy_tau_within"] = float(pearsonr(zg, zt)[0])

    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    lines = ["# Standard speculative sampling from caches", ""]
    lines += ["| domain | q_greedy alpha | q_s=E[TV] alpha | D1/2(q_s) | tau_sim(g=4) 0.8B | tau_sim(g=4) 27B |",
              "|---|---:|---:|---:|---:|---:|"]
    for dom, e in stats["domains"].items():
        r0, r1 = e["rows"][0], e["rows"][-1]
        lines.append(f"| {dom} | {e['q_greedy']['alpha']:.3f} | {e['q_s']['alpha']:.3f} | "
                     f"{e['q_s']['D_half']:.1f}x | {r0['tau_sim_g4']['tau']:.2f} | "
                     f"{r1['tau_sim_g4']['tau']:.2f} |")
    lines += ["", "## Greedy vs standard SD", ""]
    for k, v in stats["relation"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines += ["", "## Cells", "",
              "| cell | D | q_greedy | q_s | tau_sim(g=4) |",
              "|---|---:|---:|---:|---:|"]
    for k, c in stats["cells"].items():
        lines.append(f"| {k} | {c['D']:.1f} | {c['q_greedy']:.4f} | {c['q_s']:.4f} | "
                     f"{c['tau_sim_g4']['tau']:.2f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print("saved", os.path.join(OUT, "stats.json"))


if __name__ == "__main__":
    main()
