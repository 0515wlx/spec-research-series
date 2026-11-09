"""Required acceptance versus measured acceptance.

For each retained width we report

* alpha_acc = 1 - q_g (greedy) and 1 - E[TV] (standard sampling);
* alpha*_FLOP = 1 - q*_FLOP, the break-even acceptance under the optimistic
  FLOP roofline with tau = 1;
* alpha*_measured, the break-even acceptance at the measured batch-1 cost,
  reported as infeasible when the cost exceeds gamma + 1.

The gap alpha* - alpha_acc is the quantitative version of ``small perturbation,
not small rejection'': at 99% width the draft accepts 93% of tokens, but the
roofline requires 99.8%.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from infra.cost_frontier import accepted, q_star

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
MLP_SHARE = 0.6088
GAMMA = 4


def load_q(scan, sets):
    out = {}
    for w in WIDTHS:
        qs, tvs = [], []
        for s in sets:
            p = os.path.join(ROOT, scan, f"{s}_frac{w:g}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            qs.append(1.0 - z["match"].astype(bool).mean())
            tvs.append(z["tv"].astype(np.float64).mean())
        if qs:
            out[w] = {"q": float(np.mean(qs)), "tv": float(np.mean(tvs)),
                      "q_wiki": float(qs[0])}
    return out


def load_measured():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "out/latency/verify_9b_w*.json"))):
        d = json.load(open(p))
        if "target_one_s" not in d:
            continue
        w = float(d["width"])
        tau = d["target_verify_s"] / d["target_one_s"]
        rho = d["rho_token"]
        cost = GAMMA * rho + tau
        # solve A(alpha)=cost; >1 means no acceptance can break even.
        if cost >= GAMMA + 1:
            astar = None
        else:
            astar = 1.0 - q_star(rho, GAMMA)  # q_star uses tau=1; recompute below
            # direct solve for the measured cost
            lo, hi = 0.0, 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if accepted(mid, GAMMA) >= cost:
                    hi = mid
                else:
                    lo = mid
            astar = 0.5 * (lo + hi)
        out[w] = {"rho_token": float(rho), "tau": float(tau),
                  "cost": float(cost), "alpha_star": astar}
    return out


def main():
    q = load_q("out/rank_scan/qwen3.5-9b_actwidth", ["wikitext", "fineweb"])
    meas = load_measured()
    rows = []
    for w in WIDTHS:
        if w not in q:
            continue
        rho = 1.0 - MLP_SHARE * (1.0 - w)
        qs = q_star(rho, GAMMA)
        astar = 1.0 - qs
        acc_g = 1.0 - q[w]["q"]
        acc_s = 1.0 - q[w]["tv"]
        m = meas.get(w, {})
        rows.append({
            "width": w, "rho_flop": rho, "q_star": qs,
            "alpha_star_flop": astar,
            "alpha_acc_greedy": acc_g, "alpha_acc_sampling": acc_s,
            "gap_greedy": astar - acc_g, "gap_sampling": astar - acc_s,
            "rejection_ratio": (q[w]["q"] / qs) if qs > 0 else None,
            "measured_cost": m.get("cost"),
            "measured_alpha_star": m.get("alpha_star"),
        })

    os.makedirs(os.path.join(ROOT, "out/paper_stats"), exist_ok=True)
    with open(os.path.join(ROOT, "out/paper_stats/alpha_gap.json"), "w") as f:
        json.dump(rows, f, indent=1)

    ws = np.asarray([r["width"] for r in rows])
    acc_g = np.asarray([r["alpha_acc_greedy"] for r in rows])
    acc_s = np.asarray([r["alpha_acc_sampling"] for r in rows])
    astar = np.asarray([r["alpha_star_flop"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    ax = axes[0]
    ax.plot(100 * ws, astar, "o-", color="#c44e52", lw=1.6, ms=3.5,
            label=r"required $\alpha^\star_{\rm FLOP}$")
    ax.plot(100 * ws, acc_g, "s-", color="#4c72b0", lw=1.5, ms=3.2,
            label=r"measured greedy $\alpha_{\rm acc}$")
    ax.plot(100 * ws, acc_s, "^-", color="#55a868", lw=1.2, ms=3.0,
            label=r"measured sampling $1-\mathbb{E}[{\rm TV}]$")
    ax.set_xlabel("retained FFN width (%)")
    ax.set_ylabel("acceptance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) required vs measured acceptance", fontsize=9)

    ax = axes[1]
    gap = astar - acc_g
    ax.semilogy(100 * ws, gap, "o-", color="#c44e52", lw=1.6, ms=3.5)
    for w, g in zip(ws, gap):
        if w in (0.5, 0.98, 0.99):
            ax.annotate(f"{g:.3f}", (100 * w, g), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("retained FFN width (%)")
    ax.set_ylabel(r"$\alpha^\star_{\rm FLOP}-\alpha_{\rm acc}$")
    ax.grid(alpha=0.25, which="both")
    ax.set_title("(b) acceptance gap", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper/figures/fig_alpha_gap.pdf"),
                bbox_inches="tight")

    lines = [
        r"\begin{table}[H]",
        r"\centering", r"\footnotesize",
        r"\caption{Required versus measured acceptance on Qwen3.5-9B English raw text (mean of WikiText and FineWeb). $\alpha^\star_{\rm FLOP}$ is the break-even acceptance under the optimistic FLOP roofline ($\tau=1$, $\gamma=4$); $\alpha_{\rm acc}=1-q_g$ is measured. The last two columns use the measured batch-1 cost; ``--'' means the measured cost exceeds $\gamma+1$, so no acceptance can break even. Only the $30\%$-width row has a finite $\alpha^\star_{\rm meas}$ because its measured cost is $4.90<5$; every other measured row is cost-only infeasible.}",
        r"\label{tab:alpha_gap}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Width & $\rho_{\rm FLOP}$ & $q^\star_{\rm FLOP}$ & $\alpha^\star_{\rm FLOP}$ & $\alpha_{\rm acc}$ & gap & $C_{\rm meas}$ & $\alpha^\star_{\rm meas}$ \\",
        r"\midrule",
    ]
    for r in rows:
        mcost = "--" if r["measured_cost"] is None else f"{r['measured_cost']:.2f}"
        mastar = ("--" if r["measured_alpha_star"] is None
                  else f"{r['measured_alpha_star']:.3f}")
        lines.append(
            f"{r['width']:.2f} & {r['rho_flop']:.3f} & {r['q_star']:.4f} & "
            f"{r['alpha_star_flop']:.4f} & {r['alpha_acc_greedy']:.4f} & "
            f"{r['gap_greedy']:.4f} & {mcost} & {mastar} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(ROOT, "paper/tab_alpha_gap.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote fig_alpha_gap.pdf and tab_alpha_gap.tex ({len(rows)} widths)")
    for r in rows:
        print(f"{r['width']:.2f} alpha*={r['alpha_star_flop']:.4f} "
              f"acc={r['alpha_acc_greedy']:.4f} gap={r['gap_greedy']:.4f} "
              f"ratio={r['rejection_ratio']:.1f}x")


if __name__ == "__main__":
    main()
