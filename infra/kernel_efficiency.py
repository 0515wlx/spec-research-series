"""Break-even kernel-efficiency ratio for the post-hoc draft.

If the draft-side kernel has effective throughput c times the target's, the
relative pass ratio is rho_eff = rho_FLOP / c, and break-even is

    A(1-q, gamma) = gamma * rho_FLOP / c + tau.

For a measured q we solve for the minimum c that reopens the window; for a given
c we solve for the break-even q*(c).  This quantifies the same-kernel /
equal-efficiency assumption rather than leaving it qualitative.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from infra.cost_frontier import accepted

WIDTHS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
MLP_SHARE = 0.6088
GAMMA = 4


def q_star_for_cost(cost: float, gamma=GAMMA) -> float:
    if cost >= gamma + 1:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if accepted(1.0 - mid, gamma) >= cost:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def min_c(rho: float, q: float, tau: float, gamma=GAMMA):
    need = accepted(1.0 - q, gamma) - tau
    if need <= 0:
        return None
    return gamma * rho / need


def load_q():
    out = {}
    for w in WIDTHS:
        qs = []
        for s in ("wikitext", "fineweb"):
            p = f"out/rank_scan/qwen3.5-9b_actwidth/{s}_frac{w:g}.npz"
            if os.path.exists(p):
                z = np.load(p)
                qs.append(1.0 - z["match"].astype(bool).mean())
        if qs:
            out[w] = float(np.mean(qs))
    return out


def load_tau():
    vals = []
    for p in glob.glob("out/latency/verify_9b_w*.json"):
        d = json.load(open(p))
        if "target_one_s" in d:
            vals.append(d["target_verify_s"] / d["target_one_s"])
    return float(np.mean(vals)) if vals else 1.32


def main():
    q = load_q()
    tau_meas = load_tau()
    rows = []
    for w, qv in q.items():
        rho = 1.0 - MLP_SHARE * (1.0 - w)
        rows.append({
            "width": w, "rho_flop": rho, "q": qv,
            "alpha_acc": 1.0 - qv,
            "c_star_tau1": min_c(rho, qv, 1.0),
            "c_star_measured": min_c(rho, qv, tau_meas),
            "q_star_c2": q_star_for_cost(GAMMA * rho / 2.0 + tau_meas),
            "q_star_c3": q_star_for_cost(GAMMA * rho / 3.0 + tau_meas),
            "q_star_c5": q_star_for_cost(GAMMA * rho / 5.0 + tau_meas),
        })
    # combined width 0.5 + W4
    p = "out/quant/combined_w0.5_w4_wikitext.json"
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append({
            "width": "0.5 + W4", "rho_flop": d["rho_flop"], "q": d["q"],
            "alpha_acc": 1.0 - d["q"],
            "c_star_tau1": min_c(d["rho_flop"], d["q"], 1.0),
            "c_star_measured": min_c(d["rho_flop"], d["q"], tau_meas),
            "q_star_c2": q_star_for_cost(GAMMA * d["rho_flop"] / 2.0 + tau_meas),
            "q_star_c3": q_star_for_cost(GAMMA * d["rho_flop"] / 3.0 + tau_meas),
            "q_star_c5": q_star_for_cost(GAMMA * d["rho_flop"] / 5.0 + tau_meas),
        })
    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/kernel_efficiency.json", "w") as f:
        json.dump({"tau_measured": tau_meas, "gamma": GAMMA, "rows": rows}, f, indent=1)

    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        (r"\caption{Break-even kernel-efficiency ratio. If the draft kernel has effective throughput "
         r"$c$ times the target's, the cost is $\gamma\rho_{\rm FLOP}/c+\tau$. $c^\star$ is the "
         r"minimum $c$ that reopens the window at the measured raw-text rejection; ``--'' means no "
         r"finite $c$ helps because even a free draft leaves the measured cost above $\gamma+1$. "
         r"The last three columns are $q^\star(c)$ at $c=2,3,5$ with the measured "
         rf"$\tau_1={tau_meas:.3f}$. The $0.5+$W4 row uses the composed operator's measured rejection.}}"),
        r"\label{tab:kernel_eff}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Width & $\rho_{\rm FLOP}$ & $q$ & $c^\star$ ($\tau=1$) & $c^\star$ ($\tau_1$) & $q^\star(c{=}2)$ & $q^\star(c{=}3)$ & $q^\star(c{=}5)$ \\",
        r"\midrule",
    ]
    for r in rows:
        w = r["width"]
        wtxt = w if isinstance(w, str) else f"{w:.2f}"
        def fmt(x):
            return "--" if x is None else f"{x:.1f}"
        lines.append(
            f"{wtxt} & {r['rho_flop']:.3f} & {r['q']:.4f} & "
            f"{fmt(r['c_star_tau1'])} & {fmt(r['c_star_measured'])} & "
            f"{r['q_star_c2']:.4f} & {r['q_star_c3']:.4f} & {r['q_star_c5']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open("paper/tab_kernel_eff.tex", "w") as f:
        f.write("\n".join(lines))
    print(json.dumps({"tau_measured": tau_meas, "rows": rows}, indent=1))
    print("wrote paper/tab_kernel_eff.tex")


if __name__ == "__main__":
    main()
