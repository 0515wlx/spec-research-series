"""Turn rank-sweep rejection measurements into a batch-1 cost frontier.

For a target with per-pass verification cost ``C_T``, draft per-pass cost
``C_D = rho C_T`` and per-token acceptance ``alpha = 1 - q``, the expected
accepted tokens per speculative step are

    A(alpha, gamma) = sum_{i=0}^{gamma} alpha^i
                    = (1 - alpha^{gamma+1}) / (1 - alpha),

the step costs ``gamma C_D + C_T``, and the batch-1 speedup is

    S = A(alpha, gamma) / (1 + gamma rho).

``S >= 1`` is equivalent to ``A(alpha, gamma) >= 1 + gamma rho``.  We solve
that equation for the break-even rejection ``q_star`` and compare it with the
measured ``q`` (and its cluster-bootstrap interval).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def accepted(alpha: float, gamma: int) -> float:
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(q: float, rho: float, gamma: int) -> float:
    return accepted(1.0 - q, gamma) / (1.0 + gamma * rho)


def q_star(rho: float, gamma: int) -> float:
    """Solve A(1-q, gamma) = 1 + gamma rho on [0,1]."""
    target = 1.0 + gamma * rho
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if accepted(1.0 - mid, gamma) >= target:
            lo = mid          # q still small enough
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_scan(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        if "q" in r and "frac" in r:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="glob of rank-scan json files")
    ap.add_argument("--model", required=True, help="model key for cost accounting")
    ap.add_argument("--gammas", default="4,8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="svd")
    a = ap.parse_args()

    from infra.svd_compress import model_cost_ratio
    gammas = [int(x) for x in a.gammas.split(",")]
    rows = load_scan(a.scan)
    out = []
    for r in rows:
        cost = model_cost_ratio(a.model, float(r["frac"]), a.mode)
        rho = cost["total_ratio"]
        rec = dict(r)
        rec["rho"] = rho
        for g in gammas:
            rec[f"q_star_g{g}"] = q_star(rho, g)
            rec[f"speedup_g{g}"] = speedup(r["q"], rho, g)
            rec[f"accepted_per_cost_g{g}"] = accepted(1.0 - r["q"], g) / (1.0 + g * rho)
            lo, hi = rec["q_ci"]
            # S is decreasing in q, so the interval flips.
            rec[f"speedup_ci_g{g}"] = [speedup(hi, rho, g), speedup(lo, rho, g)]
            rec[f"no_go_g{g}"] = bool(lo > rec[f"q_star_g{g}"])
        out.append(rec)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out} ({len(out)} rows)")
    for r in sorted(out, key=lambda x: x["frac"]):
        line = (f"frac={r['frac']:<5g} rho={r['rho']:.3f} q={r['q']:.4f} "
                f"CI=[{r['q_ci'][0]:.4f},{r['q_ci'][1]:.4f}]")
        for g in gammas:
            line += (f" | g{g}: q*={r[f'q_star_g{g}']:.3f} "
                     f"S={r[f'speedup_g{g}']:.2f} no-go={r[f'no_go_g{g}']}")
        print(line)


if __name__ == "__main__":
    main()
