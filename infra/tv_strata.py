"""Decompose the shallow TV / q capacity exponent by target-margin stratum.

The reviewer asked why standard-SD q_s = E[TV] also decays slowly with draft
capacity, and whether that is a property of TV itself or of the measurement
distribution.  We stratify tokens by the target top-1 margin and report, for
each draft, q and E[TV] inside each stratum, plus the local exponent within the
stratum.  This shows whether the shallow aggregate exponent is driven by
low-margin tokens whose TV stays high.
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = "out/tv_strata"
os.makedirs(OUT, exist_ok=True)

DRAFTS = [("qwen3.5-0.8b", 0.8), ("qwen3.5-2b", 2.0), ("qwen3.5-4b", 4.0),
          ("qwen3.5-9b", 9.0), ("qwen3.5-27b", 27.0)]
SETS = [("MATH-500", "math500_"), ("WikiText raw", "wikitext_"),
        ("FineWeb raw", "fineweb_"), ("WikiText gen", "wikitext_gen_"),
        ("FineWeb gen", "fineweb_gen_")]


def load(prefix):
    out = {}
    for name, D in DRAFTS:
        p = f"out/accept/{prefix}{name}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        out[name] = {"D": D, "margin": z["margin"].astype(float),
                     "tv": z["tv"].astype(float), "match": z["match"].astype(bool)}
    return out


def slope(D, y):
    D = np.asarray(D, float)
    y = np.asarray(y, float)
    ok = y > 0
    return float(-np.polyfit(np.log(D[ok]), np.log(y[ok]), 1)[0])


def main():
    summary = {}
    for label, prefix in SETS:
        rows = load(prefix)
        if len(rows) < 3:
            continue
        names = list(rows)
        margin = rows[names[0]]["margin"]
        n = len(margin)
        # Quintiles of the target margin; Q1 = smallest margins.
        edges = np.quantile(margin, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges[-1] += 1e-9
        bins = np.clip(np.digitize(margin, edges[1:-1]), 0, 4)
        D = [rows[nm]["D"] for nm in names]
        entry = {"strata": {}}
        for b in range(5):
            m = bins == b
            qs, tvs = [], []
            for nm in names:
                qs.append(1.0 - rows[nm]["match"][m].mean())
                tvs.append(rows[nm]["tv"][m].mean())
            entry["strata"][f"Q{b+1}"] = {
                "n": int(m.sum()),
                "margin_lo": float(edges[b]), "margin_hi": float(edges[b + 1]),
                "q": qs, "tv": tvs,
                "alpha_q": slope(D, qs), "beta_tv": slope(D, tvs),
                "q_reduction": float(qs[0] / qs[-1]) if qs[-1] > 0 else None,
                "tv_reduction": float(tvs[0] / tvs[-1]) if tvs[-1] > 0 else None,
            }
        # Share of aggregate E[TV] contributed by each stratum (at 0.8B and 27B).
        for nm in [names[0], names[-1]]:
            tot = rows[nm]["tv"].sum()
            entry.setdefault("tv_share", {})[nm] = [
                float(rows[nm]["tv"][bins == b].sum() / tot) for b in range(5)
            ]
        summary[label] = entry
        print(f"\n== {label} ==")
        print("  stratum  n      margin range       q(.8->27)      TV(.8->27)   alpha_q  beta_TV")
        for b in range(5):
            s = entry["strata"][f"Q{b+1}"]
            print(f"  Q{b+1}  {s['n']:6d}  {s['margin_lo']:.3f}-{s['margin_hi']:.3f}  "
                  f"{s['q'][0]:.3f}->{s['q'][-1]:.3f}  {s['tv'][0]:.3f}->{s['tv'][-1]:.3f}  "
                  f"{s['alpha_q']:.3f}   {s['beta_tv']:.3f}")

    with open(os.path.join(OUT, "tv_strata.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = ["# TV/q decomposition by target-margin stratum", "",
             "Q1 is the lowest-margin quintile. Local exponents are over 0.8B--27B.",
             "", "| set | stratum | margin range | q 0.8B | q 27B | TV 0.8B | TV 27B | alpha_q | beta_TV |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, e in summary.items():
        for b in range(5):
            s = e["strata"][f"Q{b+1}"]
            lines.append(f"| {label} | Q{b+1} | {s['margin_lo']:.3f}-{s['margin_hi']:.3f} | "
                         f"{s['q'][0]:.3f} | {s['q'][-1]:.3f} | {s['tv'][0]:.3f} | "
                         f"{s['tv'][-1]:.3f} | {s['alpha_q']:.3f} | {s['beta_tv']:.3f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved", os.path.join(OUT, "tv_strata.json"), "and summary.md")


if __name__ == "__main__":
    main()
