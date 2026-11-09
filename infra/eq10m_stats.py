"""Summarise the equal-token (10M) distillation series."""
from __future__ import annotations

import json
import os

import numpy as np

SETS = [("MATH-500", "math500"), ("WikiText raw", "wikitext"),
        ("WikiText gen", "wikitext_gen"), ("FineWeb raw", "fineweb"),
        ("FineWeb gen", "fineweb_gen")]
SIZES = [("0.8b", 0.8), ("2b", 2.0), ("4b", 4.0), ("9b", 9.0), ("27b", 27.0)]
OUT = "out/accept/eq10m"


def q(path):
    return json.load(open(path))["q"] if os.path.exists(path) else None


def main():
    rows = []
    for key, D in SIZES:
        rec = {"size": key, "D": D}
        for label, dom in SETS:
            off = q(f"out/accept/{dom}_qwen3.5-{key}.json")
            eq = q(os.path.join(OUT, f"{key}_{dom}.json"))
            rec[dom] = {"off": off, "eq": eq,
                        "delta": (eq - off) if (eq is not None and off is not None) else None,
                        "ratio": (eq / off) if (eq not in (None, 0) and off) else None}
        rows.append(rec)

    lines = ["# Equal-token 10M distillation series", "",
             "All runs use `distill_controlled.jsonl`, 10M tokens, lr 5e-5, warmup 50.",
             "", "| size | set | off-the-shelf q | 10M q | Δ | ratio |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        for label, dom in SETS:
            v = r[dom]
            if v["off"] is None:
                continue
            if v["eq"] is None:
                lines.append(f"| {r['size']} | {label} | {v['off']:.5f} | n/a | n/a | n/a |")
            else:
                lines.append(f"| {r['size']} | {label} | {v['off']:.5f} | {v['eq']:.5f} | "
                             f"{v['delta']:+.5f} | {v['ratio']:.2f}× |")
    for label, dom in SETS:
        pts = [(r["D"], r[dom]["eq"]) for r in rows if r[dom]["eq"] is not None]
        if len(pts) >= 3:
            D = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            alpha = float(-np.polyfit(np.log(D), np.log(y), 1)[0])
            lines += ["", f"- {label}: equal-token exponent over {len(pts)} sizes = {alpha:.3f}"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
