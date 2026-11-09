"""Summarise the multi-seed gentle positive control (2B, 500k, lr 5e-6)."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

OUT = "out/accept/positive_seeds"
DOMAINS = ["fineweb", "fineweb_gen", "wikitext", "wikitext_gen", "math500"]
BASE = "out/accept"


def main():
    rows = {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        b = os.path.basename(p)
        m = re.match(r".*-s(\d+)_([a-z0-9_]+)\.json$", b)
        if not m:
            continue
        seed, dom = int(m.group(1)), m.group(2)
        d = json.load(open(p))
        rows.setdefault(dom, {})[seed] = d

    lines = ["# Multi-seed gentle positive control (2B, 500k tokens, lr 5e-6)", "",
             "| set | off-the-shelf q | seeds q (mean ± sd) | Δ vs base | all seeds improve? |",
             "|---|---:|---|---:|---|"]
    summary = {}
    for dom in DOMAINS:
        if dom not in rows:
            continue
        base_p = os.path.join(BASE, f"{dom}_qwen3.5-2b.json")
        if not os.path.exists(base_p):
            continue
        base = json.load(open(base_p))["q"]
        qs = [rows[dom][s]["q"] for s in sorted(rows[dom])]
        mean, sd = float(np.mean(qs)), float(np.std(qs, ddof=1)) if len(qs) > 1 else 0.0
        lines.append(f"| {dom} | {base:.5f} | {mean:.5f} ± {sd:.5f} | "
                     f"{mean - base:+.5f} ({(mean/base-1)*100:+.2f}%) | "
                     f"{all(q < base for q in qs)} |")
        summary[dom] = {"base": base, "q": qs, "mean": mean, "sd": sd,
                        "delta": mean - base, "all_improve": all(q < base for q in qs)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
