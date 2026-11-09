"""Summarise the positive-control lr x token grid."""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

BASE = {d: json.load(open(f"out/accept/{d}_qwen3.5-2b.json"))["q"]
        for d in ["fineweb", "fineweb_gen", "wikitext", "wikitext_gen", "math500"]}
DOMAINS = list(BASE)


def main():
    rows = defaultdict(lambda: defaultdict(list))
    for p in glob.glob("out/accept/positive_grid/*.json"):
        b = os.path.basename(p)
        m = re.match(r".*-adapt(\d+)-lr([0-9e.-]+)-s(\d+)_([a-z0-9_]+)\.json", b)
        if not m:
            continue
        t, lr, _s, dom = m.groups()
        rows[(float(lr), int(t))][dom].append(json.load(open(p))["q"])
    # Reuse the three existing 5e-6 / 0.5M seeds.
    for p in glob.glob("out/accept/positive_seeds/*-lr5e-6-s*_*.json"):
        b = os.path.basename(p)
        m = re.match(r".*-s(\d+)_([a-z0-9_]+)\.json", b)
        if m:
            rows[(5e-6, 500000)][m.group(2)].append(json.load(open(p))["q"])

    out = {}
    lines = ["# Positive control: learning rate x token budget (2B, seeds 0/1)", "",
             "Δq vs off-the-shelf 2B; mean over seeds (the 5e-6/0.5M cell has 3 seeds).", "",
             "| lr | tokens | " + " | ".join(DOMAINS) + " |",
             "|---:|---:|" + "---:|" * len(DOMAINS)]
    for (lr, t), v in sorted(rows.items()):
        cells = []
        rec = {"lr": lr, "tokens": t}
        for dom in DOMAINS:
            x = float(np.mean(v[dom])) if v[dom] else None
            rec[dom] = x
            rec[dom + "_delta"] = (x - BASE[dom]) if x is not None else None
            cells.append("n/a" if x is None else
                         f"{x - BASE[dom]:+.4f} ({(x / BASE[dom] - 1) * 100:+.1f}%)")
        lines.append(f"| {lr:.0e} | {t} | " + " | ".join(cells) + " |")
        out[f"lr{lr}_t{t}"] = rec
    with open("out/accept/positive_grid/summary.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("out/accept/positive_grid/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
