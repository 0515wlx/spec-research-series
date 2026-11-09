"""Summarize the 2B deployment-distribution adaptation control.

Two runs are reported:
  * ``adapt2m``: 2M tokens at lr 2e-5 (aggressive; overfits);
  * ``adapt500k-lr5e6``: 500k tokens at lr 5e-6 (gentle; the positive control).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

OUT = "out/accept/positive"
DOMAINS = ["fineweb", "fineweb_gen", "wikitext", "wikitext_gen", "math500"]
BASE = {d: f"out/accept/{d}_qwen3.5-2b.json" for d in DOMAINS}
RUN_LABEL = {
    "qwen3.5-2b-fineweb-adapt2m": "aggressive lr 2e-5",
    "qwen3.5-2b-fineweb-adapt500k-lr5e6": "gentle lr 5e-6",
}


def boot_diff(a, b, n_boot=2000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    d = [a[rng.integers(0, n, n)].mean() - b[rng.integers(0, n, n)].mean()
         for _ in range(n_boot)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def main():
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        base = os.path.basename(path)
        if "_quick" in base:
            continue
        m = re.match(r"(qwen3\.5-2b-fineweb-adapt[^_]+)_(tokens-\d+)_"
                     r"(fineweb_gen|fineweb|wikitext_gen|wikitext|math500)\.json$", base)
        if not m:
            continue
        run, tag, dom = m.group(1), m.group(2), m.group(3)
        runs.setdefault((run, tag), {})[dom] = json.load(open(path))

    base = {d: json.load(open(p)) for d, p in BASE.items() if os.path.exists(p)}
    rows = []
    for (run, tag) in sorted(runs, key=lambda x: (x[0], int(x[1].split("-")[1]))):
        rec = {"run": RUN_LABEL.get(run, run), "tag": tag, "domains": {}}
        for dom in DOMAINS:
            if dom not in runs[(run, tag)] or dom not in base:
                continue
            d = runs[(run, tag)][dom]
            b = base[dom]
            rec["domains"][dom] = {
                "q": float(d["q"]), "E_TV": float(d["E_TV"]),
                "q_base": float(b["q"]), "E_TV_base": float(b["E_TV"]),
                "delta_q": float(d["q"] - b["q"]),
                "delta_q_ci95": boot_diff(d.get("q_per_seq", []), b.get("q_per_seq", [])),
            }
        rows.append(rec)

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)

    lines = ["# Positive control: 2B adaptation to target-generated FineWeb", "",
             "| run | tokens | " + " | ".join(DOMAINS) + " |",
             "|---|---:|---|" + "---|" * len(DOMAINS)]
    for r in rows:
        cells = []
        for dom in DOMAINS:
            v = r["domains"].get(dom)
            cells.append(f"{v['q']:.4f} ({v['delta_q']:+.4f})" if v else "n/a")
        tok = int(r["tag"].split("-")[1])
        lines.append(f"| {r['run']} | {tok/1e6:.2f}M | " + " | ".join(cells) + " |")
    lines += ["", "Off-the-shelf 2B baselines: " +
              ", ".join(f"{d}={base[d]['q']:.4f}" for d in DOMAINS if d in base) + ".", "",
              "Negative delta is better. The first 100 FineWeb prompts are held out from training."]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
