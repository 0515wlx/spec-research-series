"""Compare off-the-shelf vs trained 27B checkpoints on each evaluation domain.

Reads the ``out/accept/zero3/*.json`` files produced by
``scripts/18_eval_zero3_ckpts.sh`` plus the off-the-shelf
``out/accept/<domain>_qwen3.5-27b.json`` baseline, and reports whether the OOD
disagreement q moves with 5M/10M/20M tokens of distillation.

Usage:
  python -m infra.train_effect --glob "out/accept/zero3/*.json" \
      --out out/fit/zero3/train_effect.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

TOK = re.compile(r"tokens-(?P<n>\d+)")
DOMAINS = ["math500", "wikitext", "fineweb", "wikitext_gen", "fineweb_gen",
           "c4_gen", "gutenberg_gen", "code_gen"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def sem(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    return float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0


def paired_bootstrap(a, b, n: int = 2000, seed: int = 0) -> dict:
    """Paired sequence-level bootstrap of mean(b) - mean(a)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"mean": float(b.mean() - a.mean()), "ci95": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return {"mean": float(b.mean() - a.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/accept/zero3/*.json")
    ap.add_argument("--baseline-dir", default="out/accept")
    ap.add_argument("--model", default="qwen3.5-27b")
    ap.add_argument("--out", default="out/fit/zero3/train_effect.json")
    ap.add_argument("--latex", default=None,
                    help="also write a LaTeX table body to this path")
    a = ap.parse_args()

    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(a.glob)):
        base = os.path.basename(path)
        m = TOK.search(base)
        if not m:
            continue
        dom = next((d for d in DOMAINS if base.endswith(f"_{d}.json")), None)
        if dom is None:
            continue
        r = load(path)
        rows.setdefault(dom, {})[int(m.group("n"))] = {
            "q": r["q"], "E_TV": r.get("E_TV"), "q_h": r.get("q_h"),
            "tau_inf": r.get("tau_inf"), "tokens": r.get("tokens"),
            "q_per_seq": r.get("q_per_seq"), "path": path,
        }

    summary = {}
    for dom in DOMAINS:
        base_path = os.path.join(a.baseline_dir, f"{dom}_{a.model}.json")
        if not os.path.exists(base_path):
            continue
        b = load(base_path)
        pts = rows.get(dom, {})
        entry = {
            "baseline_q": b["q"], "baseline_E_TV": b.get("E_TV"),
            "baseline_q_h": b.get("q_h"),
            "baseline_q_per_seq": b.get("q_per_seq"),
            "checkpoints": {},
        }
        for n in sorted(pts):
            ck = {
                "q": pts[n]["q"], "E_TV": pts[n]["E_TV"],
                "q_h": pts[n]["q_h"], "tokens": pts[n]["tokens"],
                "delta_q": pts[n]["q"] - b["q"],
                "q_per_seq": pts[n]["q_per_seq"],
            }
            if b.get("q_per_seq") and pts[n].get("q_per_seq") \
                    and len(b["q_per_seq"]) == len(pts[n]["q_per_seq"]):
                ck["delta_q_paired_boot"] = paired_bootstrap(
                    b["q_per_seq"], pts[n]["q_per_seq"])
            entry["checkpoints"][n] = ck
        summary[dom] = entry

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"{'domain':16s} {'base q':>8s} " +
          " ".join(f"{n/1e6:>6.0f}M" for n in sorted(
              {k for e in summary.values() for k in e['checkpoints']})) +
          "   delta(20M)")
    for dom, e in summary.items():
        cks = sorted(e["checkpoints"])
        vals = " ".join(f"{e['checkpoints'][n]['q']:8.4f}" if n in e["checkpoints"]
                        else " " * 8 for n in sorted(
                            {k for x in summary.values() for k in x['checkpoints']}))
        last = e["checkpoints"][cks[-1]]["delta_q"] if cks else float("nan")
        print(f"{dom:16s} {e['baseline_q']:8.4f} {vals}   {last:+.4f}")
    print(f"saved {a.out}")

    if a.latex:
        names = {"math500": "MATH-500", "wikitext": "WikiText-2 raw",
                 "fineweb": "FineWeb-Edu raw", "wikitext_gen": "WikiText gen.",
                 "fineweb_gen": "FineWeb gen.", "c4_gen": "C4 gen.",
                 "gutenberg_gen": "Gutenberg gen.", "code_gen": "Code gen."}
        milestones = sorted({k for e in summary.values() for k in e["checkpoints"]})
        lines = []
        for dom, e in summary.items():
            cells = [f"{e['baseline_q']:.4f}"]
            for n in milestones:
                c = e["checkpoints"].get(n)
                cells.append(f"{c['q']:.4f}" if c else "--")
            ck = e["checkpoints"].get(milestones[-1]) if milestones else None
            d = ck.get("delta_q_paired_boot") if ck else None
            if d and d.get("ci95"):
                cells.append(f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]")
            elif ck:
                cells.append(f"{ck['delta_q']:+.4f}")
            lines.append(f"{names.get(dom, dom):18s} & " + " & ".join(cells) + r" \\")
        header = " & ".join(["Corpus", "27B off-the-shelf"] +
                            [f"{n/1e6:.0f}M" for n in milestones] + [r"$\Delta$(last)"])
        tex = ("% generated by python -m infra.train_effect\n"
               + header + r" \\" + "\n\\midrule\n" + "\n".join(lines) + "\n")
        os.makedirs(os.path.dirname(a.latex) or ".", exist_ok=True)
        with open(a.latex, "w") as f:
            f.write(tex)
        print(f"saved {a.latex}")


if __name__ == "__main__":
    main()
