"""Fit q(D) scaling models and plot the capacity wall."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from infra.metrics import bootstrap_floor, fit_models
from infra import model_registry

PAT = re.compile(r"(?P<domain>[a-z0-9]+)_(?P<model>qwen3(?:\.[568])?-[0-9.]+b)")


def parse_path(p: str) -> tuple[str, str, float] | None:
    base = os.path.basename(p).replace(".json", "")
    m = PAT.search(base)
    if not m:
        return None
    domain, model = m.group("domain"), m.group("model")
    if model in model_registry.MODELS:
        return domain, model, model_registry.MODELS[model].params_b
    mm = re.search(r"([0-9.]+)b", model)
    return (domain, model, float(mm.group(1))) if mm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/accept/*.json")
    ap.add_argument("--out", default="out/fit/capacity")
    ap.add_argument("--min-points", type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    grouped: dict[str, dict] = {}
    for p in sorted(glob.glob(a.glob)):
        parsed = parse_path(p)
        if not parsed:
            continue
        domain, model, D = parsed
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "q" not in r:
            continue
        grouped.setdefault(domain, {})[D] = {
            "model": model, "path": p, "q": r["q"], "E_TV": r.get("E_TV"),
            "tau_inf": r.get("tau_inf"), "q_per_seq": r.get("q_per_seq"),
            "tokens": r.get("tokens"),
        }

    summary = {}
    for domain, pts in grouped.items():
        if len(pts) < a.min_points:
            print(f"[skip] {domain}: only {len(pts)} points")
            continue
        D = np.array(sorted(pts))
        q = np.array([pts[d]["q"] for d in D])
        fits = fit_models(D, q)
        boot = {"q_inf_mean": None, "q_inf_ci95": None, "n": 0}
        if all(pts[d].get("q_per_seq") for d in D):
            boot = bootstrap_floor(D, [np.asarray(pts[d]["q_per_seq"]) for d in D])
        entry = {
            "domain": domain,
            "D": D.tolist(),
            "q": q.tolist(),
            "tau_inf": [pts[d].get("tau_inf") for d in D],
            "models": [pts[d]["model"] for d in D],
            "fits": fits,
            "bootstrap_q_inf": boot,
        }
        summary[domain] = entry
        print(f"\n== {domain} ==")
        for d in D:
            print(f"  D={d:5.1f}B  q={pts[d]['q']:.4f}  tau_inf={pts[d].get('tau_inf')}")
        for name in ["floor", "power", "loglinear", "invlog"]:
            f = fits.get(name, {})
            if "bic" in f:
                print(f"  {name:9s} R2={f['r2']:.4f} BIC={f['bic']:.2f} params={['%.4g' % x for x in f['params']]}")
        print(f"  best={fits.get('_best_by_bic')}  q_inf_boot={boot}")

    with open(os.path.join(a.out, "fit.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nsaved", os.path.join(a.out, "fit.json"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axes[0]
        for domain, e in summary.items():
            ax.plot(e["D"], e["q"], "o-", label=domain)
        ax.set_xscale("log")
        ax.set_xlabel("draft capacity D (B params)")
        ax.set_ylabel("q = 1 - agreement")
        ax.set_title("(a) q vs capacity")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1]
        for domain, e in summary.items():
            ax.plot(e["D"], e["tau_inf"], "o-", label=domain)
        ax.set_xscale("log")
        ax.set_xlabel("draft capacity D (B)")
        ax.set_ylabel(r"$\tau_\infty = 1/q$")
        ax.set_title("(b) long-gamma acceptance length")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

        ax = axes[2]
        domains = list(summary)
        vals = [summary[d]["bootstrap_q_inf"].get("q_inf_mean") for d in domains]
        lo = [summary[d]["bootstrap_q_inf"].get("q_inf_ci95", [None, None])[0] for d in domains]
        hi = [summary[d]["bootstrap_q_inf"].get("q_inf_ci95", [None, None])[1] for d in domains]
        x = np.arange(len(domains))
        ax.bar(x, vals, color="steelblue")
        if all(v is not None for v in lo):
            ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                        fmt="none", ecolor="k", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(domains, rotation=30, ha="right")
        ax.set_ylabel(r"$q_\infty$ (bootstrap mean)")
        ax.set_title("(c) irreducible OOD floor")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "capacity_wall.png"), dpi=160)
        print("saved", os.path.join(a.out, "capacity_wall.png"))
    except Exception as e:
        print("plot skipped:", repr(e))


if __name__ == "__main__":
    main()
