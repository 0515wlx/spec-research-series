"""Aggregate the real standard-SD loop runs and compare them with cache estimates."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from infra.std_sd_stats import DRAFTS, simulate_tau, split_by_seq

OUT = "out/std_sd"


def boot_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    rows = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*_g*.json"))):
        name = os.path.basename(path)
        if name.startswith("pilot") or name.startswith("stats"):
            continue
        d = json.load(open(path))
        if not d.get("rows"):
            continue
        dom, draft, gamma = d["domain"], d["draft"], d["gamma"]
        key = f"{dom}/{draft}/g{gamma}"
        vals = {k: [r[k] for r in d["rows"] if r.get(k) is not None]
                for k in ["alpha_empirical", "alpha_token", "E_TV_loop",
                          "q_greedy_loop", "tau"]}
        rec = {"domain": dom, "draft": draft, "gamma": gamma, "n": len(d["rows"])}
        for k, v in vals.items():
            rec[k] = float(np.mean(v)) if v else None
            rec[k + "_ci95"] = boot_ci(v) if v else None
        # Cache estimates on the same domain/draft.
        jp = f"out/accept/{dom}_{draft}.json"
        npz = f"out/accept/{dom}_{draft}.npz"
        if os.path.exists(jp):
            j = json.load(open(jp))
            rec["cache_q_greedy"] = float(j["q"])
            rec["cache_E_TV"] = float(j["E_TV"])
            rec["cache_alpha_token"] = float(1.0 - j["E_TV"])
            rec["cache_tau_inf"] = float(j["tau_inf"])
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            parts = split_by_seq(z["tv"].astype(float), z["seq_id"])
            rec["cache_tau_sim"] = simulate_tau(np.concatenate(parts), gamma)
        # Ratios that test the cache approximation.
        if rec.get("cache_E_TV") is not None and rec.get("E_TV_loop") is not None:
            rec["E_TV_ratio_loop_over_cache"] = rec["E_TV_loop"] / max(rec["cache_E_TV"], 1e-12)
        if rec.get("cache_tau_sim") and rec.get("tau"):
            rec["tau_ratio_loop_over_sim"] = rec["tau"] / rec["cache_tau_sim"]
        rows[key] = rec

    with open(os.path.join(OUT, "loop_summary.json"), "w") as f:
        json.dump(rows, f, indent=1)

    lines = ["# Real standard-SD loop validation", "",
             "| cell | n | alpha_emp | alpha_token | E[TV]_loop | tau | cache E[TV] | "
             "tau_sim | E[TV] ratio | tau ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rows.items():
        def f(x, nd=3):
            return "n/a" if x is None else f"{x:.{nd}f}"
        lines.append(
            f"| {k} | {r['n']} | {f(r['alpha_empirical'])} | {f(r['alpha_token'])} | "
            f"{f(r['E_TV_loop'])} | {f(r['tau'], 2)} | {f(r.get('cache_E_TV'))} | "
            f"{f(r.get('cache_tau_sim'), 2)} | {f(r.get('E_TV_ratio_loop_over_cache'))} | "
            f"{f(r.get('tau_ratio_loop_over_sim'))} |")
    with open(os.path.join(OUT, "loop_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", os.path.join(OUT, "loop_summary.json"))


if __name__ == "__main__":
    main()
