"""Aggregate the generated-OOD standard-SD loop and compare with the cache."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

OUT = "out/std_sd"
DRAFTS = [("qwen3.5-0.8b", 0.8), ("qwen3.5-2b", 2.0), ("qwen3.5-4b", 4.0),
          ("qwen3.5-9b", 9.0), ("qwen3.5-27b", 27.0)]
DOMAINS = ["wikitext_gen", "fineweb_gen"]


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(-np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    summary = {}
    for dom in DOMAINS:
        rows = []
        for name, D in DRAFTS:
            p = os.path.join(OUT, f"gen_{dom}_{name}_g4.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            cache_p = f"out/accept/{dom}_{name}.json"
            cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
            rows.append({
                "draft": name, "D": D, "n": d["n"],
                "tau_loop": d["tau"], "alpha_emp": d["alpha_empirical"],
                "alpha_token": d["alpha_token"], "E_TV_loop": d["E_TV_loop"],
                "q_greedy_loop": d["q_greedy_loop"],
                "E_TV_cache": cache.get("E_TV"), "q_greedy_cache": cache.get("q"),
                "tau_sim_cache": cache.get("per_gamma", {}).get("4", {}).get("tau"),
                "E_TV_ratio": (d["E_TV_loop"] / cache["E_TV"]) if cache.get("E_TV") else None,
                "q_ratio": (d["q_greedy_loop"] / cache["q"]) if cache.get("q") else None,
            })
        if len(rows) < 3:
            continue
        D = [r["D"] for r in rows]
        entry = {"rows": rows}
        for k in ["tau_loop", "E_TV_loop", "q_greedy_loop", "E_TV_cache", "q_greedy_cache"]:
            entry[f"exp_{k}"] = slope(D, [r[k] for r in rows])
        summary[dom] = entry
        print(f"\n== {dom} ==")
        print("draft       tau_loop  ETV_loop  q_g_loop  ETV_cache  q_g_cache  ETV ratio  q ratio")
        for r in rows:
            print(f"{r['draft']:16s} {r['tau_loop']:.2f}     {r['E_TV_loop']:.3f}    "
                  f"{r['q_greedy_loop']:.3f}     {r['E_TV_cache']:.3f}      {r['q_greedy_cache']:.3f}      "
                  f"{r['E_TV_ratio']:.2f}      {r['q_ratio']:.2f}")
        print(f"loop exponents: tau={entry['exp_tau_loop']:.3f} "
              f"E[TV]={entry['exp_E_TV_loop']:.3f} q_g={entry['exp_q_greedy_loop']:.3f}  "
              f"cache: E[TV]={entry['exp_E_TV_cache']:.3f} q_g={entry['exp_q_greedy_cache']:.3f}")

    with open(os.path.join(OUT, "gen_loop_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = ["# Standard-SD real loop on generated OOD", "",
             "30 prompts/cell, gamma=4, 96 new tokens, all five drafts.", "",
             "| set | draft | tau | E[TV] loop | q_g loop | E[TV] cache | q_g cache | E[TV] ratio | q ratio |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dom, e in summary.items():
        for r in e["rows"]:
            lines.append(f"| {dom} | {r['draft']} | {r['tau_loop']:.2f} | "
                         f"{r['E_TV_loop']:.3f} | {r['q_greedy_loop']:.3f} | "
                         f"{r['E_TV_cache']:.3f} | {r['q_greedy_cache']:.3f} | "
                         f"{r['E_TV_ratio']:.2f} | {r['q_ratio']:.2f} |")
        lines.append(f"| {dom} | exponent | {e['exp_tau_loop']:.3f} | "
                     f"{e['exp_E_TV_loop']:.3f} | {e['exp_q_greedy_loop']:.3f} | "
                     f"{e['exp_E_TV_cache']:.3f} | {e['exp_q_greedy_cache']:.3f} | | |")
    with open(os.path.join(OUT, "gen_loop_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved", os.path.join(OUT, "gen_loop_summary.json"))


if __name__ == "__main__":
    main()
