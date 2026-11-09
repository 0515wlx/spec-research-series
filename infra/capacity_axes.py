"""Re-express the capacity axis in total / compute / active parameters and FLOPs.

The paper's x-axis is nominal model size in billions.  A reviewer correctly
noted that the 27B dense draft is only "super-capacity" relative to the
target's *active* parameters, not its total parameters.  This script reports
the same q(D) points under four definitions and refits the local exponent:

  * total parameters            (checkpoint tensors, text backbone)
  * compute parameters          (matmul weights, excludes embedding lookup)
  * active parameters           (dense = total; MoE target = top-k experts)
  * forward FLOPs per token     (2N + attention, seq_len=512)

For the off-the-shelf dense drafts the four axes are near-proportional, so the
*local exponent* is almost invariant; what changes is the location of the 27B
point relative to the target and the implied absolute capacity multiple.
"""
from __future__ import annotations

import json
import os

import numpy as np

from infra import flops, model_registry

OUT = "out/capacity_axes"
os.makedirs(OUT, exist_ok=True)

DRAWN = [
    ("qwen3.5-0.8b", 0.8),
    ("qwen3.5-2b", 2.0),
    ("qwen3.5-4b", 4.0),
    ("qwen3.5-9b", 9.0),
    ("qwen3.5-27b", 27.0),
]
SETS = [
    ("MATH-500", "math500_", "out/accept"),
    ("GSM8K", "gsm8k_", "out/accept"),
    ("HumanEval", "humaneval_", "out/accept"),
    ("MT-Bench", "mtbench_", "out/accept"),
    ("WikiText raw", "wikitext_", "out/accept"),
    ("FineWeb raw", "fineweb_", "out/accept"),
    ("WikiText gen", "wikitext_gen_", "out/accept"),
    ("FineWeb gen", "fineweb_gen_", "out/accept"),
]


def active_params(key: str, stats: flops.ParamStats, cfg) -> float:
    """Active matmul parameters per token (top-k experts for MoE)."""
    spec = model_registry.get(key)
    if spec.kind != "moe":
        return float(stats.compute)
    n_exp = int(getattr(cfg, "num_experts", 0))
    k_exp = int(getattr(cfg, "num_experts_per_tok", 0))
    inter = int(getattr(cfg, "moe_intermediate_size", 0))
    shared = int(getattr(cfg, "shared_expert_intermediate_size", 0) or 0)
    hidden = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    # per-expert MLP (gate, up, down); shared expert is always active.
    per_expert = 3 * hidden * inter
    active_mlp = n_layers * ((k_exp + (1 if shared else 0)) * per_expert)
    # total matmul params minus the inactive experts.
    inactive = n_layers * max(n_exp - k_exp - (1 if shared else 0), 0) * per_expert
    return float(stats.compute - inactive)


def forward_flops(stats, cfg, n_active, seq_len=512):
    """Inference FLOPs/token: 2N for matmuls + attention term (forward only)."""
    train = flops.flops_per_token(stats, cfg, seq_len)  # 6N + 3*attn
    # train = 6*compute + 3*attn -> forward = 2*active + attn
    attn = (train - 6.0 * stats.compute) / 3.0
    return 2.0 * n_active + attn


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(-np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    axis = {}
    for key, _D in DRAWN + [("qwen3.5-35b-a3b", 35.0)]:
        p = model_registry.get(key).path
        stats = flops.param_stats(p)
        cfg = flops._config(p)
        act = active_params(key, stats, cfg)
        axis[key] = {
            "total_b": stats.total / 1e9,
            "compute_b": stats.compute / 1e9,
            "active_b": act / 1e9,
            "flops_per_token_fwd": forward_flops(stats, cfg, act),
            "kind": model_registry.get(key).kind,
        }

    out = {"axis_definitions": axis, "sets": {}}
    for label, prefix, adir in SETS:
        rows = []
        for key, _D in DRAWN:
            p = os.path.join(adir, f"{prefix}{key}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            rows.append({"key": key, "q": float(j["q"]), "E_TV": float(j["E_TV"]),
                         **axis[key]})
        if len(rows) < 3:
            continue
        q = [r["q"] for r in rows]
        entry = {"rows": rows}
        for ax in ["total_b", "compute_b", "active_b", "flops_per_token_fwd"]:
            x = [r[ax] for r in rows]
            entry[f"alpha_{ax}"] = slope(x, q)
        # q ratio and capacity multiple relative to the 27B probe.
        entry["q_ratio_0.8_over_27"] = rows[0]["q"] / rows[-1]["q"]
        entry["27b_over_target_active"] = axis["qwen3.5-27b"]["active_b"] / axis["qwen3.5-35b-a3b"]["active_b"]
        entry["27b_over_target_total"] = axis["qwen3.5-27b"]["total_b"] / axis["qwen3.5-35b-a3b"]["total_b"]
        entry["27b_over_target_flops"] = (axis["qwen3.5-27b"]["flops_per_token_fwd"] /
                                          axis["qwen3.5-35b-a3b"]["flops_per_token_fwd"])
        out["sets"][label] = entry
        print(f"\n== {label} ==")
        print("  alpha(total/compute/active/FLOPs): "
              + " / ".join(f"{entry['alpha_'+ax]:.3f}" for ax in
                           ["total_b", "compute_b", "active_b", "flops_per_token_fwd"]))
        print(f"  q(0.8)/q(27)={entry['q_ratio_0.8_over_27']:.3f}  "
              f"27B/target active={entry['27b_over_target_active']:.2f} "
              f"total={entry['27b_over_target_total']:.2f} "
              f"FLOPs={entry['27b_over_target_flops']:.2f}")

    with open(os.path.join(OUT, "capacity_axes.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# q(D) under different capacity definitions", "",
             "| model | total B | compute B | active B | fwd GFLOP/token |",
             "|---|---:|---:|---:|---:|"]
    for k, a in axis.items():
        lines.append(f"| {k} | {a['total_b']:.3f} | {a['compute_b']:.3f} | "
                     f"{a['active_b']:.3f} | {a['flops_per_token_fwd']/1e9:.2f} |")
    lines += ["", "| set | alpha total | alpha compute | alpha active | alpha FLOPs | q(0.8)/q(27) |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, e in out["sets"].items():
        lines.append(f"| {label} | {e['alpha_total_b']:.3f} | {e['alpha_compute_b']:.3f} | "
                     f"{e['alpha_active_b']:.3f} | {e['alpha_flops_per_token_fwd']:.3f} | "
                     f"{e['q_ratio_0.8_over_27']:.3f} |")
    lines += ["", f"27B / target active = {out['sets']['MATH-500']['27b_over_target_active']:.2f}x; "
              f"27B / target total = {out['sets']['MATH-500']['27b_over_target_total']:.2f}x; "
              f"27B / target forward FLOPs = {out['sets']['MATH-500']['27b_over_target_flops']:.2f}x."]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved", os.path.join(OUT, "capacity_axes.json"), "and summary.md")


if __name__ == "__main__":
    main()
