"""Compiled/fused narrow-MLP latency check (review request: kernel closure).

The paper's measured batch-1 cost uses a padded narrow MLP with ordinary dense
GEMMs.  A reviewer reasonably asks how close a fused/compiled kernel can get to
the roofline.  This script measures, on the same GPU and prefix:

  * target one-token pass and gamma+1 verification pass;
  * compressed draft one-token pass with the padded narrow MLP (uncompiled);
  * the same draft after ``torch.compile`` (Inductor, max-autotune);
  * the corresponding nominal FLOP and weight-byte ratios.

It reports rho_meas, rho_FLOP, rho_mem, tau, and the fraction of the ideal
saving realized: eta = (1-rho_meas)/(1-rho_mem).  A value near 1 means the
implementation is close to the memory roofline; a value near 0 means launch or
padding overhead dominates.  The point is not that compilation changes the
paper's no-go; it is to show whether a better kernel can move rho_1 toward the
frontier, and whether that frontier is still above the break-even.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.flops import param_stats
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device
from infra.model_registry import resolve


@torch.no_grad()
def timed(fn, repeats=20):
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / 1000.0)
    return ts


def ratios(model_key, width):
    s = param_stats(resolve(model_key))
    phi_flop = s.mlp / s.compute
    phi_bytes = s.mlp / s.total
    return {
        "rho_FLOP": float(1.0 - phi_flop * (1.0 - width)),
        "rho_mem": float(1.0 - phi_bytes * (1.0 - width)),
        "phi_flop": float(phi_flop),
        "phi_bytes": float(phi_bytes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--widths", default="0.5,0.98")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--compile-scope", default="mlp", choices=["mlp", "model"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda:0"
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    prefix = torch.randint(100, 200000, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, 200000, (1, 1), dtype=torch.long, device=dev)
    verify = torch.randint(100, 200000, (1, a.gamma + 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        pt = target(input_ids=prefix, use_cache=True).past_key_values

    def target_one():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    def target_verify():
        c = copy.deepcopy(pt)
        target(input_ids=verify, past_key_values=c, use_cache=True)

    t1 = statistics.median(timed(target_one, a.repeats))
    tv = statistics.median(timed(target_verify, a.repeats))
    rows = [{"kind": "target", "target_one_s": t1, "target_verify_s": tv,
             "tau": tv / t1}]
    print(json.dumps(rows[-1]), flush=True)
    del target
    torch.cuda.empty_cache()

    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        for i, layer in enumerate(draft.model.layers):
            layer.mlp, _ = make_narrow(layer, width, "width", dev,
                                       imp.get(f"l{i:02d}") if imp else None, pad_to=128)
        with torch.no_grad():
            pd = draft(input_ids=prefix, use_cache=True).past_key_values

        def draft_one(model=draft, past=pd):
            c = copy.deepcopy(past)
            model(input_ids=one, past_key_values=c, use_cache=True)

        for _ in range(5):
            draft_one()
        base = statistics.median(timed(draft_one, a.repeats))
        ratio = ratios(a.model, width)
        row = {"kind": "padded", "width": width, "draft_one_s": base,
               "rho_meas": base / t1, "tau": tv / t1, **ratio}
        row["eta"] = (1.0 - row["rho_meas"]) / max(1e-9, 1.0 - row["rho_mem"])
        rows.append(row)
        print(json.dumps(row), flush=True)

        try:
            if a.compile_scope == "mlp":
                for layer in draft.model.layers:
                    layer.mlp = torch.compile(layer.mlp, mode=a.compile_mode,
                                              fullgraph=False)
                compiled = draft
            else:
                compiled = torch.compile(draft, mode=a.compile_mode,
                                         fullgraph=False)
            with torch.no_grad():
                pc = compiled(input_ids=prefix, use_cache=True).past_key_values

            def draft_one_c(model=compiled, past=pc):
                c = copy.deepcopy(past)
                model(input_ids=one, past_key_values=c, use_cache=True)

            for _ in range(5):
                draft_one_c()
            comp = statistics.median(timed(draft_one_c, a.repeats))
            row = {"kind": "compiled", "width": width, "draft_one_s": comp,
                   "rho_meas": comp / t1, "tau": tv / t1, **ratio}
            row["eta"] = (1.0 - row["rho_meas"]) / max(1e-9, 1.0 - row["rho_mem"])
            rows.append(row)
            print(json.dumps(row), flush=True)
        except Exception as e:
            rows.append({"kind": "compiled", "width": width, "error": repr(e)})
            print("compile failed:", repr(e), flush=True)

        del draft
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "prefix": a.prefix, "gamma": a.gamma,
                   "rows": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
