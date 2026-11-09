"""Batch-size scan of the two cost ratios (appendix check).

For B in {1,4,8,16}, measures one target token pass, one draft token pass, and
the target verification pass over gamma+1 positions with KV caches.  Reports
rho_B = T_D(B,1)/T_T(B,1), tau_B = T_T(B,gamma+1)/T_T(B,1), and the cost
gamma*rho_B+tau_B.  This is a sanity check for the claim that the FLOP frontier
is the optimistic limit; it is not used as a production throughput benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device


@torch.no_grad()
def timed(fn, repeats=10):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--widths", default="0.5,0.98")
    ap.add_argument("--batch-sizes", default="1,4,8,16")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        for i, layer in enumerate(draft.model.layers):
            layer.mlp, _ = make_narrow(layer, width, "width", dev,
                                       imp.get(f"l{i:02d}"), pad_to=128)
        for B in [int(x) for x in a.batch_sizes.split(",")]:
            prefix = torch.randint(100, 200000, (B, a.prefix), dtype=torch.long, device=dev)
            one = torch.randint(100, 200000, (B, 1), dtype=torch.long, device=dev)
            verify = torch.randint(100, 200000, (B, a.gamma + 1), dtype=torch.long, device=dev)
            with torch.no_grad():
                # Prefill through the backbone only: the full forward would
                # materialize [B, prefix, vocab] logits (65GB at B=128).
                pt = target.model(input_ids=prefix, use_cache=True).past_key_values
                pd = draft.model(input_ids=prefix, use_cache=True).past_key_values

            def t_one():
                c = copy.deepcopy(pt)
                target(input_ids=one, past_key_values=c, use_cache=True)

            def t_verify():
                c = copy.deepcopy(pt)
                target(input_ids=verify, past_key_values=c, use_cache=True)

            def d_one():
                c = copy.deepcopy(pd)
                draft(input_ids=one, past_key_values=c, use_cache=True)

            tt = statistics.median(timed(t_one, a.repeats))
            tv = statistics.median(timed(t_verify, a.repeats))
            td = statistics.median(timed(d_one, a.repeats))
            rho = td / tt
            tau = tv / tt
            rows.append({"width": width, "batch": B, "target_one_s": tt,
                         "draft_one_s": td, "target_verify_s": tv,
                         "rho": rho, "tau": tau,
                         "cost": a.gamma * rho + tau})
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.empty_cache()
        del draft
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "prefix": a.prefix, "gamma": a.gamma, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
