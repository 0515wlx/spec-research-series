"""Measured batch-1 cost for MoE active-expert reduction.

Loads the 35B-A3B target on one 80GB GPU, measures one target token pass and the
gamma+1 verification pass, patches the routers to top-k', and measures the draft
one-token pass.  Reports rho_1, tau_1, and gamma*rho+tau.  This is a service
sanity check; the main paper's MoE no-go is the nominal roofline comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.model_loader import load_text_model
from infra.moe_scan import _patch_router, _set_routers, _restore_routers


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
    ap.add_argument("--model", default="qwen3.5-35b-a3b")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--prefix", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda:0"
    _patch_router()
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(model.config, "vocab_size", 200000))
    high = min(200000, vocab - 2)
    prefix = torch.randint(100, high, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, high, (1, 1), dtype=torch.long, device=dev)
    verify = torch.randint(100, high, (1, a.gamma + 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        pt = model(input_ids=prefix, use_cache=True).past_key_values

    def t_one():
        c = copy.deepcopy(pt)
        model(input_ids=one, past_key_values=c, use_cache=True)

    def t_verify():
        c = copy.deepcopy(pt)
        model(input_ids=verify, past_key_values=c, use_cache=True)

    t1 = statistics.median(timed(t_one, a.repeats))
    tv = statistics.median(timed(t_verify, a.repeats))
    tau = tv / t1
    print(f"target one {t1:.4f}s verify {tv:.4f}s tau {tau:.3f}", flush=True)

    old = _set_routers(model, a.top_k, 1.0, None)
    with torch.no_grad():
        pd = model(input_ids=prefix, use_cache=True).past_key_values

    def d_one():
        c = copy.deepcopy(pd)
        model(input_ids=one, past_key_values=c, use_cache=True)

    for _ in range(3):
        d_one()
    d1 = statistics.median(timed(d_one, a.repeats))
    _restore_routers(old)
    rho = d1 / t1
    cost = a.gamma * rho + tau
    out = {"model": a.model, "top_k": a.top_k, "prefix": a.prefix,
           "gamma": a.gamma, "target_one_s": t1, "target_verify_s": tv,
           "draft_one_s": d1, "rho": rho, "tau": tau, "cost": cost,
           "cost_only_no_go": bool(cost > a.gamma + 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
