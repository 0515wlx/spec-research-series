"""Batch-size scan for a pretrained draft against the 9B target.

This complements ``pretrained_latency.py``: at batch 1 the 4B draft is
memory-bound and nearly as expensive as the target, so its nominal FLOP ratio
does not translate into a wall-clock saving.  Here we measure rho_B and tau_B
at B=1,4,8,16, so the matched-FLOP comparison can state when, if ever, the
pretrained 4B draft becomes cheaper per verified token.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.model_loader import load_text_model


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
    ap.add_argument("--target", default="qwen3.5-9b")
    ap.add_argument("--draft", default="qwen3.5-4b")
    ap.add_argument("--batch-sizes", default="1,4,8,16")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda:0"
    target = load_text_model(a.target, dtype="bfloat16", device_map={"": dev})
    draft = load_text_model(a.draft, dtype="bfloat16", device_map={"": dev})
    rows = []
    for B in [int(x) for x in a.batch_sizes.split(",")]:
        prefix = torch.randint(100, 200000, (B, a.prefix), dtype=torch.long, device=dev)
        one = torch.randint(100, 200000, (B, 1), dtype=torch.long, device=dev)
        verify = torch.randint(100, 200000, (B, a.gamma + 1), dtype=torch.long, device=dev)
        with torch.no_grad():
            pt = target(input_ids=prefix, use_cache=True).past_key_values
            pd = draft(input_ids=prefix, use_cache=True).past_key_values

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
        rho, tau = td / tt, tv / tt
        rows.append({"batch": B, "target_one_s": tt, "draft_one_s": td,
                     "target_verify_s": tv, "rho": rho, "tau": tau,
                     "cost": a.gamma * rho + tau})
        print(json.dumps(rows[-1]), flush=True)
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"target": a.target, "draft": a.draft, "prefix": a.prefix,
                   "gamma": a.gamma, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
