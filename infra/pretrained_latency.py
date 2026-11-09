"""Measure pretrained-draft latency against the 9B target (appendix check).

Reports the same batch-1 per-token pass ratio used in the compressed-draft
table, so the matched-cost comparison can use measured rather than nominal
costs for both sides.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen3.5-9b")
    ap.add_argument("--drafts", default="qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    prefix = torch.randint(100, 200000, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, 200000, (1, 1), dtype=torch.long, device=dev)

    target = load_text_model(a.target, dtype="bfloat16", device_map={"": dev})
    with torch.no_grad():
        pt = target(input_ids=prefix, use_cache=True).past_key_values

    def target_one():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    base = statistics.median(timed(target_one, a.repeats))
    del target
    torch.cuda.empty_cache()

    rows = []
    for name in a.drafts.split(","):
        model = load_text_model(name, dtype="bfloat16", device_map={"": dev})
        with torch.no_grad():
            pc = model(input_ids=prefix, use_cache=True).past_key_values

        def draft_one():
            c = copy.deepcopy(pc)
            model(input_ids=one, past_key_values=c, use_cache=True)

        t = statistics.median(timed(draft_one, a.repeats))
        rows.append({"draft": name, "target_one_s": base, "draft_one_s": t,
                     "rho_measured": t / base})
        print(json.dumps(rows[-1]), flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"target": a.target, "prefix": a.prefix, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
