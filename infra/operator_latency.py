"""Measured batch-1 cost for non-FFN-width post-hoc operators.

The main paper measures FFN-width latency; the reviewer asks whether depth,
attention-head, and combined operators are also cost-only no-go in a real
implementation.  This script measures, on one GPU with KV caches:

  * target one-token pass and gamma+1 verification pass;
  * one-token pass after applying the operator in place;
  * rho_1 = draft_one / target_one, tau_1 = target_verify / target_one,
    cost = gamma * rho_1 + tau_1.

Modes: ``depth`` (drop last layers), ``head`` (physical KV-group removal),
``width_depth`` (FFN width + depth), ``width_head`` (FFN width + head).
Head/depth operator application reuses the same code as the acceptance scans.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch
from torch import nn

from infra.head_scan import head_importance, prune_heads
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, load_tokenizer, model_device


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
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--mode", required=True,
                    choices=["width", "depth", "head", "width_depth", "width_head"])
    ap.add_argument("--drop", type=int, default=1)
    ap.add_argument("--head-keep", type=float, default=0.75)
    ap.add_argument("--width", type=float, default=0.9)
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--calib-tokens", type=int, default=20000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda:0"
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(model.config, "vocab_size", 200000))
    high = min(200000, vocab - 2)
    prefix = torch.randint(100, high, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, high, (1, 1), dtype=torch.long, device=dev)
    verify = torch.randint(100, high, (1, a.gamma + 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        pt = model(input_ids=prefix, use_cache=True).past_key_values

    def target_one():
        c = copy.deepcopy(pt)
        model(input_ids=one, past_key_values=c, use_cache=True)

    def target_verify():
        c = copy.deepcopy(pt)
        model(input_ids=verify, past_key_values=c, use_cache=True)

    t1 = statistics.median(timed(target_one, a.repeats))
    tv = statistics.median(timed(target_verify, a.repeats))
    tau = tv / t1
    print(f"target one {t1:.4f}s verify {tv:.4f}s tau {tau:.3f}", flush=True)

    # apply the operator after the target measurements
    if a.mode in ("head", "width_head"):
        tok = load_tokenizer(a.model)
        rows = [json.loads(line) for line in open(a.calib)]
        scores = head_importance(model, tok, rows, a.calib_tokens, 1024, dev)
        prune_heads(model, a.head_keep, scores)
    if a.mode in ("depth", "width_depth"):
        layers = list(model.model.layers)
        model.model.layers = nn.ModuleList(layers[: len(layers) - a.drop])
        torch.cuda.empty_cache()
    if a.mode in ("width", "width_depth", "width_head"):
        imp = torch.load(a.importance, map_location="cpu", weights_only=True)
        for i, layer in enumerate(model.model.layers):
            layer.mlp, _ = make_narrow(layer, a.width, "width",
                                       next(layer.parameters()).device,
                                       imp.get(f"l{i:02d}") if imp else None)

    with torch.no_grad():
        pd = model(input_ids=prefix, use_cache=True).past_key_values

    def draft_one():
        c = copy.deepcopy(pd)
        model(input_ids=one, past_key_values=c, use_cache=True)

    for _ in range(5):
        draft_one()
    d1 = statistics.median(timed(draft_one, a.repeats))
    rho = d1 / t1
    cost = a.gamma * rho + tau
    out = {"model": a.model, "mode": a.mode, "drop": a.drop,
           "head_keep": a.head_keep, "width": a.width,
           "prefix": a.prefix, "gamma": a.gamma,
           "target_one_s": t1, "target_verify_s": tv,
           "draft_one_s": d1, "rho": rho, "tau": tau, "cost": cost,
           "cost_only_no_go": bool(cost > a.gamma + 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
