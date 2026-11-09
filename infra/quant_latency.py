"""Measured batch-1 cost of a real weight-only INT4/INT8 draft.

The target is the same model in bf16.  We measure one target token pass, the
gamma+1 verification pass, and one quantized-draft token pass with KV caches,
then report rho_1, tau_1 and the cost gamma*rho_1+tau_1.  Both models live on
one GPU; the target is measured first and the quantized copy second.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.cost_frontier import accepted
from infra.model_loader import load_text_model, model_device
from infra.quant_kernel import apply_int4, apply_int8, quantized_bytes


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


def q_star_measured(rho: float, tau: float, gamma: int) -> float:
    target = gamma * rho + tau
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if accepted(1.0 - mid, gamma) >= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--bits", type=int, choices=[4, 8], required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--inner-k-tiles", type=int, default=8)
    ap.add_argument("--skip-lm-head", action="store_true")
    ap.add_argument("--quantize-embedding", action="store_true")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = a.device
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    vocab = int(getattr(target.config, "vocab_size", 200000))
    high = min(200000, vocab - 2)
    prefix = torch.randint(100, high, (1, a.prefix), dtype=torch.long, device=dev)
    one = torch.randint(100, high, (1, 1), dtype=torch.long, device=dev)
    verify = torch.randint(100, high, (1, a.gamma + 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        pt = target(input_ids=prefix, use_cache=True).past_key_values

    def target_one():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    def target_verify():
        c = copy.deepcopy(pt)
        target(input_ids=verify, past_key_values=c, use_cache=True)

    for _ in range(3):
        target_one()
    t1 = statistics.median(timed(target_one, a.repeats))
    tv = statistics.median(timed(target_verify, a.repeats))
    tau = tv / t1
    print(f"target one {t1:.4f}s verify {tv:.4f}s tau {tau:.3f}", flush=True)

    draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    base_bytes = sum(p.numel() for p in draft.parameters()) * 2
    if a.bits == 4:
        names = apply_int4(draft, group_size=a.group,
                           inner_k_tiles=a.inner_k_tiles,
                           skip_lm_head=a.skip_lm_head,
                           skip_embedding=not a.quantize_embedding)
    else:
        names = apply_int8(draft, skip_lm_head=a.skip_lm_head,
                           skip_embedding=not a.quantize_embedding)
    dev = model_device(draft)
    bytes_info = quantized_bytes(draft)
    with torch.no_grad():
        pd = draft(input_ids=prefix, use_cache=True).past_key_values

    def draft_one():
        c = copy.deepcopy(pd)
        draft(input_ids=one, past_key_values=c, use_cache=True)

    for _ in range(5):
        draft_one()
    d1 = statistics.median(timed(draft_one, a.repeats))
    rho = d1 / t1
    cost = a.gamma * rho + tau
    out = {
        "model": a.model, "bits": a.bits, "group": a.group,
        "skip_lm_head": a.skip_lm_head,
        "quantize_embedding": a.quantize_embedding,
        "quantized_layers": len(names),
        "prefix": a.prefix, "gamma": a.gamma,
        "target_one_s": t1, "target_verify_s": tv, "draft_one_s": d1,
        "rho_1": rho, "tau_1": tau, "cost": cost,
        "cost_only_no_go": bool(cost > a.gamma + 1),
        "target_bf16_bytes": int(base_bytes),
        "quantized_bytes": bytes_info,
        "rho_mem_weights": bytes_info["total_bytes"] / base_bytes,
        "q_star_measured": q_star_measured(rho, tau, a.gamma),
        "q_star_mem_weights": q_star_measured(bytes_info["total_bytes"] / base_bytes,
                                              1.0, a.gamma),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ["rho_1", "tau_1", "cost", "rho_mem_weights",
                       "q_star_measured", "q_star_mem_weights"]}, indent=1))


if __name__ == "__main__":
    main()
