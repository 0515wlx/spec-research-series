"""Batch-1 verification-pass latency for a compressed draft.

Measures the exact cost components of the SD step with KV caches:
  * draft: gamma sequential one-token passes
  * target: one verification pass over gamma+1 tokens
Both use the same 512-token prefix cache; the prefix build is outside the timed
region.  Reports rho = draft_time / target_verification_time with a bootstrap CI.


NOTE (review 11): the timed closures below deep-copy the KV cache *inside* the
timed region.  That copy is not part of a serving pass and biases rho/tau
toward one.  Use ``infra.latency_clean`` for new measurements; the numbers in
the paper from this script are treated as protocol-biased/implementation-
conditional.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch

from infra.joint_scan import NarrowMLP, make_narrow
from infra.layerwise_reconstruct import collect_calibration, reconstruct_layer
from infra.model_loader import load_text_model, load_tokenizer, model_device


def build_draft(model, importance, width, calib_rows, tok, device, calib_tokens=16000):
    X, Y = collect_calibration(model, tok, calib_rows, calib_tokens, 1024, device)
    for i, layer in enumerate(model.model.layers):
        narrow, _ = make_narrow(layer, width, "width", device,
                                importance.get(f"l{i:02d}"), pad_to=0)
        W, _, _, _ = reconstruct_layer(narrow, X[i], Y[i])
        layer.mlp = NarrowMLP(narrow.config, narrow.hidden_size, narrow.intermediate_size,
                              narrow.gate_w, narrow.up_w, W, narrow.act_fn).to(device)
        del narrow
        torch.cuda.empty_cache()
    return model


@torch.inference_mode()
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
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = load_tokenizer(a.model)
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": "cuda:0"})
    draft = load_text_model(a.model, dtype="bfloat16", device_map={"": "cuda:0"})
    dev = model_device(target)
    rows = [json.loads(line) for line in open(a.calib)]
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    print("building draft", flush=True)
    draft = build_draft(draft, imp, a.width, rows, tok, dev)
    print("ready", flush=True)

    prefix = torch.randint(100, 200000, (1, a.prefix), dtype=torch.long, device=dev)
    with torch.inference_mode():
        pt = target(input_ids=prefix, use_cache=True).past_key_values
        pd = draft(input_ids=prefix, use_cache=True).past_key_values
        verify = torch.randint(100, 200000, (1, a.gamma + 1), dtype=torch.long, device=dev)
        one = torch.randint(100, 200000, (1, 1), dtype=torch.long, device=dev)

    def target_verify():
        c = copy.deepcopy(pt)
        target(input_ids=verify, past_key_values=c, use_cache=True)

    def draft_gamma():
        c = copy.deepcopy(pd)
        for _ in range(a.gamma):
            draft(input_ids=one, past_key_values=c, use_cache=True)

    def target_one():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    def draft_one():
        c = copy.deepcopy(pd)
        draft(input_ids=one, past_key_values=c, use_cache=True)

    tt = timed(target_verify, a.repeats)
    dt = timed(draft_gamma, a.repeats)
    t1 = timed(target_one, a.repeats)
    d1 = timed(draft_one, a.repeats)
    rho = statistics.median(dt) / statistics.median(tt)
    # bootstrap the ratio over repeats
    import numpy as np
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(tt), len(tt))
        boots.append(np.median(np.array(dt)[i]) / np.median(np.array(tt)[i]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out = {"model": a.model, "width": a.width, "prefix": a.prefix,
           "gamma": a.gamma, "target_verify_s": statistics.median(tt),
           "draft_gamma_s": statistics.median(dt),
           "target_one_s": statistics.median(t1),
           "draft_one_s": statistics.median(d1),
           "rho": rho, "rho_ci": ci,
           "rho_token": statistics.median(d1) / statistics.median(t1),
           "target_times": tt, "draft_times": dt,
           "target_one_times": t1, "draft_one_times": d1}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ["rho", "rho_ci", "target_verify_s",
                                          "draft_gamma_s"]}, indent=1))


if __name__ == "__main__":
    main()
