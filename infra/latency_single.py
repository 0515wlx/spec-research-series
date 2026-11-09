"""Single-resident batch-1 decode-cost measurement (rho_1, tau_1).

Loads one model at a time so that large targets (Qwen3.5-27B) fit on an
80GB A100.  Run ``--mode target`` once and ``--mode draft`` once per width, in
separate processes; each process exits after its measurement, so a large
model is never held alive across loads.

With a 512-token prefix already in the KV cache we time

  * one target token pass                 -> T_T(1)
  * a gamma+1 target verification pass    -> T_T(gamma+1)
  * one compressed-draft token pass       -> T_D(1)
  * gamma sequential draft token passes   -> T_D(gamma)

and report rho_1 = T_D(1)/T_T(1), tau_1 = T_T(gamma+1)/T_T(1), and the
cost-model quantities gamma*rho_1 + tau_1 and S_max = (gamma+1)/(gamma*rho_1+tau_1).
The compressed draft pads the narrow MLP to a multiple of ``--pad-to`` and uses
ordinary dense GEMMs, i.e. the measured upper bound on rho_1.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics

import torch

from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device


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


def _inputs(prefix_len, gamma, device):
    prefix = torch.randint(100, 200000, (1, prefix_len), dtype=torch.long, device=device)
    one = torch.randint(100, 200000, (1, 1), dtype=torch.long, device=device)
    verify = torch.randint(100, 200000, (1, gamma + 1), dtype=torch.long, device=device)
    return prefix, one, verify


def measure_target(a, device):
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": device})
    prefix, one, verify = _inputs(a.prefix, a.gamma, device)
    with torch.inference_mode():
        past = model(input_ids=prefix, use_cache=True).past_key_values

    def target_one():
        c = copy.deepcopy(past)
        model(input_ids=one, past_key_values=c, use_cache=True)

    def target_verify():
        c = copy.deepcopy(past)
        model(input_ids=verify, past_key_values=c, use_cache=True)

    for _ in range(3):
        target_one()
        target_verify()
    t1 = statistics.median(timed(target_one, a.repeats))
    tv = statistics.median(timed(target_verify, a.repeats))
    out = {"model": a.model, "prefix": a.prefix, "gamma": a.gamma,
           "repeats": a.repeats, "target_one_s": t1,
           "target_verify_s": tv, "tau_1": tv / t1}
    return out


def measure_draft(a, device, tau_1):
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": device})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    for i, layer in enumerate(model.model.layers):
        layer.mlp, _ = make_narrow(layer, a.width, "width", device,
                                   imp.get(f"l{i:02d}") if imp else None,
                                   pad_to=a.pad_to)
    prefix, one, _ = _inputs(a.prefix, a.gamma, device)
    with torch.inference_mode():
        past = model(input_ids=prefix, use_cache=True).past_key_values

    def draft_one():
        c = copy.deepcopy(past)
        model(input_ids=one, past_key_values=c, use_cache=True)

    def draft_gamma():
        c = copy.deepcopy(past)
        for _ in range(a.gamma):
            model(input_ids=one, past_key_values=c, use_cache=True)

    for _ in range(3):
        draft_one()
        draft_gamma()
    d1 = statistics.median(timed(draft_one, a.repeats))
    dg = statistics.median(timed(draft_gamma, a.repeats))
    return {"model": a.model, "width": a.width, "pad_to": a.pad_to,
            "prefix": a.prefix, "gamma": a.gamma,
            "draft_one_s": d1, "draft_gamma_s": dg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default=None)
    ap.add_argument("--mode", default="both", choices=["target", "draft", "both"])
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pad-to", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-json", default=None,
                    help="target measurement JSON for --mode draft")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.mode == "target":
        out = measure_target(a, a.device)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1))
        return

    if a.target_json is None:
        raise SystemExit("--target-json is required for --mode draft")
    tgt = json.load(open(a.target_json))
    if a.width is None:
        raise SystemExit("--width is required for --mode draft")
    m = measure_draft(a, a.device, tgt["tau_1"])
    rho = m["draft_one_s"] / tgt["target_one_s"]
    tau = tgt["tau_1"]
    cost = a.gamma * rho + tau
    row = {"kind": "draft", "rho_1": rho, "tau_1": tau,
           "gamma_rho_plus_tau": cost,
           "S_max": (a.gamma + 1) / cost if cost > 0 else None, **m}
    with open(a.out, "w") as f:
        json.dump(row, f, indent=1)
    print(json.dumps({k: row[k] for k in
                      ["width", "rho_1", "tau_1", "gamma_rho_plus_tau", "S_max"]},
                     indent=1))


if __name__ == "__main__":
    main()
