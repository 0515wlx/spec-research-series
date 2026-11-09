"""Batch-1 latency decomposition and gamma scan for a compressed draft.

This is an appendix-only measurement.  It uses the same padded narrow-MLP
implementation as ``latency_pair.py`` (no ridge readout) and the same
KV-cache timing protocol as ``verify_latency.py``.

For each width we report

* the absolute per-pass latency of the target and the compressed draft;
* the verification ratio ``tau(gamma) = T_T(gamma+1) / T_T(1)`` and the draft
  ratio ``rho(gamma) = gamma * T_D(1) / T_T(1)`` for several gamma;
* a marginal module decomposition obtained by replacing the MLP, the token
  mixer (full or linear attention), or the LM head with a zero/identity path.

The decomposition is a first-order ablation, not a profiler trace: marginal
costs need not add exactly to the full forward because launch overhead and
kernel fusion overlap.


NOTE (review 11): the module-ablation timings deep-copy the KV cache inside the
timed region (see the helpers ``target_one`` / ``full_draft``).  This biases
rho and tau toward one.  Use ``infra.latency_clean`` for new measurements.
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


def zero_forward_factory(kind):
    if kind == "mlp":
        return lambda x: torch.zeros_like(x)
    if kind == "full_attn":
        def _zero_full(*a, **k):
            h = k["hidden_states"] if "hidden_states" in k else a[0]
            return torch.zeros_like(h), None
        return _zero_full
    if kind == "linear_attn":
        def _zero_linear(*a, **k):
            h = k["hidden_states"] if "hidden_states" in k else a[0]
            return torch.zeros_like(h)
        return _zero_linear
    raise ValueError(kind)


def patch_zero(model, kind):
    saved = []
    for layer in model.model.layers:
        if kind == "mlp":
            mod = layer.mlp
        elif kind == "full_attn" and getattr(layer, "block_type", "") == "full_attention":
            mod = layer.self_attn
        elif kind == "linear_attn" and getattr(layer, "block_type", "") == "linear_attention":
            mod = layer.linear_attn
        else:
            continue
        saved.append((mod, mod.forward))
        mod.forward = zero_forward_factory(kind)
    return saved


def restore(saved):
    for mod, fn in saved:
        mod.forward = fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--widths", default="0.1,0.5,0.98")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda:0"
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    # Stay inside the target vocabulary: the hard-coded 200k upper bound is
    # valid for the Qwen families but out of range for Llama (128k).
    vocab = int(getattr(target.config, "vocab_size", 200000))
    prefix = torch.randint(100, vocab, (1, a.prefix), dtype=torch.long, device=dev)
    gammas = [int(x) for x in a.gammas.split(",")]

    with torch.no_grad():
        pt = target(input_ids=prefix, use_cache=True).past_key_values
        one = torch.randint(100, vocab, (1, 1), dtype=torch.long, device=dev)

    def target_one():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    def target_verify(g):
        ids = torch.randint(100, vocab, (1, g + 1), dtype=torch.long, device=dev)
        c = copy.deepcopy(pt)
        target(input_ids=ids, past_key_values=c, use_cache=True)

    t1 = timed(target_one, a.repeats)
    t1_med = statistics.median(t1)

    # Target-side gamma scan (target-independent of draft width).
    tau = {}
    for g in gammas:
        tv = timed(lambda g=g: target_verify(g), a.repeats)
        tau[str(g)] = {
            "verify_s": statistics.median(tv),
            "tau": statistics.median(tv) / t1_med,
            "times": tv,
        }

    # Module ablation on the target (full forward, one token, no cache copy).
    def full_once():
        c = copy.deepcopy(pt)
        target(input_ids=one, past_key_values=c, use_cache=True)

    def backbone_once():
        c = copy.deepcopy(pt)
        target.model(input_ids=one, past_key_values=c, use_cache=True)

    full_t = statistics.median(timed(full_once, a.repeats))
    back_t = statistics.median(timed(backbone_once, a.repeats))
    del target
    torch.cuda.empty_cache()

    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": dev})
        for i, layer in enumerate(draft.model.layers):
            layer.mlp, _ = make_narrow(layer, width, "width", dev,
                                       imp.get(f"l{i:02d}"), pad_to=128)
        with torch.no_grad():
            pd = draft(input_ids=prefix, use_cache=True).past_key_values
        warm = [draft(input_ids=one, past_key_values=copy.deepcopy(pd), use_cache=True)
                for _ in range(5)]
        torch.cuda.synchronize()

        d1 = statistics.median(timed(lambda: draft(input_ids=one,
                                                   past_key_values=copy.deepcopy(pd),
                                                   use_cache=True), a.repeats))
        gam = {}
        for g in gammas:
            def draft_gamma(g=g):
                c = copy.deepcopy(pd)
                for _ in range(g):
                    draft(input_ids=one, past_key_values=c, use_cache=True)
            dg = statistics.median(timed(draft_gamma, a.repeats))
            gam[str(g)] = {"draft_gamma_s": dg,
                           "rho_gamma": g * d1 / t1_med,
                           "denom": g * d1 / t1_med + tau[str(g)]["tau"]}

        def full_draft():
            c = copy.deepcopy(pd)
            draft(input_ids=one, past_key_values=c, use_cache=True)

        def back_draft():
            c = copy.deepcopy(pd)
            draft.model(input_ids=one, past_key_values=c, use_cache=True)

        dfull = statistics.median(timed(full_draft, a.repeats))
        dback = statistics.median(timed(back_draft, a.repeats))

        saved = patch_zero(draft, "mlp")
        dnomlp = statistics.median(timed(full_draft, a.repeats))
        restore(saved)
        saved = patch_zero(draft, "full_attn")
        dnofull = statistics.median(timed(full_draft, a.repeats))
        restore(saved)
        saved = patch_zero(draft, "linear_attn")
        dnolattn = statistics.median(timed(full_draft, a.repeats))
        restore(saved)

        rows.append({
            "width": width,
            "target_one_s": t1_med,
            "draft_one_s": d1,
            "rho_token": d1 / t1_med,
            "gamma": gam,
            "ablation": {
                "full_s": dfull,
                "backbone_s": dback,
                "lm_head_marginal_s": dfull - dback,
                "mlp_marginal_s": dfull - dnomlp,
                "full_attn_marginal_s": dfull - dnofull,
                "linear_attn_marginal_s": dfull - dnolattn,
                "no_mlp_s": dnomlp,
                "no_full_attn_s": dnofull,
                "no_linear_attn_s": dnolattn,
            },
        })
        print(json.dumps({"width": width, "rho_token": d1 / t1_med,
                          "ablation": rows[-1]["ablation"]}), flush=True)
        del draft
        torch.cuda.empty_cache()

    out = {"model": a.model, "prefix": a.prefix, "target_one_s": t1_med,
           "target_ablation": {"full_s": full_t, "backbone_s": back_t,
                               "lm_head_marginal_s": full_t - back_t},
           "target_gamma": tau, "drafts": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
