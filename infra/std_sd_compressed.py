"""Standard speculative-sampling loop with a compressed target-as-draft.

Builds a 98%-width draft from the target by activation-aware pruning plus the
layer-wise ridge reconstruction used in the paper, then runs the standard SD
sampling rule of Leviathan et al. / Chen et al. on raw OOD prompts.  Reports
empirical acceptance length, E[TV], greedy rejection on the sampled prefixes,
and a wall-clock comparison against a target-only sampling loop.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.joint_scan import NarrowMLP, make_narrow
from infra.layerwise_reconstruct import collect_calibration, reconstruct_layer
from infra.model_loader import load_text_model, load_tokenizer, model_device
from infra.spec_sampling import sd_sample


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
def target_only(tok, prompt_ids, max_new, device, seed=0):
    torch.manual_seed(seed)
    cur = list(prompt_ids)
    emitted = []
    t0 = time.time()
    for _ in range(max_new):
        ids = torch.tensor(cur, dtype=torch.long, device=device)[None]
        logits = target_logits = None
        out = None
        out = _target(input_ids=ids, use_cache=False)
        p = F.softmax(out.logits[0, -1].float(), dim=-1)
        x = int(torch.multinomial(p, 1).item())
        cur.append(x)
        emitted.append(x)
        if x == tok.eos_token_id:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"tokens": len(emitted), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global _target
    tok = load_tokenizer(a.model)
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": "cuda:0"})
    draft = load_text_model(a.model, dtype="bfloat16", device_map={"": "cuda:0"})
    dev = model_device(target)
    rows = [json.loads(line) for line in open(a.calib)]
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    print("building compressed draft", flush=True)
    draft = build_draft(draft, imp, a.width, rows, tok, dev)
    print("draft ready", flush=True)

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    prompts = [s["input_ids"] for s in seqs]
    recs = []
    _target = target
    for i, ids in enumerate(prompts):
        r = sd_sample(draft, target, tok, ids, a.gamma, a.max_new, dev, dev,
                      {tok.eos_token_id}, seed=i)
        base = target_only(tok, ids, a.max_new, dev, seed=i)
        sd_tokens = r["tokens"]
        sd_sec = r["draft_seconds"] + r["target_seconds"]
        rec = {k: r[k] for k in ["alpha_empirical", "alpha_token", "E_TV_loop",
                                 "q_greedy_loop", "tau", "steps", "tokens",
                                 "accepted", "proposed", "draft_seconds",
                                 "target_seconds"]}
        rec["target_only_tokens"] = base["tokens"]
        rec["target_only_seconds"] = base["seconds"]
        rec["sd_tokens_per_s"] = sd_tokens / max(sd_sec, 1e-9)
        rec["target_tokens_per_s"] = base["tokens"] / max(base["seconds"], 1e-9)
        rec["speedup"] = rec["sd_tokens_per_s"] / max(rec["target_tokens_per_s"], 1e-12)
        recs.append(rec)
        if i % 5 == 0:
            print(f"  {i+1}/{len(prompts)} tau={r['tau']:.2f} "
                  f"speedup={rec['speedup']:.2f}", flush=True)

    def mean(k):
        v = [r[k] for r in recs if r[k] is not None]
        return float(np.mean(v)) if v else None

    out = {"model": a.model, "width": a.width, "data": a.data,
           "gamma": a.gamma, "max_new": a.max_new, "prompts": len(recs),
           "tau": mean("tau"), "alpha_empirical": mean("alpha_empirical"),
           "E_TV_loop": mean("E_TV_loop"), "q_greedy_loop": mean("q_greedy_loop"),
           "speedup": mean("speedup"),
           "sd_tokens_per_s": mean("sd_tokens_per_s"),
           "target_tokens_per_s": mean("target_tokens_per_s"),
           "per_prompt": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ["tau", "alpha_empirical", "E_TV_loop",
                                          "q_greedy_loop", "speedup"]}, indent=1))


if __name__ == "__main__":
    main()
