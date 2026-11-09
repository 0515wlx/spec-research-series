"""Exact re-prefill speculative loop (cost upper bound) for the hybrid model.

The negative-crop loop is fast but its linear-attention recurrent state can drift
after several blocks (an identity draft eventually rejects).  This script removes
the rollback entirely: after every block it re-prefills the committed prefix from
scratch for both models.  It is therefore *exact* but not a fair serving loop: the
target does more work than a production engine would do.  The measured speedup is
a lower bound on the speedup of a correct rollback implementation, i.e. a cost
upper bound.

Use ``serve_sd_loop_hybrid.py`` for the optimistic negative-crop number and this
script for the conservative exact number.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from infra import data
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, load_tokenizer, model_device


@torch.inference_mode()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1]


@torch.inference_mode()
def target_only(model, ids, max_new):
    t0 = time.perf_counter()
    past, logits = prefill(model, ids)
    n = 0
    for _ in range(max_new):
        nxt = logits.argmax(-1, keepdim=True)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
        n += 1
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "tokens": n}


@torch.inference_mode()
def speculative_reprefill(model_t, model_d, ids, gamma, max_new):
    t0 = time.perf_counter()
    _, logits = prefill(model_t, ids)
    bonus = int(logits.argmax(-1))
    emitted = [bonus]
    stats = {"proposed": 0, "accepted": 0, "blocks": 0}
    while len(emitted) < max_new:
        seq = torch.cat([ids, torch.tensor([emitted], dtype=torch.long,
                                            device=ids.device)], dim=1)
        # draft proposes gamma tokens after the current committed prefix
        outd = model_d(input_ids=seq, use_cache=True)
        dp = outd.past_key_values
        d_logits = outd.logits[:, -1]
        prop = []
        for _ in range(gamma):
            nxt = int(d_logits.argmax(-1))
            prop.append(nxt)
            outd = model_d(input_ids=torch.tensor([[nxt]], dtype=torch.long,
                                                   device=ids.device),
                           past_key_values=dp, use_cache=True)
            dp = outd.past_key_values
            d_logits = outd.logits[:, -1]
        # target verifies by re-prefilling the committed prefix + proposals
        ver = torch.cat([seq, torch.tensor([prop], dtype=torch.long,
                                           device=ids.device)], dim=1)
        out = model_t(input_ids=ver, use_cache=True)
        base = seq.shape[1] - 1
        pred = out.logits[0, base:base + gamma].argmax(-1).tolist()
        acc = 0
        for i in range(gamma):
            if pred[i] == prop[i]:
                acc += 1
            else:
                break
        new_bonus = int(out.logits[0, base + gamma].argmax(-1))
        stats["proposed"] += gamma
        stats["accepted"] += acc
        stats["blocks"] += 1
        emitted.extend(prop[:acc])
        if len(emitted) < max_new:
            emitted.append(new_bonus)
        bonus = new_bonus
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "tokens": len(emitted),
            "alpha": stats["accepted"] / max(stats["proposed"], 1),
            "blocks": stats["blocks"],
            "tau": len(emitted) / max(stats["blocks"], 1)}


def compress_draft(model, importance, width, device):
    for i, layer in enumerate(model.model.layers):
        layer.mlp, _ = make_narrow(layer, width, "width", device,
                                   importance.get(f"l{i:02d}") if importance else None,
                                   pad_to=128)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = load_tokenizer(a.target)
    seqs = data.load_target_sequences(a.data, limit=a.limit)
    prompts = []
    for s in seqs:
        ids = s["input_ids"][-a.prompt_tokens:]
        if len(ids) >= 32:
            prompts.append(torch.tensor(ids, dtype=torch.long, device=a.device)[None])
    print(f"prompts={len(prompts)} prefix={prompts[0].shape[1]}", flush=True)

    target = load_text_model(a.target, dtype="bfloat16", device_map={"": a.device})
    draft = load_text_model(a.target, dtype="bfloat16", device_map={"": a.device})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    draft = compress_draft(draft, imp, a.width, model_device(draft))
    print("models ready", flush=True)

    for i in range(min(a.warmup, len(prompts))):
        target_only(target, prompts[i], 8)
        speculative_reprefill(target, draft, prompts[i], a.gamma, 8)

    base = [target_only(target, p, a.max_new) for p in prompts]
    sd = [speculative_reprefill(target, draft, prompts[i], a.gamma, a.max_new)
          for i, p in enumerate(prompts)]
    base_tps = [r["tokens"] / r["seconds"] for r in base]
    sd_tps = [r["tokens"] / r["seconds"] for r in sd]
    out = {
        "target": a.target, "draft": f"compressed@{a.width}",
        "loop": "exact re-prefill (cost upper bound)",
        "gamma": a.gamma, "max_new": a.max_new, "n": len(prompts),
        "target_only_tok_s": statistics.median(base_tps),
        "speculative_tok_s": statistics.median(sd_tps),
        "speedup": statistics.median(sd_tps) / statistics.median(base_tps),
        "alpha": statistics.median([r["alpha"] for r in sd]),
        "tau": statistics.median([r["tau"] for r in sd]),
        "base": base, "sd": sd,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ["target_only_tok_s", "speculative_tok_s",
                                         "speedup", "alpha", "tau"]}, indent=1))


if __name__ == "__main__":
    main()
