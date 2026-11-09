"""End-to-end greedy SD loop for hybrid linear-attention caches.

This is ``serve_sd_loop.py`` with a cache-rollback shim for Qwen3.5's
``DynamicCache``.  Linear-attention layers do not expose positive ``crop(n)``;
after ``activate_past_recording()`` they accept a negative integer that removes
the last ``n`` positions.  The loop is otherwise identical: target verification
with exact rollback, draft rollback, same prompts and token budget for the
target-only and speculative runs.
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


def crop_last(cache, n: int):
    if n <= 0:
        return
    try:
        cache.crop(n)
    except RuntimeError:
        # Hybrid/linear-attention cache: record recurrent states, then pass a
        # negative count (number of trailing positions to remove).
        if hasattr(cache, "activate_past_recording"):
            cache.activate_past_recording()
        cache.crop(-n)


@torch.inference_mode()
def prefill(model, ids, record=False):
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    if record and hasattr(cache, "activate_past_recording"):
        cache.activate_past_recording()
    return cache, out.logits[:, -1]


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
def speculative(model_t, model_d, ids, gamma, max_new):
    t0 = time.perf_counter()
    tp, logits = prefill(model_t, ids, record=True)
    bonus = int(logits.argmax(-1))
    dp, _ = prefill(model_d, ids, record=True)
    emitted = [bonus]
    stats = {"proposed": 0, "accepted": 0, "blocks": 0}
    while len(emitted) < max_new:
        prop = []
        x = torch.tensor([[bonus]], dtype=torch.long, device=ids.device)
        out = model_d(input_ids=x, past_key_values=dp, use_cache=True)
        dp = out.past_key_values
        d_logits = out.logits[:, -1]
        for _ in range(gamma):
            nxt = int(d_logits.argmax(-1))
            prop.append(nxt)
            out = model_d(input_ids=torch.tensor([[nxt]], dtype=torch.long, device=ids.device),
                          past_key_values=dp, use_cache=True)
            dp = out.past_key_values
            d_logits = out.logits[:, -1]
        ver_ids = torch.tensor([[bonus] + prop], dtype=torch.long, device=ids.device)
        out = model_t(input_ids=ver_ids, past_key_values=tp, use_cache=True)
        tp = out.past_key_values
        pred = out.logits[0, :gamma].argmax(-1).tolist()
        acc = 0
        for i in range(gamma):
            if pred[i] == prop[i]:
                acc += 1
            else:
                break
        new_bonus = int(out.logits[0, gamma].argmax(-1))
        stats["proposed"] += gamma
        stats["accepted"] += acc
        stats["blocks"] += 1
        emitted.extend(prop[:acc])
        if len(emitted) < max_new:
            emitted.append(new_bonus)
        # target cache: drop rejected proposals
        crop_last(tp, gamma - acc)
        # draft cache: drop rejected proposals; if all accepted, append last
        if acc <= gamma - 1:
            crop_last(dp, gamma - 1 - acc)
        else:
            outd = model_d(input_ids=torch.tensor([[prop[-1]]], dtype=torch.long,
                                                   device=ids.device),
                           past_key_values=dp, use_cache=True)
            dp = outd.past_key_values
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
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
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
        speculative(target, draft, prompts[i], a.gamma, 8)

    base = [target_only(target, p, a.max_new) for p in prompts]
    sd = [speculative(target, draft, prompts[i], a.gamma, a.max_new)
          for i, p in enumerate(prompts)]
    base_tps = [r["tokens"] / r["seconds"] for r in base]
    sd_tps = [r["tokens"] / r["seconds"] for r in sd]
    out = {
        "target": a.target, "draft": f"compressed@{a.width}",
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
