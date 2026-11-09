"""Activation-aware importance scores for structured FFN width pruning.

For each intermediate channel ``j`` of layer ``l`` we score

    s_j = mean_t |a_tj| * ||W_down[:, j]||_2 ,

where ``a = act(gate(x)) * up(x)`` is the post-activation intermediate.  This
is the first-order contribution of channel ``j`` to the block output, and is a
stronger structured-pruning criterion than raw weight norm (Wanda-style).
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from infra import data
from infra.model_loader import load_text_model, load_tokenizer, model_device


@torch.inference_mode()
def collect(model, tok, rows, max_tokens, max_len, device):
    layers = model.model.layers
    sums = [None] * len(layers)
    counts = [0] * len(layers)
    handles = []

    def make_hook(i):
        def hook(mod, inp, out):
            # Qwen3_5MLP.forward: out = down(act(gate(x)) * up(x)); recompute a.
            x = inp[0]
            a = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            v = a.abs().float().sum(dim=(0, 1)).cpu()
            sums[i] = v if sums[i] is None else sums[i] + v
            counts[i] += a.shape[0] * a.shape[1]
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(i)))
    n = 0
    for r in rows:
        text = r.get("text") or r.get("prompt") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 16:
            continue
        model(input_ids=ids, use_cache=False)
        n += ids.shape[1]
        if n >= max_tokens:
            break
    for h in handles:
        h.remove()
    return [s / max(c, 1) for s, c in zip(sums, counts)], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--data", nargs="+",
                    default=["data/processed/distill_controlled.jsonl"])
    ap.add_argument("--max-tokens", type=int, default=200000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for path in a.data:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    tok = load_tokenizer(a.model)
    model = load_text_model(a.model, dtype="bfloat16", device_map=a.device_map)
    dev = model_device(model)
    acts, n = collect(model, tok, rows, a.max_tokens, a.max_len, dev)
    print(f"collected {n} tokens", flush=True)
    out = {}
    for i, layer in enumerate(model.model.layers):
        wd = layer.mlp.down_proj.weight.detach().float().cpu()
        importance = acts[i] * wd.pow(2).sum(0).sqrt()
        out[f"l{i:02d}"] = importance
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    print(f"saved {len(out)} layers -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
