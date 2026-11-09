"""Fair end-to-end greedy speculative-decoding throughput loop with KV caches.

The existing ``spec_loop.py`` is a reference acceptance diagnostic: it re-runs
the full target prefix every block, so its wall-clock is not a serving
measurement.  This script implements the standard KV-cache rollback:

  * ``target_past`` is the cache for the committed prefix *excluding* the last
    emitted bonus token;
  * a block feeds ``[bonus] + proposals`` to the target and verifies the
    proposals against the shifted logits;
  * the target cache is cropped by the number of rejected proposals, and the
    draft cache is cropped/updated to the same committed prefix.

We benchmark target-only decoding and speculative decoding on the same prompts,
batch size 1, same GPU, and the same number of generated tokens.  The loop is
used as a serving sanity check; the paper's cost claim remains the roofline
frontier.  Standard dense Llama attention is used because its cache supports
exact rollback; Qwen3.5's hybrid linear-attention cache does not.
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
    out_ids = []
    for _ in range(max_new):
        nxt = logits.argmax(-1, keepdim=True)
        out_ids.append(int(nxt))
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "tokens": len(out_ids)}


@torch.inference_mode()
def speculative(model_t, model_d, ids, gamma, max_new):
    t0 = time.perf_counter()
    # The first bonus token comes from a normal target prefill.
    tp, logits = prefill(model_t, ids)
    bonus = int(logits.argmax(-1))
    dp, _ = prefill(model_d, ids)
    emitted = [bonus]
    stats = {"proposed": 0, "accepted": 0, "blocks": 0}
    while len(emitted) < max_new:
        # draft proposes gamma tokens starting after the current bonus
        prop = []
        d_logits = None
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
        # verification: [bonus] + proposals
        ver_ids = torch.tensor([[bonus] + prop], dtype=torch.long, device=ids.device)
        out = model_t(input_ids=ver_ids, past_key_values=tp, use_cache=True)
        tp = out.past_key_values
        # logits at position i predict proposal i (positions 0..gamma-1)
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
        # target cache currently covers committed_prefix + bonus + all proposals.
        # Keep committed + accepted proposals; drop rejected proposals.
        tp.crop(gamma - acc)
        # draft cache covers committed + bonus + proposals[:-1].
        # Keep committed + accepted proposals; append p_gamma if fully accepted.
        if acc <= gamma - 1:
            dp.crop(gamma - 1 - acc)
        else:
            # dp currently ends at p_{gamma-1}; append the last proposal
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
                                   importance.get(f"l{i:02d}") if importance else None)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/path/to/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default=None,
                    help="pretrained draft path; omit to use a compressed target")
    ap.add_argument("--importance", default=None)
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = load_tokenizer(a.target)
    seqs = data.load_target_sequences(a.data, limit=a.limit)
    prompts = []
    for s in seqs:
        ids = s["input_ids"][-a.prompt_tokens:]
        if len(ids) >= 32:
            prompts.append(torch.tensor(ids, dtype=torch.long, device="cuda:0")[None])
    print(f"prompts={len(prompts)} prefix={prompts[0].shape[1]}", flush=True)

    target = load_text_model(a.target, dtype="bfloat16", device_map={"": "cuda:0"})
    if a.draft:
        draft = load_text_model(a.draft, dtype="bfloat16", device_map={"": "cuda:0"})
    else:
        draft = load_text_model(a.target, dtype="bfloat16", device_map={"": "cuda:0"})
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
        "target": a.target, "draft": a.draft or f"compressed@{a.width}",
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
