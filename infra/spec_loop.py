"""Real greedy speculative-decoding loop (HF reference engine).

Measures per-block accepted length, per-depth acceptance, and draft/target timings.
This is intentionally simple (target re-runs the full prefix each block) and is meant to
validate teacher-forced estimates, not to be a throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from infra import data
from infra.model_loader import load_text_model, load_tokenizer, model_device


def _prompt_ids(tok, domain: str, prompt: str, thinking: bool, max_prompt_tokens: int):
    spec = data.get_domain(domain)
    text = data.chat_prompt(tok, prompt, thinking=thinking) if spec.chat else prompt
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-max_prompt_tokens:]


@torch.inference_mode()
def sd_generate(draft, target, tok, prompt_ids, gamma: int, max_new: int,
                ddev, tdev, eos_ids):
    cur = list(prompt_ids)
    emitted = []
    blocks = []
    n_acc = n_prop = 0
    draft_s = target_s = 0.0
    while len(emitted) < max_new:
        ids = torch.tensor(cur, dtype=torch.long, device=ddev)[None]
        t0 = time.time()
        o = draft(input_ids=ids, use_cache=True)
        past = o.past_key_values
        prop = []
        nxt = o.logits[0, -1].argmax()
        prop.append(int(nxt))
        for _ in range(gamma - 1):
            o = draft(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[0, -1].argmax()
            prop.append(int(nxt))
        if ddev.type == "cuda":
            torch.cuda.synchronize(ddev)
        draft_s += time.time() - t0

        t1 = time.time()
        pv = torch.tensor(prop, dtype=torch.long, device=tdev)[None]
        xt = torch.tensor(cur, dtype=torch.long, device=tdev)[None]
        o2 = target(input_ids=torch.cat([xt, pv], dim=1), use_cache=False)
        # logits at position len(cur)-1+i predict token len(cur)+i
        ver = o2.logits[0, len(cur) - 1: len(cur) - 1 + gamma].argmax(-1).tolist()
        if tdev.type == "cuda":
            torch.cuda.synchronize(tdev)
        target_s += time.time() - t1

        acc = 0
        for i in range(gamma):
            if prop[i] == ver[i]:
                acc += 1
            else:
                break
        n_acc += acc
        n_prop += gamma
        new = prop[:acc]
        if acc < gamma:
            new.append(ver[acc])
        else:
            # bonus token: the target's next token after the full accepted block
            new.append(int(o2.logits[0, -1].argmax()))
        cur.extend(new)
        emitted.extend(new)
        blocks.append({"accepted": acc, "proposed": gamma, "depth": len(blocks)})
        if new and new[-1] in eos_ids:
            break
    return {
        "alpha": n_acc / max(n_prop, 1),
        "tau": len(emitted) / max(len(blocks), 1),
        "steps": len(blocks),
        "tokens": len(emitted),
        "draft_seconds": draft_s,
        "target_seconds": target_s,
        "blocks": blocks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--device-map-draft", default="cuda:0")
    ap.add_argument("--device-map-target", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.data:
        seqs = data.load_target_sequences(a.data, limit=a.limit)
        prompts = [{"prompt": s.get("prompt", ""), "input_ids": s["input_ids"]} for s in seqs]
    else:
        prompts = data.load_prompts(a.domain, limit=a.limit)

    tok = load_tokenizer(a.target)
    draft = load_text_model(a.draft, dtype=a.dtype, device_map=a.device_map_draft)
    target = load_text_model(a.target, dtype=a.dtype, device_map=a.device_map_target)
    ddev, tdev = model_device(draft), model_device(target)
    eos_ids = {tok.eos_token_id}
    print(f"draft={a.draft}@{ddev} target={a.target}@{tdev}", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        if "input_ids" in p:
            ids = p["input_ids"]
        else:
            ids = _prompt_ids(tok, a.domain, p["prompt"], a.thinking, a.max_prompt_tokens)
        r = sd_generate(draft, target, tok, ids, a.gamma, a.max_new, ddev, tdev, eos_ids)
        r["id"] = p.get("id", i)
        rows.append(r)
        print(f"  {i+1}/{len(prompts)} tau={r['tau']:.2f} alpha={r['alpha']:.3f} "
              f"steps={r['steps']} {time.time()-t0:.0f}s", flush=True)

    summary = {
        "target": a.target, "draft": a.draft, "domain": a.domain, "gamma": a.gamma,
        "max_new": a.max_new, "n": len(rows),
        "alpha": sum(r["alpha"] for r in rows) / max(len(rows), 1),
        "tau": sum(r["tau"] for r in rows) / max(len(rows), 1),
        "draft_seconds": sum(r["draft_seconds"] for r in rows),
        "target_seconds": sum(r["target_seconds"] for r in rows),
        "rows": rows,
    }
    # per-depth acceptance: each block has gamma proposal slots; depth d is accepted
    # iff the block's accepted prefix is longer than d.
    depth = {}
    for r in rows:
        for b in r["blocks"]:
            for d in range(a.gamma):
                depth.setdefault(d, [0, 0])
                depth[d][0] += int(b["accepted"] > d)
                depth[d][1] += 1
    summary["per_depth_accept"] = {str(k): v[0] / max(v[1], 1) for k, v in sorted(depth.items())}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ["alpha", "tau", "n", "draft_seconds", "target_seconds"]}, indent=1))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
