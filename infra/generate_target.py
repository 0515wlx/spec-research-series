"""Generate target-model greedy continuations for each domain.

Output format (JSONL):
  {"id", "domain", "prompt", "input_ids", "output_ids", "text", "meta"}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from infra import data
from infra.model_loader import load_text_model, load_tokenizer, model_device


def _tokenize_prompt(tok, domain: str, prompt: str, thinking: bool, max_prompt_tokens: int):
    spec = data.get_domain(domain)
    if spec.chat:
        text = data.chat_prompt(tok, prompt, thinking=thinking)
    else:
        text = prompt
    ids = tok(text, add_special_tokens=False, return_tensors=None)["input_ids"]
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    return ids


@torch.inference_mode()
def generate_batch(model, tok, prompts, max_new: int, device, thinking: bool,
                   max_prompt_tokens: int, sample: bool = False, temperature: float = 1.0):
    ids = [_tokenize_prompt(tok, p["domain"], p["prompt"], thinking, max_prompt_tokens) for p in prompts]
    maxlen = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(ids), maxlen), pad, dtype=torch.long)
    att = torch.zeros((len(ids), maxlen), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
        att[i, maxlen - len(x):] = 1
    input_ids = input_ids.to(device)
    att = att.to(device)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=att,
        max_new_tokens=max_new,
        do_sample=sample,
        num_beams=1,
        use_cache=True,
        pad_token_id=pad,
        eos_token_id=tok.eos_token_id,
    )
    if sample:
        gen_kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    out = model.generate(**gen_kwargs)
    results = []
    for i, x in enumerate(ids):
        gen = out[i, maxlen:].tolist()
        # strip trailing pad; keep the first eos as a stop marker but drop it from output_ids
        while gen and gen[-1] == pad:
            gen.pop()
        if tok.eos_token_id in gen:
            gen = gen[:gen.index(tok.eos_token_id)]
        results.append({"input_ids": x, "output_ids": gen})
    return results


def wikitext_rows(tok, n_chunks: int, seq_len: int):
    """Token-level raw continuation chunks from WikiText-2 test."""
    text = data.load_text()
    ids_all = tok(text, add_special_tokens=False)["input_ids"]
    half = seq_len // 2
    rows = []
    for i in range(n_chunks):
        chunk = ids_all[i * seq_len:(i + 1) * seq_len]
        if len(chunk) < seq_len:
            break
        rows.append({
            "id": f"wikitext-{i}", "domain": "wikitext", "prompt": tok.decode(chunk[:half]),
            "input_ids": chunk[:half], "output_ids": chunk[half:],
            "text": tok.decode(chunk[half:]),
            "meta": {"target": "none", "mode": "raw_continuation", "seq_len": seq_len},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data", default=None, help="processed prompt JSONL; default data/processed/{domain}.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--seq-len", type=int, default=1024, help="total tokens for raw wikitext chunks")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--thinking", action="store_true", help="allow Qwen3.5 thinking mode")
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generate-wikitext", action="store_true",
                    help="generate target continuations from raw WikiText prompts (instead of raw continuation)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.domain == "wikitext" and not a.generate_wikitext:
        tok = load_tokenizer(a.target)
        rows = wikitext_rows(tok, n_chunks=a.limit or 100, seq_len=a.seq_len)
        data.write_jsonl(a.out, rows)
        print(f"wrote {len(rows)} wikitext rows -> {a.out}")
        return

    tok = load_tokenizer(a.target)
    if a.data:
        prompts = [json.loads(l) for l in open(a.data)]
        if a.limit:
            prompts = prompts[:a.limit]
    else:
        prompts = data.load_prompts(a.domain, limit=a.limit)

    model = load_text_model(a.target, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    print(f"target={a.target} device={dev} prompts={len(prompts)}", flush=True)

    t0 = time.time()
    n = 0
    with open(a.out, "w") as f:
        for i in range(0, len(prompts), a.batch_size):
            batch = prompts[i:i + a.batch_size]
            torch.manual_seed(a.seed + i)
            outs = generate_batch(model, tok, batch, a.max_new, dev, a.thinking,
                                  a.max_prompt_tokens, sample=a.sample,
                                  temperature=a.temperature)
            for p, o in zip(batch, outs):
                rec = {
                    "id": p.get("id", f"{a.domain}-{n}"),
                    "domain": a.domain,
                    "prompt": p["prompt"],
                    "input_ids": o["input_ids"],
                    "output_ids": o["output_ids"],
                    "text": tok.decode(o["output_ids"], skip_special_tokens=False),
                    "meta": {
                        "target": a.target,
                        "max_new": a.max_new,
                        "thinking": bool(a.thinking),
                        "source": p.get("meta", {}).get("source"),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            if (i // a.batch_size) % 10 == 0:
                print(f"  {n}/{len(prompts)}  {time.time()-t0:.0f}s", flush=True)
    print(f"wrote {n} rows -> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
