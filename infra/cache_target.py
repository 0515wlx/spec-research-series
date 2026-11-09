"""Cache target top-k log-probabilities for target-generated sequences.

The capacity scan evaluates many drafts against the same target prefixes.  Running the
35B target once and caching top-k log-probs (+ residual mass) avoids repeating it for
every draft, and keeps TV/KL accurate enough (include the residual bucket).

Output .npz:
  top_idx   [N, T, K] int32
  top_logp  [N, T, K] float16
  p0        [N, T]    float32   target residual mass outside top-k
  ent       [N, T]    float32
  margin    [N, T]    float32   p(top1) - p(top2)
  argmax    [N, T]    int32
  target_ids[N, T]    int32
  gen_len   [N]       int32
  seq_id    [N]       int32
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.model_loader import load_text_model, load_tokenizer, model_device


@torch.inference_mode()
def cache_batch(model, seqs, topk: int, device, logp_dtype="float16"):
    fulls = [s["input_ids"] + s["output_ids"] for s in seqs]
    xs = [f[:-1] for f in fulls]
    maxlen = max(len(x) for x in xs)
    x = torch.zeros((len(xs), maxlen), dtype=torch.long)
    att = torch.zeros((len(xs), maxlen), dtype=torch.long)
    for i, xx in enumerate(xs):
        x[i, :len(xx)] = torch.tensor(xx, dtype=torch.long)
        att[i, :len(xx)] = 1
    x, att = x.to(device), att.to(device)
    logits = model(input_ids=x, attention_mask=att, use_cache=False).logits.float()
    outs = []
    for i, s in enumerate(seqs):
        start = len(s["input_ids"]) - 1
        L = len(s["output_ids"])
        lg = logits[i, start:start + L]
        logp = F.log_softmax(lg, dim=-1)
        tk = logp.topk(topk, dim=-1)
        p = tk.values.exp()
        p0 = (1.0 - p.sum(-1)).clamp_min(1e-8)
        ent = -(p * tk.values).sum(-1)
        margin = p[:, 0] - p[:, 1]
        outs.append({
            "top_idx": tk.indices.cpu().numpy().astype(np.int32),
            "top_logp": tk.values.cpu().numpy().astype(logp_dtype),
            "p0": p0.cpu().numpy().astype(np.float32),
            "ent": ent.cpu().numpy().astype(np.float32),
            "margin": margin.cpu().numpy().astype(np.float32),
            "argmax": tk.indices[:, 0].cpu().numpy().astype(np.int32),
            "target_ids": np.asarray(s["output_ids"], dtype=np.int32),
        })
    return outs


def pack(all_out: list[dict], topk: int, logp_dtype="float16") -> dict:
    T = max(len(o["target_ids"]) for o in all_out)
    N, K = len(all_out), topk
    top_idx = np.zeros((N, T, K), dtype=np.int32)
    top_logp = np.zeros((N, T, K), dtype=logp_dtype)
    p0 = np.zeros((N, T), dtype=np.float32)
    ent = np.zeros((N, T), dtype=np.float32)
    margin = np.zeros((N, T), dtype=np.float32)
    argmax = np.zeros((N, T), dtype=np.int32)
    tgt = np.zeros((N, T), dtype=np.int32)
    gen_len = np.zeros(N, dtype=np.int32)
    for i, o in enumerate(all_out):
        L = len(o["target_ids"])
        gen_len[i] = L
        top_idx[i, :L] = o["top_idx"]
        top_logp[i, :L] = o["top_logp"]
        p0[i, :L] = o["p0"]
        ent[i, :L] = o["ent"]
        margin[i, :L] = o["margin"]
        argmax[i, :L] = o["argmax"]
        tgt[i, :L] = o["target_ids"]
    return dict(top_idx=top_idx, top_logp=top_logp, p0=p0, ent=ent, margin=margin,
                argmax=argmax, target_ids=tgt, gen_len=gen_len,
                seq_id=np.arange(N, dtype=np.int32))


def build_cache(target: str, seqs: list[dict], topk: int = 256, batch_size: int = 4,
                device_map="auto", dtype: str = "bfloat16", verbose: bool = True,
                logp_dtype: str = "float16") -> dict:
    model = load_text_model(target, dtype=dtype, device_map=device_map)
    dev = model_device(model)
    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    out: list[dict | None] = [None] * len(seqs)
    t0 = time.time()
    for i in range(0, len(order), batch_size):
        ids = order[i:i + batch_size]
        outs = cache_batch(model, [seqs[k] for k in ids], topk, dev, logp_dtype)
        for k, o in zip(ids, outs):
            out[k] = o
        if verbose and (i // batch_size) % 10 == 0:
            print(f"  cache {i + len(ids)}/{len(seqs)}  {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return pack(out, topk, logp_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--logp-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = build_cache(a.target, seqs, a.topk, a.batch_size, a.device_map, a.dtype,
                        logp_dtype=a.logp_dtype)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **cache)
    print(f"saved {a.out}  N={len(seqs)} T={cache['top_idx'].shape[1]} K={a.topk}")


if __name__ == "__main__":
    main()
