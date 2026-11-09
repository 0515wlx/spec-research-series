"""Teacher-forced acceptance / divergence evaluation for one (draft, domain).

Uses a cached target top-k distribution (``infra.cache_target``) and runs only the draft.
Saves per-token arrays (.npz) and a summary (.json) with q, E[TV], q_h, tau(gamma),
survival S(k), clustering, and per-sequence q for bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from infra import data
from infra.cache_target import build_cache
from infra.metrics import clustering, metrics_topk, simulate_blocks, survival
from infra.model_loader import load_text_model, model_device


@torch.inference_mode()
def eval_batch(model, seqs, cache, idxs, device):
    fulls = [seqs[k]["input_ids"] + seqs[k]["output_ids"] for k in idxs]
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
    for i, k in enumerate(idxs):
        s = seqs[k]
        start = len(s["input_ids"]) - 1
        L = len(s["output_ids"])
        lg = logits[i, start:start + L]
        m = metrics_topk(
            lg,
            torch.from_numpy(cache["top_idx"][k, :L]).long().to(device),
            torch.from_numpy(cache["top_logp"][k, :L]).float().to(device),
            torch.from_numpy(cache["p0"][k, :L]).float().to(device),
            torch.from_numpy(cache["argmax"][k, :L]).long().to(device),
            torch.from_numpy(cache["target_ids"][k, :L]).long().to(device),
            torch.from_numpy(cache["margin"][k, :L]).float().to(device),
            torch.from_numpy(cache["ent"][k, :L]).float().to(device),
        )
        outs.append({kk: v.detach().cpu().numpy() for kk, v in m.items()})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gammas", default="1,2,4,8,16,32,64")
    ap.add_argument("--ks", default="1,2,3,5,8,12,16,24,32,48,64")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--recompute-cache", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cache_path = a.cache or os.path.join(
        "out", "target_cache", os.path.basename(a.data).replace(".jsonl", "") + f"_k{a.topk}.npz"
    )
    if os.path.exists(cache_path) and not a.recompute_cache:
        cache = dict(np.load(cache_path))
        print(f"loaded target cache {cache_path}", flush=True)
    else:
        print(f"building target cache -> {cache_path}", flush=True)
        cache = build_cache(a.target, seqs, a.topk, a.batch_size, a.device_map, a.dtype)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, **cache)
        print(f"saved target cache {cache_path}", flush=True)

    draft = load_text_model(a.draft, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(draft)
    print(f"draft={a.draft} device={dev} seqs={len(seqs)}", flush=True)

    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    all_out: list[dict | None] = [None] * len(seqs)
    t0 = time.time()
    for i in range(0, len(order), a.batch_size):
        idxs = order[i:i + a.batch_size]
        outs = eval_batch(draft, seqs, cache, idxs, dev)
        for k, o in zip(idxs, outs):
            all_out[k] = o
        if (i // a.batch_size) % 10 == 0:
            print(f"  eval {i + len(idxs)}/{len(seqs)}  {time.time()-t0:.0f}s", flush=True)
            torch.cuda.empty_cache()

    keys = ["match", "tv", "kl_td", "kl_dt", "js", "hell", "margin", "rho",
            "ent_t", "ent_d", "p0", "q0", "rank", "target_ids", "pred_d"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0) for k in keys}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)

    gammas = [int(x) for x in a.gammas.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    per_seq_q = [1.0 - float(o["match"].mean()) for o in all_out]

    summary = {
        "target": a.target,
        "draft": a.draft,
        "data": a.data,
        "cache": cache_path,
        "topk": a.topk,
        "seqs": len(seqs),
        "tokens": int(len(match)),
        "q": q,
        "match_rate": float(match.mean()),
        "E_TV": float(cat["tv"].mean()),
        "E_KL_td": float(cat["kl_td"].mean()),
        "E_KL_dt": float(cat["kl_dt"].mean()),
        "E_JS": float(cat["js"].mean()),
        "E_margin": float(cat["margin"].mean()),
        "E_rho": float(cat["rho"].mean()),
        "q_h": float((cat["rho"] >= 1).mean()),
        "frac_rho_lt1": float((cat["rho"] < 1).mean()),
        "tau_inf": (1.0 / q) if q > 0 else None,
        "survival": survival(match, ks),
        "per_gamma": simulate_blocks(match, gammas),
        "clustering": clustering(match),
        "q_per_seq": per_seq_q,
        "eval_seconds": time.time() - t0,
    }

    np.savez(a.out.replace(".json", ".npz"), **cat, gen_len=seq_len, seq_id=seq_id)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ["q", "E_TV", "q_h", "tau_inf", "tokens"]}, indent=1))
    print(f"saved {a.out} and {a.out.replace('.json', '.npz')}")


if __name__ == "__main__":
    main()
