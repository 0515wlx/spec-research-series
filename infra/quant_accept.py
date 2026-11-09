"""Acceptance evaluation of a weight-only fake-quantized draft (E5 control).

Example:
  python -m infra.quant_accept --draft qwen3.5-9b --bits 4 --data data/processed/math500_target.jsonl \
    --cache out/target_cache/math500_k256.npz --out out/accept/quant_math500_9b_w4.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from infra import data
from infra.accept_eval import eval_batch
from infra.metrics import clustering, simulate_blocks, survival
from infra.model_loader import load_text_model, model_device
from infra.quant import apply_fake_quant, quant_stats, refresh_fake_quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--skip-lm-head", action="store_true")
    ap.add_argument("--lm-head-bits", type=int, default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gammas", default="1,2,4,8,16,32,64")
    ap.add_argument("--ks", default="1,2,3,5,8,12,16,24,32,48,64")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = dict(np.load(a.cache))
    model = load_text_model(a.draft, dtype=a.dtype, device_map=a.device_map)
    info = {"draft": a.draft, "bits": a.bits, "group": a.group,
            "skip_lm_head": a.skip_lm_head, "lm_head_bits": a.lm_head_bits}
    if a.bits < 16:
        names = apply_fake_quant(model, bits=a.bits, group=a.group,
                                 skip_lm_head=a.skip_lm_head, lm_head_bits=a.lm_head_bits)
        info["quantized_linears"] = len(names)
        info["sizes"] = quant_stats(model)
        refresh_fake_quant(model)
    dev = model_device(model)
    print(f"draft={a.draft} bits={a.bits} device={dev} seqs={len(seqs)}", flush=True)

    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    all_out: list[dict | None] = [None] * len(seqs)
    t0 = time.time()
    for i in range(0, len(order), a.batch_size):
        idxs = order[i:i + a.batch_size]
        outs = eval_batch(model, seqs, cache, idxs, dev)
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
    summary = {
        "target": "cached", "draft": a.draft, "quant": info,
        "data": a.data, "cache": a.cache, "topk": int(cache["top_idx"].shape[-1]),
        "seqs": len(seqs), "tokens": int(len(match)), "q": q,
        "match_rate": float(match.mean()),
        "E_TV": float(cat["tv"].mean()), "E_KL_td": float(cat["kl_td"].mean()),
        "E_KL_dt": float(cat["kl_dt"].mean()), "E_JS": float(cat["js"].mean()),
        "E_margin": float(cat["margin"].mean()), "E_rho": float(cat["rho"].mean()),
        "q_h": float((cat["rho"] >= 1).mean()),
        "frac_rho_lt1": float((cat["rho"] < 1).mean()),
        "tau_inf": (1.0 / q) if q > 0 else None,
        "survival": survival(match, ks),
        "per_gamma": simulate_blocks(match, gammas),
        "clustering": clustering(match),
        "q_per_seq": [1.0 - float(o["match"].mean()) for o in all_out],
        "eval_seconds": time.time() - t0,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out.replace(".json", ".npz"), **cat, gen_len=seq_len, seq_id=seq_id)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ["q", "E_TV", "q_h", "tau_inf", "tokens"]}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
