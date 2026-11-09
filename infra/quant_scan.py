"""Acceptance/certificate scan for a real weight-only INT4/INT8 draft.

This is the quantization-axis complement to `infra.quant_accept` (which uses
fake quantization and therefore cannot support a cost claim).  The target cache
is built from the *same* model in bf16, so `q` here is the quantization-induced
top-1 flip rate, not a capacity gap.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m infra.quant_scan --model qwen3.5-9b \
    --bits 4 --data data/processed/wikitext_target.jsonl \
    --cache out/target_cache/wikitext_target9b_k256.npz \
    --out out/quant/wikitext_w4.json
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
from infra.model_loader import load_text_model, model_device
from infra.quant_kernel import (apply_int4, apply_int8, quantized_bytes)


def bootstrap_lb(q_lb, seq_id, n_seq, B=2000, seed=0):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, q_lb.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--bits", type=int, choices=[4, 8], required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--inner-k-tiles", type=int, default=8)
    ap.add_argument("--skip-lm-head", action="store_true",
                    help="keep lm_head in bf16 (standard weight-only setting)")
    ap.add_argument("--quantize-embedding", action="store_true")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = dict(np.load(a.cache))
    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    base_bytes = sum(p.numel() for p in model.parameters()) * 2
    if a.bits == 4:
        names = apply_int4(model, group_size=a.group,
                           inner_k_tiles=a.inner_k_tiles,
                           skip_lm_head=a.skip_lm_head,
                           skip_embedding=not a.quantize_embedding)
    else:
        names = apply_int8(model, skip_lm_head=a.skip_lm_head,
                           skip_embedding=not a.quantize_embedding)
    dev = model_device(model)
    bytes_info = quantized_bytes(model)
    rho_mem_weights = bytes_info["total_bytes"] / base_bytes
    print(f"model={a.model} bits={a.bits} layers={len(names)} device={dev} "
          f"seqs={len(seqs)} bytes={bytes_info['total_gb']:.2f}GB "
          f"rho_mem_weights={rho_mem_weights:.3f}", flush=True)

    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    all_out = [None] * len(seqs)
    t0 = time.time()
    for i in range(0, len(order), a.batch_size):
        idxs = order[i:i + a.batch_size]
        outs = eval_batch(model, seqs, cache, idxs, dev)
        for k, o in zip(idxs, outs):
            all_out[k] = o
        if (i // a.batch_size) % 10 == 0:
            print(f"  eval {i + len(idxs)}/{len(seqs)} {time.time() - t0:.0f}s",
                  flush=True)
            torch.cuda.empty_cache()

    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t", "target_ids"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0) for k in keys}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    q_lb_arr = (cat["margin_log"] + cat["d_margin_log"] < 0)
    q_lb = float(q_lb_arr.mean())
    out = {
        "model": a.model, "bits": a.bits, "group": a.group,
        "skip_lm_head": a.skip_lm_head,
        "quantize_embedding": a.quantize_embedding,
        "quantized_layers": len(names),
        "data": a.data, "cache": a.cache, "seqs": len(seqs),
        "tokens": int(len(match)), "q": q, "match_rate": float(match.mean()),
        "E_TV": float(cat["tv"].mean()),
        "E_margin": float(cat["margin"].mean()),
        "E_abs_d_margin_log": float(cat["abs_d_margin_log"].mean()),
        "E_err_k": float(cat["err_k"].mean()),
        "q_lb_top2": q_lb,
        "q_lb_ci95": bootstrap_lb(q_lb_arr.astype(np.float64), seq_id, len(seqs)),
        "q_ci95": bootstrap_lb(match.astype(np.float64), seq_id, len(seqs)),
        "bytes": bytes_info,
        "target_bf16_bytes": int(base_bytes),
        "rho_mem_weights": rho_mem_weights,
        "per_gamma": {},
        "eval_seconds": time.time() - t0,
    }
    from infra.metrics import simulate_blocks
    out["per_gamma"] = simulate_blocks(match, [int(x) for x in a.gammas.split(",")])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out.replace(".json", ".npz"), **cat, gen_len=seq_len,
             seq_id=seq_id)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ["q", "q_lb_top2", "E_TV", "rho_mem_weights", "tokens"]},
                     indent=1))


if __name__ == "__main__":
    main()
