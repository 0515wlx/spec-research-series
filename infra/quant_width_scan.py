"""Quality scan for the combined structured width + weight-only quantization operator.

The review asks whether the paper's no-go covers the operator obtained by
composing the two axes: reduce the FFN width *and* store the remaining weights in
INT4.  This script measures that composed draft against the same bf16 target
cache used elsewhere, so ``q`` is the top-1 flip rate of the composed operator,
not a capacity gap.

The FLOP ratio is unchanged by quantization (``rho_FLOP`` is the width ratio);
the memory ratio is the actual packed-byte ratio returned by
``quant_kernel.quantized_bytes``.  We report both frontiers and the measured
``q`` / ``q_lb``.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m infra.quant_width_scan --model qwen3.5-9b \
    --width 0.5 --bits 4 --data data/processed/wikitext_target.jsonl \
    --cache out/target_cache/wikitext_target9b_k256.npz \
    --out out/quant/combined_w050_w4_wikitext.json
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
from infra.cost_frontier import q_star
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device
from infra.quant_kernel import (Int4NarrowMLP, apply_int4, quantized_bytes)


def bootstrap_lb(x, seq_id, n_seq, B=2000, seed=0, q=0.025):
    per = np.zeros(n_seq)
    cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, x.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return [float(np.percentile(boot, 100 * q)),
            float(np.percentile(boot, 100 * (1 - q)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--bits", type=int, choices=[4], default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--inner-k-tiles", type=int, default=8)
    ap.add_argument("--skip-lm-head", action="store_true", default=True)
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gammas", default="1,2,4,8")
    ap.add_argument("--mlp-share", type=float, default=0.6088)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    z = np.load(a.cache)
    cache = {k: z[k] for k in z.files}
    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    target_bytes = sum(p.numel() for p in model.parameters()) * 2
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)

    # 1) structured width reduction.  The packed INT4 layout needs the
    # intermediate dimension to be a multiple of 128; pad with zero channels
    # (which leave the function unchanged) and report the effective width.
    padded = 0
    for i, layer in enumerate(model.model.layers):
        layer.mlp, info = make_narrow(layer, a.width, "width",
                                      next(layer.parameters()).device,
                                      imp.get(f"l{i:02d}") if imp else None,
                                      pad_to=128)
        padded = int(info.get("padded_rank", info["rank"]))
        inter = int(info["inter"])
    width_eff = padded / inter
    # 2) weight-only quantization of all remaining linears
    names = apply_int4(model, group_size=a.group,
                       inner_k_tiles=a.inner_k_tiles,
                       skip_lm_head=a.skip_lm_head, skip_embedding=True)
    # 3) quantize the narrow FFN matrices themselves
    for i, layer in enumerate(model.model.layers):
        layer.mlp = Int4NarrowMLP.from_narrow(
            layer.mlp, group_size=a.group, inner_k_tiles=a.inner_k_tiles)

    bytes_info = quantized_bytes(model)
    rho_mem = bytes_info["total_bytes"] / target_bytes
    rho_flop = 1.0 - a.mlp_share * (1.0 - width_eff)
    gammas = [int(x) for x in a.gammas.split(",")]

    print(f"model={a.model} width={a.width} bits={a.bits} "
          f"layers={len(names)} seqs={len(seqs)} "
          f"rho_flop={rho_flop:.4f} rho_mem={rho_mem:.4f}", flush=True)

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
            print(f"  eval {i + len(idxs)}/{len(seqs)} {time.time()-t0:.0f}s",
                  flush=True)
            torch.cuda.empty_cache()

    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t", "target_ids"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0) for k in keys}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    flip = (cat["margin_log"].astype(np.float64)
            + cat["d_margin_log"].astype(np.float64)) < 0
    q = 1.0 - float(match.mean())
    q_lb = float(flip.mean())
    n_seq = len(seqs)
    from infra.metrics import simulate_blocks
    out = {
        "model": a.model, "width": a.width, "width_effective": width_eff,
        "rank": padded, "intermediate": inter,
        "bits": a.bits,
        "group": a.group, "skip_lm_head": a.skip_lm_head,
        "quantized_layers": len(names), "data": a.data, "cache": a.cache,
        "seqs": len(seqs), "tokens": int(len(match)),
        "q": q, "match_rate": float(match.mean()), "q_lb_top2": q_lb,
        "q_ci95": bootstrap_lb(~match, seq_id, n_seq),
        "q_lb_ci95": bootstrap_lb(flip, seq_id, n_seq),
        "E_TV": float(cat["tv"].mean()),
        "E_margin": float(cat["margin"].mean()),
        "E_abs_d_margin_log": float(cat["abs_d_margin_log"].mean()),
        "rho_flop": rho_flop,
        "rho_mem": rho_mem,
        "q_star_flop": q_star(rho_flop, 4),
        "q_star_mem": q_star(rho_mem, 4),
        "bytes": bytes_info, "target_bf16_bytes": int(target_bytes),
        "per_gamma": simulate_blocks(match, gammas),
        "eval_seconds": time.time() - t0,
    }
    out["no_go_flop_point"] = bool(q > out["q_star_flop"])
    out["no_go_flop_lb"] = bool(out["q_lb_ci95"][0] > out["q_star_flop"])
    out["no_go_mem_point"] = bool(q > out["q_star_mem"])
    out["no_go_mem_lb"] = bool(out["q_lb_ci95"][0] > out["q_star_mem"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out.replace(".json", ".npz"), **cat, gen_len=seq_len,
             seq_id=seq_id)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ["q", "q_lb_top2", "rho_flop", "rho_mem",
                       "q_star_flop", "q_star_mem", "no_go_flop_lb",
                       "no_go_mem_lb", "tokens"]}, indent=1))


if __name__ == "__main__":
    main()
