"""Per-layer (non-uniform) FFN width allocation at a fixed average width.

This tests whether the high-width rejection floor is a channel-selection or
allocation artifact: give sensitive layers more width and insensitive layers less,
while keeping the parameter-weighted average width at the same nominal value.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m infra.nonuniform_width --model qwen3-4b \
    --data data/processed/qwen3_wikitext_target.jsonl \
    --cache out/target_cache/qwen3_wikitext_k256.npz \
    --importance out/importance/qwen3-4b-qwen3_act.pt \
    --pattern frontback,alternating,random --avg 0.98 --out out/rank_scan/nonuniform_qwen3
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
from infra.joint_scan import make_narrow
from infra.layer_scan import _summarize
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, _seq_q


def pattern_fracs(kind, n, avg, rng, edge=3, spread=0.04):
    if kind == "uniform":
        return [avg] * n
    if kind == "frontback":
        w = np.full(n, avg, dtype=np.float64)
        w[:edge] = min(1.0, avg + spread)
        w[-edge:] = min(1.0, avg + spread)
        # solve the middle value so that the mean is avg
        rest = n - 2 * edge
        w[edge:-edge] = (avg * n - w[:edge].sum() - w[-edge:].sum()) / rest
        return [float(x) for x in w]
    if kind == "alternating":
        w = np.array([avg + spread if i % 2 == 0 else avg - spread for i in range(n)])
        w = np.clip(w, 0.0, 1.0)
        w += avg - w.mean()
        return [float(x) for x in np.clip(w, 0.0, 1.0)]
    if kind == "random":
        w = rng.normal(avg, spread, n)
        w += avg - w.mean()
        return [float(x) for x in np.clip(w, 0.0, 1.0)]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--importance", default=None)
    ap.add_argument("--patterns", default="uniform,frontback,alternating,random")
    ap.add_argument("--avg", type=float, default=0.98)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs = data.load_target_sequences(a.data, limit=a.limit)
    cache = _load_cache(a.cache)
    gammas = [int(x) for x in a.gammas.split(",")]
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    rng = np.random.default_rng(0)

    os.makedirs(a.out, exist_ok=True)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    dev = model_device(model)
    n_layers = len(model.model.layers)
    rows = []
    for kind in a.patterns.split(","):
        fr = pattern_fracs(kind, n_layers, a.avg, rng)
        # rebuild from a clean model for every pattern
        del model
        torch.cuda.empty_cache()
        model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
        for i, layer in enumerate(model.model.layers):
            layer.mlp, _ = make_narrow(layer, fr[i], "width", model_device(layer),
                                       imp.get(f"l{i:02d}") if imp else None,
                                       pad_to=128)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
        outs = [None] * len(seqs)
        t0 = time.time()
        for i in range(0, len(order), a.batch_size):
            idxs = order[i:i + a.batch_size]
            for k, o in zip(idxs, eval_batch(model, seqs, cache, idxs, dev)):
                outs[k] = o
        summ, cat, seq_len, seq_id = _summarize(outs, gammas, len(seqs))
        summ.update({"pattern": kind, "avg_width": a.avg,
                     "fracs": [float(x) for x in fr], "model": a.model,
                     "set": os.path.basename(a.data)})
        rows.append(summ)
        print(json.dumps({k: summ[k] for k in ["pattern", "q", "q_lb", "E_TV"]}), flush=True)
        np.savez(os.path.join(a.out, f"{kind}_{a.avg:g}.npz"), **cat,
                 gen_len=seq_len, seq_id=seq_id,
                 q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
        with open(os.path.join(a.out, f"{kind}_{a.avg:g}.json"), "w") as f:
            json.dump(summ, f, indent=1)
    with open(os.path.join(a.out, f"summary_{a.avg:g}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
