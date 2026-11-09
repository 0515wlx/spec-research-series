"""Post-hoc depth reduction scan: drop decoder layers from a dense target.

This is a compute-reducing post-hoc operator outside the FFN-width family.  It is
used as an operator-zoo control: a dropped layer changes the residual stream, so the
perturbation is not confined to deleted coordinates.  The target top-k cache is
built from the uncompressed target, so the full-depth draft has q=0.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from infra import data
from infra.accept_eval import eval_batch
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q
from infra.metrics import simulate_blocks


def _layer_cost_ratio(model_key: str, drop: int, total_layers: int) -> dict:
    from infra.flops import param_stats
    from infra.model_registry import resolve
    s = param_stats(resolve(model_key))
    layer_params = s.mlp + s.full_attn + s.linear_attn
    non_layer = max(s.compute - layer_params, 0)
    ratio = (non_layer + layer_params * (total_layers - drop) / total_layers) / s.compute
    return {"drop_layers": drop, "total_layers": total_layers,
            "layer_share": layer_params / s.compute, "rho_FLOP": float(ratio)}


def _summarize(all_out, gammas, n_seq):
    keys = ["match", "tv", "margin_log", "d_margin_log", "margin"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0) for k in keys}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    flip = (cat["margin_log"].astype(np.float64) + cat["d_margin_log"].astype(np.float64)) < 0
    q = 1.0 - float(match.mean())
    qb = bootstrap_q(match, seq_id, n_seq)
    rng = np.random.default_rng(0)
    per = np.zeros(n_seq); cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, flip.astype(np.float64)); np.add.at(cnt, seq_id, 1)
    idx = rng.integers(0, n_seq, size=(2000, n_seq))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return {
        "q": q, "q_ci": list(qb), "q_lb": float(flip.mean()),
        "q_lb_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "E_TV": float(cat["tv"].mean()), "tokens": int(len(match)),
        "seqs": n_seq, "per_gamma": simulate_blocks(match, gammas),
    }, cat, seq_len, seq_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--drops", default="1,2,4,8,16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    gammas = [int(x) for x in a.gammas.split(",")]
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
        print(f"set {name}: {len(seqs)} seqs", flush=True)

    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    layers = list(model.model.layers)
    n = len(layers)
    print(f"loaded {a.model} with {n} layers", flush=True)

    for drop in [int(x) for x in a.drops.split(",")]:
        if drop <= 0 or drop >= n:
            continue
        tag = f"drop{drop}"
        out_json = os.path.join(a.out, f"{tag}.json")
        if os.path.exists(out_json):
            print(f"skip {tag}", flush=True)
            continue
        t0 = time.time()
        model.model.layers = nn.ModuleList(layers[: n - drop])
        cost = _layer_cost_ratio(a.model, drop, n)
        rows = []
        for name, seqs, cache in datasets:
            print(f"  {tag} set={name}", flush=True)
            order = sorted(range(len(seqs)),
                           key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
            all_out = [None] * len(seqs)
            for i in range(0, len(order), a.batch_size):
                idxs = order[i:i + a.batch_size]
                outs = eval_batch(model, seqs, cache, idxs, dev)
                for kk, o in zip(idxs, outs):
                    all_out[kk] = o
            summ, cat, seq_len, seq_id = _summarize(all_out, gammas, len(seqs))
            summ.update({"set": name, "model": a.model, "kind": "layerdrop", **cost})
            stem = os.path.join(a.out, f"{name}_{tag}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                     q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            rows.append(summ)
            print(f"    q={summ['q']:.4f} q_lb={summ['q_lb']:.4f} "
                  f"rho={summ['rho_FLOP']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=1)
        model.model.layers = nn.ModuleList(layers)
        torch.cuda.empty_cache()
        print(f"  wrote {out_json} ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
