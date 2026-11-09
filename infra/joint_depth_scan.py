"""Combined FFN-width + depth reduction scan.

The paper's support-selection principle covers both operators, but a reviewer
reasonably asks whether combining them opens a cost/acceptance window.  This
script applies a FFN width reduction to every kept layer and drops the last
``d`` decoder layers, then evaluates the same target cache.  The nominal FLOP
ratio scales the MLP parameter share by ``width`` and the decoder-layer share by
``(n-d)/n``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from infra import data
from infra.accept_eval import eval_batch
from infra.flops import param_stats
from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model, model_device
from infra.model_registry import resolve
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q


def cost_ratio(model_key, width, drop, total_layers):
    s = param_stats(resolve(model_key))
    non_layer = s.compute - s.mlp - s.full_attn - s.linear_attn
    layer = s.mlp * width + s.full_attn + s.linear_attn
    rho = (non_layer + layer * (total_layers - drop) / total_layers) / s.compute
    return {"rho_FLOP": float(rho), "width": width, "drop": drop,
            "mlp_share": s.mlp / s.compute,
            "layer_share": (s.mlp + s.full_attn + s.linear_attn) / s.compute}


def summarize(all_out, gammas, n_seq):
    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0)
           for k in keys if all_out[0].get(k) is not None}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    flip = (cat["margin_log"].astype(np.float64) + cat["d_margin_log"].astype(np.float64)) < 0
    rng = np.random.default_rng(0)
    per = np.zeros(n_seq); cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, flip.astype(np.float64)); np.add.at(cnt, seq_id, 1)
    idx = rng.integers(0, n_seq, size=(2000, n_seq))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return ({
        "q": q, "q_ci": list(bootstrap_q(match, seq_id, n_seq)),
        "q_lb": float(flip.mean()),
        "q_lb_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "E_TV": float(cat["tv"].mean()), "tokens": int(len(match)),
        "seqs": n_seq,
    }, cat, seq_len, seq_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--widths", default="0.5,0.7,0.9,0.98")
    ap.add_argument("--drops", default="1,2")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
        print(f"set {name}: {len(seqs)} seqs", flush=True)

    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    layers = list(model.model.layers)
    n = len(layers)
    orig = [layer.mlp for layer in layers]
    gammas = [int(x) for x in a.gammas.split(",")]

    for drop in [int(x) for x in a.drops.split(",")]:
        if drop <= 0 or drop >= n:
            continue
        model.model.layers = nn.ModuleList(layers[: n - drop])
        for width in [float(x) for x in a.widths.split(",")]:
            for i, layer in enumerate(model.model.layers):
                layer.mlp, _ = make_narrow(layer, width, "width", next(layer.parameters()).device,
                                           imp.get(f"l{i:02d}") if imp else None)
            cost = cost_ratio(a.model, width, drop, n)
            tag = f"width{width:g}_drop{drop}"
            with open(os.path.join(a.out, f"compress_{tag}.json"), "w") as f:
                json.dump(cost, f, indent=1)
            print(f"cell {tag} rho_FLOP={cost['rho_FLOP']:.4f}", flush=True)
            for name, seqs, cache in datasets:
                order = sorted(range(len(seqs)),
                               key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
                all_out = [None] * len(seqs)
                for i in range(0, len(order), a.batch_size):
                    idxs = order[i:i + a.batch_size]
                    outs = eval_batch(model, seqs, cache, idxs, dev)
                    for k, o in zip(idxs, outs):
                        all_out[k] = o
                summ, cat, seq_len, seq_id = summarize(all_out, gammas, len(seqs))
                summ.update({"set": name, "model": a.model, "mode": "width_depth",
                             **cost})
                stem = os.path.join(a.out, f"{name}_{tag}")
                np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                         q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
                with open(stem + ".json", "w") as f:
                    json.dump(summ, f, indent=1)
                print(f"  {name}: q={summ['q']:.4f} lb={summ['q_lb']:.4f} "
                      f"rho={summ['rho_FLOP']:.3f}", flush=True)
                torch.cuda.empty_cache()
            # restore MLPs before the next width
            for layer, mlp in zip(model.model.layers, orig[: n - drop]):
                layer.mlp = mlp
            torch.cuda.empty_cache()
        model.model.layers = nn.ModuleList(layers)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
