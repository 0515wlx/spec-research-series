"""Width-reduction scan with a shared (joint) intermediate basis.

``joint``
    For each layer, choose the rank-``r`` shared row space of the gate/up
    output matrices, i.e. the top eigenvectors of
    ``W_gate W_gate^T + W_up W_up^T``.  The compressed block is

        g = P W_gate x,  u = P W_up x,  y = W_down P^T (act(g) * u),

    which is exact at ``r = intermediate_size`` and is the optimal shared
    rank-``r`` projection of the two gate/up maps in Frobenius norm.

``width``
    Keep the ``r`` intermediate channels with the largest combined
    gate/up/down weight norm (structured pruning).

Both operators produce a genuinely narrower FFN, so the cost ratio is
``r / intermediate_size`` for the MLP block (not a factored dense matrix).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from infra import data, svd_compress
from infra.accept_eval import eval_batch
from infra.model_loader import load_text_model, model_device
from infra.rank_scan import _load_cache, bootstrap_q, _seq_q


class NarrowMLP(nn.Module):
    def __init__(self, cfg, hidden, inter, gate_w, up_w, down_w, act_fn):
        super().__init__()
        self.config = cfg
        self.hidden_size = hidden
        self.intermediate_size = inter
        self.act_fn = act_fn
        self.register_buffer("gate_w", gate_w.contiguous())
        self.register_buffer("up_w", up_w.contiguous())
        self.register_buffer("down_w", down_w.contiguous())

    def forward(self, x):
        g = F.linear(x, self.gate_w)
        u = F.linear(x, self.up_w)
        return F.linear(self.act_fn(g) * u, self.down_w)


def make_narrow(layer, frac, mode, device, importance=None, pad_to=0):
    mlp = layer.mlp
    wg = mlp.gate_proj.weight.detach().float()
    wu = mlp.up_proj.weight.detach().float()
    wd = mlp.down_proj.weight.detach().float()
    h, d = wg.shape[1], wg.shape[1]
    inter = wg.shape[0]
    r = max(1, int(round(frac * inter)))
    if mode == "width":
        if importance is not None:
            score = importance.float()
        else:
            score = (wg.pow(2).sum(1) + wu.pow(2).sum(1) + wd.pow(2).sum(0))
        idx = torch.topk(score, r).indices.sort().values
        gw, uw, dw = wg[idx, :], wu[idx, :], wd[:, idx]
        info = {"kept_score": float(score[idx].sum()), "total_score": float(score.sum())}
    elif mode == "joint":
        m = wg @ wg.T + wu @ wu.T          # (inter, inter), symmetric PSD
        evals, evecs = torch.linalg.eigh(m)
        p = evecs[:, -r:].T                 # (r, inter), orthonormal rows
        gw = p @ wg
        uw = p @ wu
        dw = wd @ p.T
        info = {"eig_energy": float(evals[-r:].sum()), "eig_total": float(evals.sum())}
        del m, evals, evecs, p
    else:
        raise ValueError(mode)
    if pad_to:
        rp = ((r + pad_to - 1) // pad_to) * pad_to
        if rp != r:
            gwp = torch.zeros(rp, gw.shape[1], dtype=gw.dtype)
            uwp = torch.zeros(rp, uw.shape[1], dtype=uw.dtype)
            dwp = torch.zeros(dw.shape[0], rp, dtype=dw.dtype)
            gwp[:r] = gw; uwp[:r] = uw; dwp[:, :r] = dw
            gw, uw, dw = gwp, uwp, dwp
            info["padded_rank"] = rp
    dtype = mlp.gate_proj.weight.dtype
    out = NarrowMLP(mlp.config, h, gw.shape[0], gw.to(device, dtype),
                    uw.to(device, dtype), dw.to(device, dtype),
                    mlp.act_fn).to(device)
    info.update({"rank": r, "inter": inter})
    del wg, wu, wd, gw, uw, dw
    return out, info


@torch.inference_mode()
def eval_model(model, seqs, cache, batch_size, gammas, device):
    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    all_out = [None] * len(seqs)
    t0 = time.time()
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        outs = eval_batch(model, seqs, cache, idxs, device)
        for k, o in zip(idxs, outs):
            all_out[k] = o
        if (i // batch_size) % 20 == 0:
            print(f"    eval {i + len(idxs)}/{len(seqs)} {time.time()-t0:.0f}s", flush=True)
    return all_out


def summarize(all_out, gammas, n_seq):
    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t"]
    cat = {k: np.concatenate([o[k] for o in all_out], axis=0)
           for k in keys if all_out[0].get(k) is not None}
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    from infra.metrics import simulate_blocks
    return ({
        "q": 1.0 - float(match.mean()), "tokens": int(len(match)),
        "seqs": len(all_out), "E_TV": float(cat["tv"].mean()),
        "E_margin": float(cat["margin"].mean()),
        "E_abs_d_margin_log": float(cat["abs_d_margin_log"].mean()),
        "E_err_k": float(cat["err_k"].mean()),
        "q_ci": bootstrap_q(match, seq_id, len(all_out)),
        "per_gamma": simulate_blocks(match, gammas),
    }, cat, seq_len, seq_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--mode", default="joint", choices=["joint", "width"])
    ap.add_argument("--fracs", default="0.05,0.1,0.2,0.5,0.8,1.0")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16,32,64")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--importance", default=None,
                    help="path to activation-importance .pt from infra.activation_importance")
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
    orig = [layer.mlp for layer in model.model.layers]
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    print(f"loaded {len(orig)} MLP blocks" + (" with activation importance" if imp else ""), flush=True)

    for frac_s in a.fracs.split(","):
        frac = float(frac_s)
        infos = {}
        t0 = time.time()
        for i, layer in enumerate(model.model.layers):
            layer_dev = next(layer.parameters()).device
            layer.mlp, info = make_narrow(layer, frac, a.mode, layer_dev,
                                          imp.get(f"l{i:02d}") if imp else None)
            infos[f"l{i:02d}"] = info
        print(f"frac={frac} compressed in {time.time()-t0:.1f}s", flush=True)
        with open(os.path.join(a.out, f"compress_frac{frac:g}.json"), "w") as f:
            json.dump(infos, f)
        for name, seqs, cache in datasets:
            print(f"  frac={frac} set={name}", flush=True)
            outs = eval_model(model, seqs, cache, a.batch_size, gammas, dev)
            summ, cat, seq_len, seq_id = summarize(outs, gammas, len(seqs))
            summ.update({"model": a.model, "mode": a.mode, "frac": frac, "set": name})
            stem = os.path.join(a.out, f"{name}_frac{frac:g}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                     q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            print(f"    q={summ['q']:.4f} tv={summ['E_TV']:.4f} ci={summ['q_ci']}",
                  flush=True)
            torch.cuda.empty_cache()
        # restore original MLP blocks
        for layer, mlp in zip(model.model.layers, orig):
            layer.mlp = mlp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
