"""Post-hoc attention-head pruning (and combined FFN-width + head pruning).

This instantiates the support-selection principle for attention heads: a head
can be removed exactly only if its output is identically zero, so every
compute-reducing head removal perturbs the residual stream.  We physically
slice the query/key/value/output projections of standard GQA attention, so the
reported nominal FLOP ratio is a genuine narrower model, not a masked head.

Head importance is activation-aware: on a calibration set we record the norm
of each head's contribution to the input of ``o_proj`` and rank KV-head groups
by the sum of their query-head scores.  For GQA we remove whole KV groups so
that the query/key/value shapes remain consistent.

The script reports q, E[TV], q_lb (top-2), and q_lb^(K) at selected K, using the
same target top-256 cache as the FFN scans.  It can be combined with a FFN
width reduction for the combined-operator experiment.
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
from infra.model_loader import load_text_model, load_tokenizer, model_device
from infra.model_registry import resolve
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q


@torch.inference_mode()
def head_importance(model, tok, rows, max_tokens, max_len, device):
    """Return per-layer query-head importance from ||head output|| on calibration."""
    acc = {}
    counts = {}
    handles = []

    def make_hook(layer_idx, n_heads):
        def hook(module, args):
            x = args[0]
            b, s, d = x.shape
            hd = d // n_heads
            xh = x.reshape(b, s, n_heads, hd).float()
            v = xh.pow(2).sum(-1).mean(dim=(0, 1))  # [n_heads]
            acc.setdefault(layer_idx, []).append(v.detach().cpu())
            counts[layer_idx] = counts.get(layer_idx, 0) + b * s
        return hook

    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "o_proj"):
            continue
        n_heads = int(getattr(attn.config, "num_attention_heads",
                              attn.o_proj.in_features // getattr(attn, "head_dim", 1)))
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(i, n_heads)))

    n = 0
    for r in rows:
        text = r.get("text") or r.get("prompt") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 16:
            continue
        model(input_ids=ids, use_cache=False)
        n += ids.shape[1]
        if n >= max_tokens:
            break
    for h in handles:
        h.remove()
    scores = {}
    for i, vals in acc.items():
        scores[i] = torch.stack(vals, 0).mean(0).numpy()
    print(f"head importance from {n} calibration tokens", flush=True)
    return scores


def _kv_groups(attn):
    n_heads = int(attn.config.num_attention_heads)
    n_kv = int(attn.config.num_key_value_heads)
    assert n_heads % n_kv == 0, (n_heads, n_kv)
    return n_heads // n_kv


def prune_heads(model, keep_ratio, scores):
    """Physically remove whole KV groups with the lowest activation score."""
    stats = []
    # Snapshot the original head counts before any layer is modified: the
    # attention modules share the model config object.
    n_heads0 = int(model.config.num_attention_heads)
    n_kv0 = int(model.config.num_key_value_heads)
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj"):
            continue
        n_heads = n_heads0
        n_kv = n_kv0
        groups = n_heads // n_kv
        head_dim = int(attn.head_dim)
        s = scores.get(i)
        if s is None:
            s = np.ones(n_heads, dtype=np.float32)
        # score per KV head = sum of its query-head group
        kv_score = np.array([s[j * groups:(j + 1) * groups].sum() for j in range(n_kv)])
        n_keep_kv = max(1, int(round(keep_ratio * n_kv)))
        kept_kv = np.sort(np.argsort(-kv_score)[:n_keep_kv])
        kept_q = np.concatenate([np.arange(j * groups, (j + 1) * groups) for j in kept_kv])
        n_keep_q = len(kept_q)

        def take_rows(linear, idx):
            if not torch.is_tensor(idx):
                idx = torch.from_numpy(idx)
            idx = idx.to(linear.weight.device)
            w = linear.weight.detach()[idx].contiguous()
            b = None if linear.bias is None else linear.bias.detach()[idx].contiguous()
            new = nn.Linear(linear.in_features, len(idx), bias=linear.bias is not None,
                            device=w.device, dtype=w.dtype)
            new.weight.data.copy_(w)
            if b is not None:
                new.bias.data.copy_(b)
            return new

        kept_q_rows = np.concatenate([np.arange(h * head_dim, (h + 1) * head_dim)
                                      for h in kept_q])
        kept_kv_rows = np.concatenate([np.arange(j * head_dim, (j + 1) * head_dim)
                                       for j in kept_kv])
        attn.q_proj = take_rows(attn.q_proj, kept_q_rows)
        attn.k_proj = take_rows(attn.k_proj, kept_kv_rows)
        attn.v_proj = take_rows(attn.v_proj, kept_kv_rows)
        # o_proj input columns correspond to query heads
        ow = attn.o_proj.weight.detach()
        cols = torch.from_numpy(kept_q_rows).to(ow.device)
        new_o = nn.Linear(len(cols), ow.shape[0], bias=attn.o_proj.bias is not None,
                          device=ow.device, dtype=ow.dtype)
        new_o.weight.data.copy_(ow[:, cols].contiguous())
        if attn.o_proj.bias is not None:
            new_o.bias.data.copy_(attn.o_proj.bias.detach())
        attn.o_proj = new_o

        attn.num_key_value_groups = int(n_keep_q // n_keep_kv)
        stats.append({"layer": i, "kept_kv": n_keep_kv, "total_kv": n_kv,
                      "kept_q": n_keep_q, "total_q": n_heads})
    return stats


def _cost_ratio(model_key, width, head_keep):
    s = param_stats(resolve(model_key))
    # attention parameters scale with the retained head fraction; MLP with width
    rho = (s.compute - s.mlp - s.full_attn - s.linear_attn
           + s.mlp * width + s.full_attn * head_keep + s.linear_attn) / s.compute
    return {"rho_FLOP": float(rho), "mlp_share": s.mlp / s.compute,
            "attn_share": (s.full_attn + s.linear_attn) / s.compute,
            "full_attn_share": s.full_attn / s.compute}


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
    ap.add_argument("--model", default="llama31-8b")
    ap.add_argument("--importance", default=None,
                    help="optional activation-importance .pt for the FFN width part")
    ap.add_argument("--mode", default="head", choices=["head", "width_head"])
    ap.add_argument("--head-keeps", default="0.75,0.5,0.25")
    ap.add_argument("--widths", default="1.0")
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--calib-tokens", type=int, default=50000)
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
    tok = load_tokenizer(a.model)
    rows = [json.loads(line) for line in open(a.calib)]
    scores = head_importance(model, tok, rows, a.calib_tokens, 1024, dev)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    orig = [layer.mlp for layer in model.model.layers]
    # snapshot attention modules for restoring between cells
    import copy
    orig_attn = [copy.deepcopy(layer.self_attn) if hasattr(layer, "self_attn") else None
                 for layer in model.model.layers]
    gammas = [int(x) for x in a.gammas.split(",")]

    for hk in [float(x) for x in a.head_keeps.split(",")]:
        for width in [float(x) for x in a.widths.split(",")]:
            # restore base attention + MLP
            for layer, mlp, attn in zip(model.model.layers, orig, orig_attn):
                layer.mlp = mlp
                if attn is not None:
                    layer.self_attn = copy.deepcopy(attn)
            if width < 1.0:
                for i, layer in enumerate(model.model.layers):
                    layer.mlp, _ = make_narrow(layer, width, "width", next(layer.parameters()).device,
                                               imp.get(f"l{i:02d}") if imp else None)
            stats = prune_heads(model, hk, scores)
            cost = _cost_ratio(a.model, width, hk)
            tag = f"head{hk:g}_width{width:g}"
            with open(os.path.join(a.out, f"compress_{tag}.json"), "w") as f:
                json.dump({"head_stats": stats, **cost}, f, indent=1)
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
                summ.update({"set": name, "model": a.model, "mode": a.mode,
                             "head_keep": hk, "width": width, **cost})
                stem = os.path.join(a.out, f"{name}_{tag}")
                np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                         q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
                with open(stem + ".json", "w") as f:
                    json.dump(summ, f, indent=1)
                print(f"  {name}: q={summ['q']:.4f} lb={summ['q_lb']:.4f} "
                      f"rho={summ['rho_FLOP']:.3f}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
