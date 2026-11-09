"""Evaluate a model whose FFN blocks have been spectrally or structurally compressed.

The target distribution is a fixed top-k cache built from the *uncompressed*
model.  This isolates the effect of compression from target/draft model mismatch:
at full rank the compressed draft is the target itself and ``q=0`` on every set.

Example
-------
CUDA_VISIBLE_DEVICES=0 python -m infra.rank_scan \
  --model qwen3.5-9b --factors out/svd/qwen3.5-9b --mode svd \
  --fracs 0.05,0.1,0.2,1.0 \
  --sets math9b:data/processed/math500_target_qwen3.5-9b.jsonl:out/target_cache/math500_target9b_k256.npz \
         wikitext:data/processed/wikitext_target.jsonl:out/target_cache/wikitext_target9b_k256.npz \
  --out out/rank_scan/qwen3.5-9b
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from infra import svd_compress
from infra.accept_eval import eval_batch
from infra.model_loader import load_text_model, model_device


def _load_cache(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _seq_q(match, seq_id, n_seq):
    out = np.zeros(n_seq, dtype=np.float64)
    cnt = np.zeros(n_seq, dtype=np.int64)
    np.add.at(out, seq_id, match.astype(np.float64))
    np.add.at(cnt, seq_id, 1)
    return out / np.maximum(cnt, 1), cnt


def bootstrap_q(match, seq_id, n_seq, B=2000, seed=0):
    per_seq, cnt = _seq_q(match, seq_id, n_seq)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    # token-weighted resampling of whole sequences
    num = (per_seq[idx] * cnt[idx]).sum(1)
    den = cnt[idx].sum(1)
    qs = 1.0 - num / np.maximum(den, 1)
    return float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5))


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


def summarize(all_out, gammas):
    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t"]
    cat = {}
    for k in keys:
        if all_out[0].get(k) is not None:
            cat[k] = np.concatenate([o[k] for o in all_out], axis=0)
    seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    from infra.metrics import simulate_blocks
    out = {
        "q": q,
        "tokens": int(len(match)),
        "seqs": len(all_out),
        "E_TV": float(cat["tv"].mean()),
        "E_margin": float(cat["margin"].mean()),
        "E_abs_d_margin_log": float(cat["abs_d_margin_log"].mean()),
        "E_err_k": float(cat["err_k"].mean()),
        "E_ent_t": float(cat["ent_t"].mean()),
        "q_ci": bootstrap_q(match, seq_id, len(all_out)),
        "per_gamma": simulate_blocks(match, gammas),
    }
    return out, cat, seq_len, seq_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--factors", default=None, help="SVD factor directory (mode=svd)")
    ap.add_argument("--mode", default="svd", choices=["svd", "width"])
    ap.add_argument("--fracs", default="0.05,0.1,0.2,1.0")
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name:data.jsonl:cache.npz entries")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16,32,64")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from infra import data
    os.makedirs(a.out, exist_ok=True)
    gammas = [int(x) for x in a.gammas.split(",")]
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        cache = _load_cache(cpath)
        datasets.append((name, seqs, cache))
        print(f"set {name}: {len(seqs)} seqs, cache {cpath}", flush=True)

    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    # Snapshot original MLP weights so that each rank starts from the base model.
    orig = {}
    for i, layer in enumerate(model.model.layers):
        for n in svd_compress.MLP_NAMES:
            orig[f"{i}_{n}"] = getattr(layer.mlp, n).weight.detach().cpu().clone()
    print(f"snapshot {len(orig)} MLP tensors", flush=True)

    for frac_s in a.fracs.split(","):
        frac = float(frac_s)
        for i, layer in enumerate(model.model.layers):
            for n in svd_compress.MLP_NAMES:
                w = getattr(layer.mlp, n).weight
                w.data.copy_(orig[f"{i}_{n}"].to(w.device, w.dtype))
        t0 = time.time()
        if a.mode == "svd":
            info = svd_compress.apply_svd(model, a.factors, frac)
        else:
            info = svd_compress.apply_width(model, frac)
        print(f"frac={frac} compressed in {time.time()-t0:.1f}s", flush=True)
        with open(os.path.join(a.out, f"compress_frac{frac:g}.json"), "w") as f:
            json.dump(info, f)
        for name, seqs, cache in datasets:
            print(f"  frac={frac} set={name}", flush=True)
            outs = eval_model(model, seqs, cache, a.batch_size, gammas, dev)
            summ, cat, seq_len, seq_id = summarize(outs, gammas)
            summ.update({"model": a.model, "mode": a.mode, "frac": frac,
                         "set": name, "factors": a.factors})
            stem = os.path.join(a.out, f"{name}_frac{frac:g}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                     q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            print(f"    q={summ['q']:.4f} tv={summ['E_TV']:.4f} "
                  f"ci={summ['q_ci']}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
