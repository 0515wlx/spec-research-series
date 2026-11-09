"""Lightweight adaptation of a post-hoc narrow draft (adaptation-budget control).

Reviewers asked whether a small adaptation budget can rescue the post-hoc
compressed draft.  This script compresses the target to a given FFN width, makes
the narrowed MLP readout (or the full narrow MLP) trainable, and runs a bounded
number of next-token CE steps on a training corpus.  It then evaluates the same
target top-256 cache as the main paper.

The point is a Pareto boundary: if even raw-text adaptation on the deployment
corpus cannot bring rejection below the frontier, the post-hoc no-go is not an
artifact of "you forgot to fine-tune."  This is not a distillation experiment:
the loss is ordinary next-token CE on the adaptation corpus, not target-response
distillation.

Modes:
  down  -- train only the down projection (fits AdamW on one 80GB GPU)
  mlp   -- train gate/up/down with SGD (no optimizer state)
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.accept_eval import eval_batch
from infra.joint_scan import NarrowMLP, make_narrow
from infra.model_loader import load_text_model, load_tokenizer, model_device
from infra.rank_scan import _load_cache, _seq_q, bootstrap_q


def load_text_rows(path, tok, max_len, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            text = r.get("text") or r.get("prompt") or ""
            if not text:
                continue
            ids = tok(text, add_special_tokens=False, truncation=True,
                      max_length=max_len)["input_ids"]
            if len(ids) < 64:
                continue
            rows.append(ids)
            if limit and len(rows) >= limit:
                break
    return rows


def make_params(draft, mode):
    params = []
    for layer in draft.model.layers:
        mlp = layer.mlp
        if not isinstance(mlp, NarrowMLP):
            continue
        if mode in ("down", "mlp"):
            mlp.down_w = torch.nn.Parameter(mlp.down_w.detach().float())
            params.append(mlp.down_w)
        if mode == "mlp":
            mlp.gate_w = torch.nn.Parameter(mlp.gate_w.detach().float())
            mlp.up_w = torch.nn.Parameter(mlp.up_w.detach().float())
            params.append(mlp.gate_w)
            params.append(mlp.up_w)
    return params


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
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--mode", default="down", choices=["down", "mlp"])
    ap.add_argument("--data", default="data/processed/wikitext_target.jsonl")
    ap.add_argument("--tokens", type=int, default=500000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default=None, choices=[None, "adamw", "sgd"])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    draft = load_text_model(a.model, dtype=a.dtype, device_map={"": dev})
    tok = load_tokenizer(a.model)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    for i, layer in enumerate(draft.model.layers):
        layer.mlp, _ = make_narrow(layer, a.width, "width", next(layer.parameters()).device,
                                   imp.get(f"l{i:02d}"))
    params = make_params(draft, a.mode)
    opt_name = a.optimizer or ("adamw" if a.mode == "down" else "sgd")
    if opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(params, lr=a.lr)
    print(f"mode={a.mode} optimizer={opt_name} trainable={sum(p.numel() for p in params)/1e6:.1f}M",
          flush=True)

    rows = load_text_rows(a.data, tok, a.max_len)
    rng = random.Random(0)
    draft.train()
    seen = step = 0
    hist = []
    while seen < a.tokens:
        ids = rng.choice(rows)
        if len(ids) > a.max_len:
            start = rng.randint(0, len(ids) - a.max_len)
            ids = ids[start:start + a.max_len]
        x = torch.tensor(ids, dtype=torch.long, device=dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = draft(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += len(ids)
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "tokens": seen, "ce": float(loss)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)

    draft.eval()
    for layer in draft.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, NarrowMLP):
            mlp.down_w = torch.nn.Parameter(mlp.down_w.detach().to(torch.bfloat16),
                                            requires_grad=False)
            if a.mode == "mlp":
                mlp.gate_w = torch.nn.Parameter(mlp.gate_w.detach().to(torch.bfloat16),
                                                requires_grad=False)
                mlp.up_w = torch.nn.Parameter(mlp.up_w.detach().to(torch.bfloat16),
                                              requires_grad=False)

    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
    gammas = [int(x) for x in a.gammas.split(",")]
    results = []
    for name, seqs, cache in datasets:
        order = sorted(range(len(seqs)),
                       key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
        all_out = [None] * len(seqs)
        for i in range(0, len(order), a.batch_size):
            idxs = order[i:i + a.batch_size]
            outs = eval_batch(draft, seqs, cache, idxs, dev)
            for k, o in zip(idxs, outs):
                all_out[k] = o
        summ, cat, seq_len, seq_id = summarize(all_out, gammas, len(seqs))
        summ.update({"set": name, "model": a.model, "width": a.width,
                     "adapt_mode": a.mode, "adapt_tokens": a.tokens,
                     "adapt_lr": a.lr, "adapt_optimizer": opt_name})
        stem = os.path.join(a.out, f"{name}_w{a.width:g}_{a.mode}")
        np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                 q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
        with open(stem + ".json", "w") as f:
            json.dump(summ, f, indent=1)
        results.append(summ)
        print(f"  {name}: q={summ['q']:.4f} lb={summ['q_lb']:.4f}", flush=True)
    with open(os.path.join(a.out, "adapt_history.json"), "w") as f:
        json.dump({"history": hist, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
