"""Margin-aware post-hoc calibration (no target responses, only top-1/top-2 labels).

Reviewers asked whether the MSE reconstruction failure is simply the wrong
objective.  This script tests the strongest cheap alternative: after structured
FFN pruning, optimize the narrow down-projections on a calibration set with a
top-1 cross-entropy plus a target-margin hinge.  The calibration signal is
*not* target-generated text; it is the target's top-1/top-2 decision on
ordinary calibration tokens (a label, not a response).

We train only the down projections (the same parameters as the ridge recovery),
so the comparison with ridge is apples-to-apples.  Evaluation uses the fixed
target top-256 cache and the same sequence-cluster bootstrap as the rest of the
paper.
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


@torch.inference_mode()
def collect_labels(target, tok, rows, max_tokens, max_len, device):
    """Return list of (input_ids, top1, top2, target_margin) sequences."""
    out = []
    n = 0
    for r in rows:
        text = r.get("text") or r.get("prompt") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
        if ids.shape[1] < 32:
            continue
        ids = ids.to(device)
        lg = target(input_ids=ids, use_cache=False).logits.float()[0]
        # position t predicts token t+1; score all next-token positions
        lg = lg[:-1]
        top = lg.topk(2, dim=-1)
        out.append({
            "ids": ids[0, :-1].cpu(),
            "top1": top.indices[:, 0].cpu(),
            "top2": top.indices[:, 1].cpu(),
            "margin": (top.values[:, 0] - top.values[:, 1]).cpu(),
        })
        n += ids.shape[1]
        if n >= max_tokens:
            break
    print(f"collected {len(out)} calibration sequences / {n} tokens", flush=True)
    return out


def make_trainable(draft):
    params = []
    for layer in draft.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, NarrowMLP):
            mlp.down_w = torch.nn.Parameter(mlp.down_w.detach().float())
            params.append(mlp.down_w)
    print(f"trainable down-projection tensors: {len(params)}", flush=True)
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
    ap.add_argument("--calib", default="data/processed/fineweb_train.jsonl")
    ap.add_argument("--calib-tokens", type=int, default=100000)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-seqs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin-weight", type=float, default=0.5)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gammas", default="1,2,4,8,16")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tokens = [json.loads(line) for line in open(a.calib)]
    tok = load_tokenizer(a.model)
    dev = "cuda:0"
    target = load_text_model(a.model, dtype=a.dtype, device_map={"": dev})
    rows = collect_labels(target, tok, tokens, a.calib_tokens, a.calib_len, dev)
    del target
    torch.cuda.empty_cache()

    draft = load_text_model(a.model, dtype=a.dtype, device_map={"": a.device_map})
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    for i, layer in enumerate(draft.model.layers):
        layer.mlp, _ = make_narrow(layer, a.width, "width", next(layer.parameters()).device,
                                   imp.get(f"l{i:02d}"))
    params = make_trainable(draft)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    draft.train()

    rng = random.Random(0)
    hist = []
    for step in range(a.steps):
        batch = [rng.choice(rows) for _ in range(a.batch_seqs)]
        maxlen = max(r["ids"].shape[0] for r in batch)
        ids = torch.zeros((len(batch), maxlen), dtype=torch.long)
        labels = torch.zeros((len(batch), maxlen), dtype=torch.long)
        top2 = torch.zeros((len(batch), maxlen), dtype=torch.long)
        tmarg = torch.zeros((len(batch), maxlen), dtype=torch.float)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
        for j, r in enumerate(batch):
            L = r["ids"].shape[0]
            ids[j, :L] = r["ids"]
            labels[j, :L] = r["top1"]
            top2[j, :L] = r["top2"]
            tmarg[j, :L] = r["margin"]
            mask[j, :L] = True
        ids, labels, top2 = ids.to(dev), labels.to(dev), top2.to(dev)
        tmarg, mask = tmarg.to(dev), mask.to(dev)
        # predict the token after each input position: shift labels by one
        x = ids
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = draft(input_ids=x, use_cache=False).logits.float()
        # logits at position t predict ids[t+1]; align with labels[t+1]
        lg = lg[:, :-1]
        lab = labels[:, 1:]
        to2 = top2[:, 1:]
        tmg = tmarg[:, 1:]
        m = mask[:, 1:]
        lg = lg[m]
        lab = lab[m]
        to2 = to2[m]
        tmg = tmg[m]
        ce = F.cross_entropy(lg, lab)
        za = lg.gather(-1, lab[:, None]).squeeze(-1)
        zb = lg.gather(-1, to2[:, None]).squeeze(-1)
        hinge = F.relu(tmg - (za - zb)).mean()
        loss = ce + a.margin_weight * hinge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0 or step == a.steps - 1:
            rec = {"step": step, "ce": float(ce), "hinge": float(hinge),
                   "loss": float(loss)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
    draft.eval()
    # Return the trained readouts to the model dtype for the standard evaluator.
    for layer in draft.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, NarrowMLP):
            mlp.down_w = torch.nn.Parameter(
                mlp.down_w.detach().to(torch.bfloat16), requires_grad=False)
    for p in params:
        p.requires_grad_(False)

    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        datasets.append((name, seqs, _load_cache(cpath)))
        print(f"set {name}: {len(seqs)} seqs", flush=True)
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
                     "calibration": "top1_ce_plus_margin_hinge",
                     "steps": a.steps, "lr": a.lr})
        stem = os.path.join(a.out, f"{name}_w{a.width:g}")
        np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id,
                 q_per_seq=_seq_q(cat["match"].astype(bool), seq_id, len(seqs))[0])
        with open(stem + ".json", "w") as f:
            json.dump(summ, f, indent=1)
        results.append(summ)
        print(f"  {name}: q={summ['q']:.4f} lb={summ['q_lb']:.4f}", flush=True)
        torch.cuda.empty_cache()
    with open(os.path.join(a.out, "calibration_history.json"), "w") as f:
        json.dump({"history": hist, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
