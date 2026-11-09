"""Layer-wise reconstruction after structured FFN pruning.

For each layer we keep the activation-aware top-r channels, then refit the
down projection by ridge regression on a calibration set:

    A = act(X W_g^T) * (X W_u^T)          (N x r)
    W_d' = argmin_W ||Y - A W^T||_F^2 + lam ||W||_F^2
         = Y^T A (A^T A + lam I)^-1.

This is the standard "prune then reconstruct" recovery step.  The calibration
inputs/outputs are collected from the *original* model (teacher forcing), so the
fit is per-layer and does not compound across layers during fitting.  Evaluation
then runs the full reconstructed model, where errors do compound.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.accept_eval import eval_batch
from infra.joint_scan import NarrowMLP, make_narrow
from infra.model_loader import load_text_model, load_tokenizer, model_device


@torch.inference_mode()
def collect_calibration(model, tok, rows, max_tokens, max_len, device):
    layers = model.model.layers
    xs = [[] for _ in layers]
    ys = [[] for _ in layers]
    handles = []

    def make_hook(i):
        def hook(mod, inp, out):
            xs[i].append(inp[0].detach().to(torch.bfloat16).cpu())
            ys[i].append(out.detach().to(torch.bfloat16).cpu())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(i)))
    n = 0
    for r in rows:
        text = r.get("text") or r.get("prompt") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len)["input_ids"].to(device)
        if ids.shape[1] < 16:
            continue
        model(input_ids=ids, use_cache=False)
        n += ids.shape[1]
        if n >= max_tokens:
            break
    for h in handles:
        h.remove()
    X = [torch.cat([z.reshape(-1, z.shape[-1]) for z in v], 0) for v in xs]
    Y = [torch.cat([z.reshape(-1, z.shape[-1]) for z in v], 0) for v in ys]
    # keep a deterministic subset of tokens
    N = min(max_tokens, X[0].shape[0])
    X = [x[:N] for x in X]
    Y = [y[:N] for y in Y]
    print(f"calibration tokens={N}", flush=True)
    return X, Y


@torch.inference_mode()
def reconstruct_layer(narrow, X, Y, batch=4096):
    """Residual ridge reconstruction around the pruned down projection.

    We fit ``W = W0 + D`` where ``W0`` is the pruned down matrix and ``D`` is
    shrunk toward zero.  This makes the pruned layer a feasible candidate, so a
    validation-selected lambda can only improve the held-out reconstruction MSE.
    """
    gw, uw = narrow.gate_w, narrow.up_w
    W0 = narrow.down_w.float()
    r = gw.shape[0]
    d_out = Y.shape[1]
    dev = gw.device
    nval = min(4000, max(1, X.shape[0] // 4))
    Xf, Yf = X[:-nval], Y[:-nval]
    Xv, Yv = X[-nval:].to(dev, torch.float32), Y[-nval:].to(dev, torch.float32)
    C = torch.zeros(r, r, device=dev, dtype=torch.float32)
    Bres = torch.zeros(d_out, r, device=dev, dtype=torch.float32)
    for s in range(0, Xf.shape[0], batch):
        xb = Xf[s:s + batch].to(dev, torch.float32)
        yb = Yf[s:s + batch].to(dev, torch.float32)
        a = F.silu(F.linear(xb, gw.float())) * F.linear(xb, uw.float())
        C += a.T @ a
        Bres += (yb - a @ W0.T).T @ a
    Av = F.silu(F.linear(Xv, gw.float())) * F.linear(Xv, uw.float())
    denom = (Yv.float() ** 2).sum().clamp_min(1e-8)
    y0 = Av @ W0.T
    mse_before = float(((Yv - y0) ** 2).sum() / denom)
    mean_diag = C.diagonal().mean().clamp_min(1e-8)
    best = None
    for scale in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        lam = scale * mean_diag
        try:
            Dt = torch.linalg.solve(C + lam * torch.eye(r, device=dev), Bres.T)
        except Exception:
            continue
        W = W0 + Dt.T
        mse = float(((Yv - Av @ W.T) ** 2).sum() / denom)
        if best is None or mse < best[0]:
            best = (mse, lam, W)
    mse_after, lam, W = best
    return W.to(torch.bfloat16), float(lam), mse_before, mse_after


def eval_model(model, seqs, cache, batch_size, device):
    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    out = [None] * len(seqs)
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        o = eval_batch(model, seqs, cache, idxs, device)
        for k, v in zip(idxs, o):
            out[k] = v
    keys = ["match", "tv", "margin", "rho", "margin_log", "d_margin_log",
            "abs_d_margin_log", "err_k", "rank", "ent_t"]
    cat = {k: np.concatenate([o[k] for o in out], axis=0)
           for k in keys if out[0].get(k) is not None}
    seq_len = np.array([len(o["match"]) for o in out], dtype=np.int32)
    seq_id = np.repeat(np.arange(len(out), dtype=np.int32), seq_len)
    match = cat["match"].astype(bool)
    q = 1.0 - float(match.mean())
    # cluster bootstrap
    n_seq = len(out)
    per = np.zeros(n_seq); cnt = np.zeros(n_seq)
    np.add.at(per, seq_id, (~match).astype(np.float64))
    np.add.at(cnt, seq_id, 1.0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_seq, size=(2000, n_seq))
    boot = (per[idx].sum(1)) / np.maximum(cnt[idx].sum(1), 1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    m = cat["margin_log"]
    dm = cat["d_margin_log"]
    flip = (m + dm) < 0
    return {"q": q, "q_ci": ci, "E_TV": float(cat["tv"].mean()),
            "q_lb_top2": float(flip.mean()), "tokens": int(len(match)),
            "seqs": n_seq}, cat, seq_len, seq_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--calib-tokens", type=int, default=16000)
    ap.add_argument("--calib-len", type=int, default=1024)
    ap.add_argument("--widths", default="0.98,0.99")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tok = load_tokenizer(a.model)
    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    rows = [json.loads(line) for line in open(a.calib)]
    X, Y = collect_calibration(model, tok, rows, a.calib_tokens, a.calib_len, dev)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        z = np.load(cpath)
        datasets.append((name, seqs, {k: z[k] for k in z.files}))
        print(f"set {name}: {len(seqs)}", flush=True)

    orig = [layer.mlp for layer in model.model.layers]
    for width_s in a.widths.split(","):
        width = float(width_s)
        t0 = time.time()
        infos = {}
        for i, layer in enumerate(model.model.layers):
            narrow, info = make_narrow(layer, width, "width", dev,
                                       imp.get(f"l{i:02d}"), pad_to=0)
            W, lam, mse0, mse1 = reconstruct_layer(narrow, X[i], Y[i])
            info.update({"lambda": lam, "rank": int(W.shape[1]),
                         "mse_before": mse0, "mse_after": mse1})
            infos[f"l{i:02d}"] = info
            layer.mlp = NarrowMLP(narrow.config, narrow.hidden_size, narrow.intermediate_size,
                                  narrow.gate_w, narrow.up_w, W, narrow.act_fn).to(dev)
            del narrow
            torch.cuda.empty_cache()
        print(f"width={width} reconstructed in {time.time()-t0:.0f}s", flush=True)
        with open(os.path.join(a.out, f"reconstruct_width{width:g}.json"), "w") as f:
            json.dump(infos, f)
        for name, seqs, cache in datasets:
            summ, cat, seq_len, seq_id = eval_model(model, seqs, cache, a.batch_size, dev)
            summ.update({"model": a.model, "width": width, "set": name,
                         "reconstruction": "layerwise_down_ridge"})
            stem = os.path.join(a.out, f"{name}_w{width:g}")
            np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id)
            with open(stem + ".json", "w") as f:
                json.dump(summ, f, indent=1)
            print(f"  {name}: q={summ['q']:.4f} ci={summ['q_ci']} "
                  f"lb={summ['q_lb_top2']:.4f}", flush=True)
            torch.cuda.empty_cache()
        # restore original MLP blocks for the next width
        for layer, mlp in zip(model.model.layers, orig):
            layer.mlp = mlp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
