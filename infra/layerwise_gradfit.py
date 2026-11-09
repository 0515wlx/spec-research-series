"""Layer-wise gradient fine-tuning after structured FFN pruning.

This is the stronger recovery check: instead of only refitting the down
projection by ridge regression, we optimise all three narrow FFN matrices
(gate/up/down) for a few Adam steps on the original layer's input/output pairs.
The objective is still layer-wise MSE; evaluation runs the full model.
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
from infra.layerwise_reconstruct import collect_calibration, eval_model
from infra.model_loader import load_text_model, load_tokenizer, model_device


@torch.inference_mode()
def ridge_init(narrow, X, Y, lam_scale=1e-4, batch=4096):
    gw, uw = narrow.gate_w, narrow.up_w
    W0 = narrow.down_w.float()
    r = gw.shape[0]
    d_out = Y.shape[1]
    dev = gw.device
    C = torch.zeros(r, r, device=dev, dtype=torch.float32)
    B = torch.zeros(d_out, r, device=dev, dtype=torch.float32)
    for s in range(0, X.shape[0], batch):
        xb = X[s:s + batch].to(dev, torch.float32)
        yb = Y[s:s + batch].to(dev, torch.float32)
        a = F.silu(F.linear(xb, gw.float())) * F.linear(xb, uw.float())
        C += a.T @ a
        B += yb.T @ a
    lam = lam_scale * C.diagonal().mean().clamp_min(1e-8)
    W = torch.linalg.solve(C + lam * torch.eye(r, device=dev), B.T).T
    return W.to(torch.bfloat16)


def fit_layer(narrow, X, Y, steps=20, batch=256, lr=1e-6, val_tokens=1024):
    """Conservative Adam fit; keep the best held-out-MSE state.

    Returns (gw, uw, dw, mse_ridge, mse_best).
    """
    dev = narrow.gate_w.device
    gw = torch.nn.Parameter(narrow.gate_w.detach().float().clone())
    uw = torch.nn.Parameter(narrow.up_w.detach().float().clone())
    dw = torch.nn.Parameter(narrow.down_w.detach().float().clone())
    opt = torch.optim.Adam([gw, uw, dw], lr=lr, betas=(0.9, 0.95))
    nval = min(val_tokens, max(1, X.shape[0] // 5))
    Xf, Yf = X[:-nval], Y[:-nval]
    Xv, Yv = X[-nval:].to(dev, torch.float32), Y[-nval:].to(dev, torch.float32)

    @torch.no_grad()
    def val_mse():
        a = F.silu(F.linear(Xv, gw.float())) * F.linear(Xv, uw.float())
        return float(F.mse_loss(F.linear(a, dw.float()), Yv))

    mse0 = val_mse()
    best = (mse0, gw.detach().clone(), uw.detach().clone(), dw.detach().clone())
    g = torch.Generator().manual_seed(0)
    n = Xf.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        xb = Xf[idx].to(dev, torch.float32)
        yb = Yf[idx].to(dev, torch.float32)
        a = F.silu(F.linear(xb, gw)) * F.linear(xb, uw)
        loss = F.mse_loss(F.linear(a, dw), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gw, uw, dw], 0.1)
        opt.step()
        m = val_mse()
        if m < best[0]:
            best = (m, gw.detach().clone(), uw.detach().clone(), dw.detach().clone())
    return best[1], best[2], best[3], mse0, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--importance", default="out/importance/qwen3.5-9b_act.pt")
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--calib-tokens", type=int, default=16000)
    ap.add_argument("--fit-tokens", type=int, default=4096)
    ap.add_argument("--width", type=float, default=0.98)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-6)
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
    X, Y = collect_calibration(model, tok, rows, a.calib_tokens, 1024, dev)
    imp = torch.load(a.importance, map_location="cpu", weights_only=True)
    datasets = []
    for spec in a.sets:
        name, dpath, cpath = spec.split(":")
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        z = np.load(cpath)
        datasets.append((name, seqs, {k: z[k] for k in z.files}))
    orig = [layer.mlp for layer in model.model.layers]
    infos = {}
    t0 = time.time()
    for i, layer in enumerate(model.model.layers):
        narrow, info = make_narrow(layer, a.width, "width", dev,
                                   imp.get(f"l{i:02d}"), pad_to=0)
        W = ridge_init(narrow, X[i][: a.fit_tokens], Y[i][: a.fit_tokens])
        narrow = NarrowMLP(narrow.config, narrow.hidden_size, narrow.intermediate_size,
                           narrow.gate_w, narrow.up_w, W, narrow.act_fn).to(dev)
        gw, uw, dw, mse0, mse1 = fit_layer(narrow, X[i][: a.fit_tokens], Y[i][: a.fit_tokens],
                                            steps=a.steps, batch=a.batch, lr=a.lr)
        info.update({"rank": int(dw.shape[1]), "steps": a.steps, "lr": a.lr,
                     "mse_ridge": mse0, "mse_grad": mse1})
        infos[f"l{i:02d}"] = info
        layer.mlp = NarrowMLP(narrow.config, narrow.hidden_size, narrow.intermediate_size,
                              gw.to(torch.bfloat16), uw.to(torch.bfloat16), dw.to(torch.bfloat16),
                              narrow.act_fn).to(dev)
        del narrow
        torch.cuda.empty_cache()
    print(f"gradfit {a.width} done in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(a.out, f"gradfit_width{a.width:g}.json"), "w") as f:
        json.dump(infos, f)
    for name, seqs, cache in datasets:
        summ, cat, seq_len, seq_id = eval_model(model, seqs, cache, a.batch_size, dev)
        summ.update({"model": a.model, "width": a.width, "set": name,
                     "reconstruction": "layerwise_gradfit"})
        stem = os.path.join(a.out, f"{name}_w{a.width:g}")
        np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id)
        with open(stem + ".json", "w") as f:
            json.dump(summ, f, indent=1)
        print(f"  {name}: q={summ['q']:.4f} ci={summ['q_ci']}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
