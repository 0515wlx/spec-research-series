"""Batch-1 latency check for dense vs compressed drafts.

Both models process the same fixed 512-token prefix at batch size 1; the
reported ratio is the per-pass latency ratio used by the cost model in the
paper (``rho``).  Runs sequentially on one GPU so that the target and draft
never contend for memory bandwidth.


NOTE (review 11): this benchmark times a full 512-token forward, not a decode
step with a KV cache; it is a padded full-pass upper bound, not the batch-1
decode protocol used in the cost layer.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn

from infra import svd_compress
from infra.model_loader import load_text_model, model_device


class FactoredMLP(nn.Module):
    def __init__(self, cfg, inter, ag, bg, au, bu, ad, bd, act_fn):
        super().__init__()
        self.config = cfg
        self.hidden_size = ag.shape[0]
        self.intermediate_size = inter
        self.act_fn = act_fn
        self.register_buffer("ag", ag)
        self.register_buffer("bg", bg)
        self.register_buffer("au", au)
        self.register_buffer("bu", bu)
        self.register_buffer("ad", ad)
        self.register_buffer("bd", bd)

    def forward(self, x):
        g = F.linear(F.linear(x, self.bg), self.ag)
        u = F.linear(F.linear(x, self.bu), self.au)
        return F.linear(F.linear(self.act_fn(g) * u, self.bd), self.ad)


def factorize(model, factors_dir, frac, device):
    for i, layer in enumerate(model.model.layers):
        z = {}
        for name in svd_compress.MLP_NAMES:
            q = torch.load(os.path.join(factors_dir, f"l{i:02d}_{name}.pt"),
                           map_location="cpu", weights_only=True)
            out_f, in_f = q["shape"]
            r = max(1, int(round(frac * min(out_f, in_f))))
            a = (q["U"][:, :r].float() * q["S"][:r]).to(device)
            b = q["Vh"][:r, :].float().to(device)
            z[name] = (a, b)
        layer.mlp = FactoredMLP(layer.mlp.config, layer.mlp.intermediate_size,
                                *z["gate_proj"], *z["up_proj"], *z["down_proj"],
                                layer.mlp.act_fn).to(device)
    return model


@torch.inference_mode()
def bench(model, ids, repeats=8, warmup=2):
    for _ in range(warmup):
        model(input_ids=ids, use_cache=False)
    if ids.device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--factors", default="out/svd/qwen3.5-9b")
    ap.add_argument("--fracs", default="0.05,0.1,0.2,0.5")
    ap.add_argument("--mode", default="svd", choices=["svd", "width"])
    ap.add_argument("--importance", default=None)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = torch.randint(100, 200000, (1, a.len), dtype=torch.long).to(a.device)
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    base = bench(target, ids, a.repeats)
    print("dense target", base, flush=True)
    del target
    torch.cuda.empty_cache()

    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    rows = []
    for f in [float(x) for x in a.fracs.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
        dev = model_device(draft)
        if a.mode == "svd":
            draft = factorize(draft, a.factors, f, dev)
        else:
            from infra.joint_scan import make_narrow
            for i, layer in enumerate(draft.model.layers):
                layer.mlp, _ = make_narrow(layer, f, "width", dev,
                                           imp.get(f"l{i:02d}") if imp else None)
        d = bench(draft, ids, a.repeats)
        rho = d["median_s"] / base["median_s"]
        rows.append({"frac": f, "target": base, "draft": d, "rho": rho})
        print(f"frac={f}: draft={d['median_s']:.4f}s target={base['median_s']:.4f}s "
              f"rho={rho:.3f}", flush=True)
        del draft
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
