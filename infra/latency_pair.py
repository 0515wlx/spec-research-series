"""Paired batch-1 latency: dense target vs compressed draft on one GPU.

Both models are resident on the same A100 so that clock state, allocator and
kernel selection are shared.  We alternate target/draft forwards and report the
median ratio.  This avoids the cross-device timing artifacts of the first
version.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F

from infra.model_loader import load_text_model, model_device
from infra.joint_scan import make_narrow


class FactoredMLP(torch.nn.Module):
    def __init__(self, cfg, inter, ag, bg, au, bu, ad, bd, act_fn):
        super().__init__()
        self.config = cfg
        self.hidden_size = ag.shape[0]
        self.intermediate_size = inter
        self.act_fn = act_fn
        self.register_buffer("ag", ag.contiguous())
        self.register_buffer("bg", bg.contiguous())
        self.register_buffer("au", au.contiguous())
        self.register_buffer("bu", bu.contiguous())
        self.register_buffer("ad", ad.contiguous())
        self.register_buffer("bd", bd.contiguous())

    def forward(self, x):
        g = F.linear(F.linear(x, self.bg), self.ag)
        u = F.linear(F.linear(x, self.bu), self.au)
        return F.linear(F.linear(self.act_fn(g) * u, self.bd), self.ad)


def factorize(model, factors_dir, frac, device):
    from infra import svd_compress
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
def time_once(model, ids):
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    model(input_ids=ids, use_cache=False)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--mode", default="width", choices=["width", "svd"])
    ap.add_argument("--factors", default="out/svd/qwen3.5-9b")
    ap.add_argument("--importance", default=None)
    ap.add_argument("--fracs", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = torch.randint(100, 200000, (1, a.len), dtype=torch.long).to(a.device)
    target = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    for _ in range(5):
        target(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    base_times = [time_once(target, ids) for _ in range(a.repeats)]
    base = statistics.median(base_times)
    print(f"dense median {base:.4f}s", flush=True)

    imp = torch.load(a.importance, map_location="cpu", weights_only=True) if a.importance else None
    rows = []
    for f in [float(x) for x in a.fracs.split(",")]:
        draft = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
        dev = model_device(draft)
        if a.mode == "svd":
            draft = factorize(draft, a.factors, f, dev)
        else:
            for i, layer in enumerate(draft.model.layers):
                layer.mlp, _ = make_narrow(layer, f, "width", dev,
                                           imp.get(f"l{i:02d}") if imp else None,
                                           pad_to=128)
        for _ in range(5):
            draft(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        dtimes = [time_once(draft, ids) for _ in range(a.repeats)]
        d = statistics.median(dtimes)
        # interleaved check
        inter = []
        for _ in range(a.repeats):
            inter.append(time_once(draft, ids) / time_once(target, ids))
        rows.append({"frac": f, "target_median_s": base, "draft_median_s": d,
                     "rho": d / base, "rho_interleaved_median": statistics.median(inter),
                     "draft_times": dtimes, "target_times": base_times})
        print(f"frac={f}: draft={d:.4f}s target={base:.4f}s rho={d/base:.3f} "
              f"inter={statistics.median(inter):.3f}", flush=True)
        del draft
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
