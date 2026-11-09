"""Structured / spectral compression of Qwen3.5 FFN blocks.

Two operators are supported:

``svd``
    Replace each MLP matrix ``W`` (``gate_proj``, ``up_proj``, ``down_proj``) by its
    best rank-``r`` approximation ``W_r = U_r S_r V_r^T``.  The reconstructed matrix
    has the original dense shape, so the standard HF model can run it without a
    custom kernel.  The *cost* of the compressed layer is the factored cost
    ``r (d + h)`` instead of ``d h``; see :func:`ffn_cost_ratio`.

``width``
    Structured neuron pruning: keep the ``r`` intermediate channels with the
    largest combined gate/up/down weight norm.  This is the literal
    "reduce the FFN internal dimension" operator.

The SVD factors are written one file per (layer, matrix), so several rank-scan
workers can share them without keeping the full 13 GB factor set in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from infra import model_registry
from infra.model_loader import load_text_model

MLP_NAMES = ("gate_proj", "up_proj", "down_proj")


def _resolve(model: str) -> str:
    return model_registry.resolve(model)


def _layers(model):
    return model.model.layers


def _mlp(layer):
    return layer.mlp


def _text_cfg(model_or_path: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(_resolve(model_or_path))
    return getattr(cfg, "text_config", cfg)


@dataclass
class CompressMeta:
    model: str
    hidden: int
    inter: int
    layers: int
    max_rank: int
    factors_dir: str


def prepare(model: str, out_dir: str, max_rank_frac: float = 0.75,
            device: str = "cuda:0", dtype: str = "bf16") -> CompressMeta:
    """Compute truncated SVD factors for every FFN matrix.

    ``max_rank_frac`` is relative to ``min(out_features, in_features)``.  We keep
    the top ``R = ceil(max_rank_frac * min(...))`` triplets plus the *full*
    singular-value spectrum (needed for the discarded-energy diagnostic).
    """
    os.makedirs(out_dir, exist_ok=True)
    m = load_text_model(model, dtype="float32", device_map=device)
    layers = _layers(m)
    cfg = _text_cfg(model)
    hidden = int(cfg.hidden_size)
    inter = int(cfg.intermediate_size)
    t0 = time.time()
    spectra = {}
    for i, layer in enumerate(layers):
        for name in MLP_NAMES:
            w = getattr(_mlp(layer), name).weight.detach().float()
            out_f, in_f = w.shape
            rmax = max(1, int(round(max_rank_frac * min(out_f, in_f))))
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            u = u[:, :rmax].to(torch.bfloat16).cpu()
            vh = vh[:rmax, :].to(torch.bfloat16).cpu()
            s_all = s.cpu()
            torch.save({"U": u, "S": s_all, "Vh": vh,
                        "shape": (out_f, in_f), "rmax": rmax},
                       os.path.join(out_dir, f"l{i:02d}_{name}.pt"))
            spectra[f"l{i:02d}_{name}"] = {
                "shape": [out_f, in_f],
                "sigma": [float(x) for x in s_all[: min(64, len(s_all))]],
                "energy": float((s_all ** 2).sum()),
                "rmax": rmax,
            }
            del w, u, s, vh
            if i % 8 == 0 and name == "gate_proj":
                print(f"  svd layer {i}/{len(layers)}  {time.time()-t0:.0f}s", flush=True)
    meta = {
        "model": model, "hidden": hidden, "inter": inter,
        "layers": len(layers), "max_rank_frac": max_rank_frac,
        "factors_dir": out_dir, "spectra": spectra,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"saved SVD factors -> {out_dir}  ({time.time()-t0:.0f}s)", flush=True)
    return CompressMeta(model, hidden, inter, len(layers),
                        int(round(max_rank_frac * min(hidden, inter))), out_dir)


def _rank_for(shape: tuple[int, int], frac: float) -> int:
    return max(1, int(round(frac * min(shape))))


def apply_svd(model, factors_dir: str, frac: float) -> dict:
    """Apply rank-``frac * min(shape)`` truncation in place.  Returns per-layer info."""
    info = {}
    for i, layer in enumerate(_layers(model)):
        for name in MLP_NAMES:
            path = os.path.join(factors_dir, f"l{i:02d}_{name}.pt")
            z = torch.load(path, map_location="cpu", weights_only=True)
            u, s, vh = z["U"], z["S"], z["Vh"]
            out_f, in_f = z["shape"]
            r = _rank_for((out_f, in_f), frac)
            if r > u.shape[1]:
                raise ValueError(f"{path}: rank {r} > stored {u.shape[1]}")
            w = getattr(_mlp(layer), name)
            dev = w.weight.device
            wr = (u[:, :r].to(dev, torch.float32) * s[:r].to(dev)) @ vh[:r, :].to(dev, torch.float32)
            w.weight.data.copy_(wr.to(w.weight.dtype))
            info[f"l{i:02d}_{name}"] = {
                "rank": r, "shape": [out_f, in_f],
                "discarded_energy": float((s[r:] ** 2).sum()),
                "total_energy": float((s ** 2).sum()),
            }
            del z, u, s, vh, wr
            torch.cuda.empty_cache()
    return info


@torch.no_grad()
def apply_width(model, frac: float) -> dict:
    """Keep the top-``frac * inter`` intermediate channels by weight norm."""
    info = {}
    for i, layer in enumerate(_layers(model)):
        mlp = _mlp(layer)
        wg = mlp.gate_proj.weight
        wu = mlp.up_proj.weight
        wd = mlp.down_proj.weight
        inter = wg.shape[0]
        r = max(1, int(round(frac * inter)))
        score = (wg.float().pow(2).sum(1) + wu.float().pow(2).sum(1)
                 + wd.float().pow(2).sum(0))
        idx = torch.topk(score, r).indices.sort().values
        # zero-out dropped channels (keeps original shapes; cost model uses r)
        mask_out = torch.zeros(inter, 1, device=wg.device, dtype=wg.dtype)
        mask_out[idx] = 1
        mask_in = mask_out.T
        mlp.gate_proj.weight.data.mul_(mask_out)
        mlp.up_proj.weight.data.mul_(mask_out)
        mlp.down_proj.weight.data.mul_(mask_in)
        info[f"l{i:02d}_width"] = {"rank": r, "inter": inter,
                                   "kept_energy": float(score[idx].sum()),
                                   "total_energy": float(score.sum())}
        del score, mask_out, mask_in
    return info


def ffn_cost_ratio(frac: float, hidden: int, inter: int, mode: str = "svd") -> float:
    """FLOP ratio of one compressed FFN block relative to the dense FFN block.

    Dense has ``3 * hidden * inter`` multiply-adds.  ``svd`` factors each of the
    three matrices into ``r x (hidden + inter)``; ``width`` keeps ``r`` channels.
    """
    r = frac * min(hidden, inter)
    if mode == "svd":
        return 3 * r * (hidden + inter) / (3 * hidden * inter)
    if mode == "width":
        return frac
    raise ValueError(mode)


def model_cost_ratio(model: str, frac: float, mode: str = "svd") -> dict:
    """Total per-token FFN/attention/embedding FLOP ratio relative to the dense model.

    The FFN block is compressed; attention and embeddings are untouched.  We use
    parameter-count shares as a FLOP proxy (all matmuls are linear in parameters
    at fixed sequence length); the paper also reports an attention-aware check.
    """
    from infra.flops import param_stats
    s = param_stats(_resolve(model))
    mlp_share = s.mlp / s.compute
    ffn_ratio = ffn_cost_ratio(frac, int(_text_cfg(model).hidden_size),
                               int(_text_cfg(model).intermediate_size), mode)
    total = (1.0 - mlp_share) + mlp_share * ffn_ratio
    return {"model": model, "mode": mode, "frac": frac,
            "mlp_share": mlp_share, "ffn_ratio": ffn_ratio,
            "total_ratio": total, "params_total_b": s.total / 1e9,
            "params_compute_b": s.compute / 1e9}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rank-frac", type=float, default=0.75)
    p.add_argument("--device", default="cuda:0")
    q = sub.add_parser("cost")
    q.add_argument("--model", required=True)
    q.add_argument("--fracs", default="0.025,0.05,0.1,0.15,0.2,0.3,0.5")
    q.add_argument("--mode", default="svd")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.model, a.out, a.max_rank_frac, a.device)
    else:
        rows = [model_cost_ratio(a.model, float(x), a.mode) for x in a.fracs.split(",")]
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
