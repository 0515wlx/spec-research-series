"""Triton fused SwiGLU MLP microbenchmark for the kernel-ceiling check.

The paper's main kernel argument is that the MLP path is a small fraction of
the batch-1 draft pass, so even a free MLP cannot rescue the FFN-width drafter.
This script adds a concrete prototype: it fuses gate/up + SiLU into one GEMV
kernel and the down projection into a second GEMV kernel, then compares them
with the padded PyTorch narrow MLP used in the latency measurements.  It
reports achieved GB/s and the fraction of the A100 HBM roofline (2.0 TB/s).

This is a microbenchmark of one layer, not an end-to-end serving engine.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import triton
import triton.language as tl

from infra.joint_scan import make_narrow
from infra.model_loader import load_text_model


@triton.jit
def _gate_up_silu(x_ptr, wg_ptr, wu_ptr, out_ptr, R, D,
                  BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R
    acc_g = tl.zeros((BLOCK_R,), tl.float32)
    acc_u = tl.zeros((BLOCK_R,), tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wg = tl.load(wg_ptr + offs_r[:, None] * D + offs_k[None, :],
                     mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        wu = tl.load(wu_ptr + offs_r[:, None] * D + offs_k[None, :],
                     mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc_g += tl.sum(wg * x[None, :], axis=1)
        acc_u += tl.sum(wu * x[None, :], axis=1)
    a = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(out_ptr + offs_r, a, mask=mask_r)


@triton.jit
def _down(a_ptr, wd_ptr, y_ptr, D, R,
          BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for r0 in range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        a = tl.load(a_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        wd = tl.load(wd_ptr + offs_d[:, None] * R + offs_r[None, :],
                     mask=mask_d[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wd * a[None, :], axis=1)
    tl.store(y_ptr + offs_d, acc, mask=mask_d)


@torch.no_grad()
def timed(fn, repeats=100):
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / 1000.0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--widths", default="0.5,0.98")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = load_text_model(a.model, dtype="bfloat16", device_map={"": "cuda:0"})
    layer = model.model.layers[0]
    mlp = layer.mlp
    wg = mlp.gate_proj.weight.detach().float()
    wu = mlp.up_proj.weight.detach().float()
    wd = mlp.down_proj.weight.detach().float()
    D = wg.shape[1]
    rows = []
    for width in [float(x) for x in a.widths.split(",")]:
        R = max(1, int(round(width * wg.shape[0])))
        # activation-aware subset is irrelevant for a timing microbenchmark;
        # use the first R channels and the corresponding down columns
        gw = wg[:R].contiguous().to("cuda:0", torch.bfloat16)
        uw = wu[:R].contiguous().to("cuda:0", torch.bfloat16)
        dw = wd[:, :R].contiguous().to("cuda:0", torch.bfloat16)
        x = torch.randn(D, device="cuda:0", dtype=torch.bfloat16)
        a_vec = torch.empty(R, device="cuda:0", dtype=torch.bfloat16)
        y = torch.empty(D, device="cuda:0", dtype=torch.bfloat16)

        def torch_mlp():
            g = torch.nn.functional.linear(x, gw)
            u = torch.nn.functional.linear(x, uw)
            return torch.nn.functional.linear(torch.nn.functional.silu(g) * u, dw)

        def triton_mlp():
            _gate_up_silu[(triton.cdiv(R, 64),)](x, gw, uw, a_vec, R, D,
                                                 BLOCK_R=64, BLOCK_K=256)
            _down[(triton.cdiv(D, 64),)](a_vec, dw, y, D, R,
                                         BLOCK_D=64, BLOCK_R=256)

        torch_mlp(); triton_mlp(); torch.cuda.synchronize()
        t_torch = timed(torch_mlp, a.repeats)
        t_triton = timed(triton_mlp, a.repeats)
        bytes_moved = 2 * (R * D + R * D + D * R)  # bf16 weights, three matrices
        gbps = bytes_moved / t_triton / 1e9
        rows.append({"width": width, "R": R, "D": D,
                     "torch_s": t_torch, "triton_s": t_triton,
                     "speedup": t_torch / t_triton,
                     "triton_GBps": gbps,
                     "HBM_fraction": gbps / 2000.0})
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
