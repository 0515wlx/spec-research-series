"""Weight-only fake quantization (RTN) for the Qwen3.5 dense drafts.

Adapted from the old `code/quant.py`; kept here so the Route A pipeline is self-contained.
Master weights stay in bf16; forward uses the dequantized weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_DIM = 128


def fake_quant_weight(w, bits=4, group=128, sym=True):
    if bits >= 16:
        return w
    out_f, in_f = w.shape
    wf = w.float()
    g = min(group, in_f)
    pad = (-in_f) % g
    if pad:
        wf = F.pad(wf, (0, pad))
    wg = wf.reshape(out_f, -1, g)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        q = (wg / scale).round().clamp(-qmax - 1, qmax)
        dq = q * scale
    else:
        qmin, qmax = 0, 2 ** bits - 1
        lo = wg.amin(-1, keepdim=True)
        hi = wg.amax(-1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-8) / qmax
        q = ((wg - lo) / scale).round().clamp(qmin, qmax)
        dq = q * scale + lo
    dq = dq.reshape(out_f, -1)
    if pad:
        dq = dq[:, :in_f]
    return dq.to(w.dtype)


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, bits=4, group=128):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.group = group
        self.wq_cache = None

    @classmethod
    def from_linear(cls, lin, bits=4, group=128):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                bits=bits, group=group)
        m = m.to(lin.weight.device, lin.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(lin.weight)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        return m

    def dequantized_weight(self):
        return fake_quant_weight(self.weight, self.bits, self.group)

    def refresh_cache(self):
        with torch.no_grad():
            self.wq_cache = self.dequantized_weight().detach()

    def forward(self, x):
        wq = self.wq_cache if self.wq_cache is not None else self.dequantized_weight()
        w = self.weight + (wq - self.weight).detach()
        return F.linear(x, w, self.bias)


def apply_fake_quant(model, bits=4, group=128, skip_lm_head=False, lm_head_bits=None,
                     min_dim=MIN_DIM):
    todo = []
    for name, module in model.named_modules():
        if isinstance(module, QLinear) or not isinstance(module, nn.Linear):
            continue
        if min(module.in_features, module.out_features) < min_dim:
            continue
        if name == "lm_head" and skip_lm_head:
            continue
        b = lm_head_bits if (name == "lm_head" and lm_head_bits is not None) else bits
        todo.append((name, module, b))
    for name, module, b in todo:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, QLinear.from_linear(module, bits=b, group=group))
    return [(n, b) for n, _, b in todo]


def refresh_fake_quant(model):
    for m in model.modules():
        if isinstance(m, QLinear):
            m.refresh_cache()


def quant_stats(model):
    qbits = qparams = other = 0
    for _, m in model.named_modules():
        if isinstance(m, QLinear):
            n = m.weight.numel()
            qparams += n
            qbits += n * m.bits
        elif isinstance(m, nn.Linear):
            other += m.weight.numel() * 16
    total = sum(p.numel() for p in model.parameters())
    qbytes = qbits // 8
    obytes = (total - qparams) * 2
    return {"quant_params_b": qparams / 1e9, "quant_bytes_gb": qbytes / 1e9,
            "other_bytes_gb": obytes / 1e9, "total_bytes_gb": (qbytes + obytes) / 1e9,
            "total_params_b": total / 1e9}
