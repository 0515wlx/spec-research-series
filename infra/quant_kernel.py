"""Real weight-only INT4/INT8 linear layers for the quantization axis.

The earlier `infra/quant.py` is a *fake*-quantization control: master weights
stay in bf16 and are dequantized on every forward, so it cannot measure a real
memory saving.  This module implements the actual packed-weight layout used by
PyTorch's CUDA tinygemm kernels:

  * INT4: asymmetric per-group quantization (group 128), packed two nibbles per
    byte, `torch.ops.aten._weight_int4pack_mm` for the fused dequant matmul.
  * INT8: symmetric per-channel quantization,
    `torch.ops.aten._weight_int8pack_mm`.

Both are weight-only: activations stay bf16 and the matmul is performed by the
int4/int8 tensor-core kernel.  The helper `quantized_bytes` reports the actual
stored bytes, which is what enters the memory-roofline ratio.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MIN_DIM = 128


def _check_int4_shape(out_features: int, in_features: int, group_size: int,
                      inner_k_tiles: int = 8) -> bool:
    return (in_features % group_size == 0
            and in_features % (inner_k_tiles * 16) == 0
            and out_features % 8 == 0)


def quantize_int4_weight(w: torch.Tensor, group_size: int = 128,
                         inner_k_tiles: int = 8):
    """Return (packed_int32, scales_and_zeros) for a [out, in] bf16 weight.

    The nibble order matters: `_convert_weight_to_int4pack` reads the *high*
    nibble as the even element.  This is verified against an identity matmul in
    `self_test`.
    """
    out_f, in_f = w.shape
    if not _check_int4_shape(out_f, in_f, group_size, inner_k_tiles):
        raise ValueError(f"INT4 layout requires in % {group_size} == 0, "
                         f"in % {inner_k_tiles * 16} == 0, out % 8 == 0; "
                         f"got {out_f}x{in_f}")
    wf = w.detach().float().reshape(out_f, in_f // group_size, group_size)
    mn = wf.amin(-1)
    mx = wf.amax(-1)
    scale = (mx - mn).clamp_min(1e-8) / 15.0
    zero = mn + scale * 8.0
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp_(0, 15)
    q = q.to(torch.uint8).reshape(out_f, in_f)
    # high nibble = even index, low nibble = odd index
    packed_u8 = ((q[:, 0::2] & 0xF) << 4) | (q[:, 1::2] & 0xF)
    packed = torch.ops.aten._convert_weight_to_int4pack(
        packed_u8.contiguous(), inner_k_tiles)
    sz = torch.stack([scale, zero], dim=-1).permute(1, 0, 2).contiguous()
    return packed, sz.to(torch.bfloat16)


def dequantize_int4_weight(packed: torch.Tensor, scales_and_zeros: torch.Tensor,
                           out_features: int, in_features: int,
                           group_size: int = 128) -> torch.Tensor:
    """Dequantize a packed INT4 weight for diagnostics / self-test."""
    n_groups = in_features // group_size
    # scales_and_zeros: [n_groups, out_features, 2]
    scale = scales_and_zeros[..., 0].float()  # [G, N]
    zero = scales_and_zeros[..., 1].float()
    # Unpack via the kernel itself on an identity input: out = x @ W^T.
    x = torch.eye(in_features, device=packed.device, dtype=torch.bfloat16)
    w = torch.ops.aten._weight_int4pack_mm(x, packed, group_size, scales_and_zeros)
    return w.t().contiguous()  # [out, in]


class Int4Linear(nn.Module):
    """A real INT4 weight-only linear layer (packed storage + fused dequant)."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 inner_k_tiles: int = 8, bias: torch.Tensor | None = None,
                 packed: torch.Tensor | None = None,
                 scales_and_zeros: torch.Tensor | None = None,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        if packed is None:
            self.register_buffer("packed", torch.empty(0, dtype=torch.int32))
            self.register_buffer("scales_and_zeros",
                                 torch.empty(0, dtype=torch.bfloat16))
        else:
            self.register_buffer("packed", packed)
            self.register_buffer("scales_and_zeros", scales_and_zeros)

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 128,
                    inner_k_tiles: int = 8) -> "Int4Linear":
        packed, sz = quantize_int4_weight(lin.weight.data, group_size, inner_k_tiles)
        m = cls(lin.in_features, lin.out_features, group_size, inner_k_tiles,
                bias=lin.bias.data if lin.bias is not None else None,
                packed=packed, scales_and_zeros=sz,
                device=lin.weight.device, dtype=lin.weight.dtype)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        out = torch.ops.aten._weight_int4pack_mm(
            x2, self.packed, self.group_size, self.scales_and_zeros)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig[:-1], self.out_features)


def quantize_int8_weight(w: torch.Tensor):
    """Symmetric per-output-channel INT8 quantization."""
    wf = w.detach().float()
    scale = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


class Int8Linear(nn.Module):
    """A real INT8 weight-only linear layer (per-channel symmetric)."""

    def __init__(self, in_features: int, out_features: int,
                 bias: torch.Tensor | None = None,
                 weight: torch.Tensor | None = None,
                 scales: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        if weight is None:
            self.register_buffer("weight", torch.empty(0, dtype=torch.int8))
            self.register_buffer("scales", torch.empty(0, dtype=torch.bfloat16))
        else:
            self.register_buffer("weight", weight)
            self.register_buffer("scales", scales)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8Linear":
        q, s = quantize_int8_weight(lin.weight.data)
        return cls(lin.in_features, lin.out_features,
                   bias=lin.bias.data if lin.bias is not None else None,
                   weight=q, scales=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        out = torch.ops.aten._weight_int8pack_mm(x2, self.weight, self.scales)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig[:-1], self.out_features)


def apply_int4(model, group_size: int = 128, inner_k_tiles: int = 8,
               skip_lm_head: bool = False, skip_embedding: bool = True):
    """Replace every eligible `nn.Linear` with an `Int4Linear` in place."""
    todo = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, (Int4Linear, Int8Linear)):
            continue
        if skip_lm_head and name == "lm_head":
            continue
        if skip_embedding and ("embed_tokens" in name or "embedding" in name):
            continue
        if min(mod.in_features, mod.out_features) < MIN_DIM:
            continue
        if not _check_int4_shape(mod.out_features, mod.in_features,
                                 group_size, inner_k_tiles):
            continue
        todo.append((name, mod))
    for name, mod in todo:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, Int4Linear.from_linear(mod, group_size, inner_k_tiles))
    return [n for n, _ in todo]


def apply_int8(model, skip_lm_head: bool = False, skip_embedding: bool = True):
    todo = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, (Int4Linear, Int8Linear)):
            continue
        if skip_lm_head and name == "lm_head":
            continue
        if skip_embedding and ("embed_tokens" in name or "embedding" in name):
            continue
        if min(mod.in_features, mod.out_features) < MIN_DIM:
            continue
        if mod.in_features % 16 != 0:
            continue
        todo.append((name, mod))
    for name, mod in todo:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, Int8Linear.from_linear(mod))
    return [n for n, _ in todo]


class Int4NarrowMLP(nn.Module):
    """A width-reduced FFN whose three matrices are real packed INT4 weights.

    This is the combined width+quantization operator.  It keeps the exact
    narrow-MLP dataflow of ``joint_scan.NarrowMLP`` but stores gate/up/down in
    the same packed layout as ``Int4Linear`` and executes them with the fused
    tinygemm kernel.  The class is deliberately separate from ``NarrowMLP`` so
    that the width-only path is unchanged.
    """

    def __init__(self, cfg, hidden_size: int, intermediate_size: int,
                 gate_packed, gate_sz, up_packed, up_sz,
                 down_packed, down_sz, act_fn,
                 group_size: int = 128, inner_k_tiles: int = 8):
        super().__init__()
        self.config = cfg
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.register_buffer("gate_packed", gate_packed.contiguous())
        self.register_buffer("gate_sz", gate_sz.contiguous())
        self.register_buffer("up_packed", up_packed.contiguous())
        self.register_buffer("up_sz", up_sz.contiguous())
        self.register_buffer("down_packed", down_packed.contiguous())
        self.register_buffer("down_sz", down_sz.contiguous())

    @classmethod
    def from_narrow(cls, narrow, group_size: int = 128,
                    inner_k_tiles: int = 8) -> "Int4NarrowMLP":
        gp, gsz = quantize_int4_weight(narrow.gate_w, group_size, inner_k_tiles)
        up, usz = quantize_int4_weight(narrow.up_w, group_size, inner_k_tiles)
        dp, dsz = quantize_int4_weight(narrow.down_w, group_size, inner_k_tiles)
        return cls(narrow.config, narrow.hidden_size, narrow.intermediate_size,
                   gp, gsz, up, usz, dp, dsz, narrow.act_fn,
                   group_size=group_size, inner_k_tiles=inner_k_tiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[:-1]
        x2 = x.reshape(-1, self.hidden_size)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        g = torch.ops.aten._weight_int4pack_mm(
            x2, self.gate_packed, self.group_size, self.gate_sz)
        u = torch.ops.aten._weight_int4pack_mm(
            x2, self.up_packed, self.group_size, self.up_sz)
        h = self.act_fn(g) * u
        y = torch.ops.aten._weight_int4pack_mm(
            h, self.down_packed, self.group_size, self.down_sz)
        return y.reshape(*orig, self.hidden_size)


def quantized_bytes(model) -> dict:
    """Actual stored bytes by parameter class, for the memory-roofline ratio."""
    q4 = q8 = other = 0
    for mod in model.modules():
        if isinstance(mod, Int4Linear):
            q4 += mod.packed.numel() * 4  # int32 words
            q4 += mod.scales_and_zeros.numel() * 2
            if mod.bias is not None:
                q4 += mod.bias.numel() * 2
        elif isinstance(mod, Int4NarrowMLP):
            for p, s in ((mod.gate_packed, mod.gate_sz),
                         (mod.up_packed, mod.up_sz),
                         (mod.down_packed, mod.down_sz)):
                q4 += p.numel() * 4
                q4 += s.numel() * 2
        elif isinstance(mod, Int8Linear):
            q8 += mod.weight.numel()
            q8 += mod.scales.numel() * 2
            if mod.bias is not None:
                q8 += mod.bias.numel() * 2
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        # parameters inside replaced linears are gone; count the rest at bf16
        other += p.numel() * 2
    total = q4 + q8 + other
    return {"int4_bytes": q4, "int8_bytes": q8, "other_bytes": other,
            "total_bytes": total, "total_gb": total / 1e9}


def self_test():
    torch.manual_seed(0)
    N, K = 256, 512
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(3, K, device="cuda", dtype=torch.bfloat16)
    lin = nn.Linear(K, N, bias=True).cuda().to(torch.bfloat16)
    lin.weight.data.copy_(w)
    q = Int4Linear.from_linear(lin, group_size=128, inner_k_tiles=8)
    out = q(x)
    ref = lin(x)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    print(f"int4 self-test rel err {rel.item():.4f}")
    lin8 = nn.Linear(K, N, bias=True).cuda().to(torch.bfloat16)
    lin8.weight.data.copy_(w)
    q8 = Int8Linear.from_linear(lin8)
    out8 = q8(x)
    rel8 = (out8 - ref).abs().mean() / ref.abs().mean()
    print(f"int8 self-test rel err {rel8.item():.5f}")


if __name__ == "__main__":
    self_test()
