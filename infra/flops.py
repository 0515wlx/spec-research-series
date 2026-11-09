"""Parameter and training-FLOP accounting for the Qwen3.5 text models.

The goal is a defensible MFU denominator: count only the text backbone that is
actually trained by ``AutoModelForCausalLM`` (vision tower and MTP head are
ignored), then apply the usual ``6N`` training-FLOP approximation plus the
full-attention quadratic term.

Param counts are read from the safetensors headers, so they are exact for the
checkpoint on disk rather than reconstructed from config (the config-based
estimate in ``infra.train_cost`` undercounts the tied-out lm_head and the
linear-attention projections).
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Optional

import torch

# Dense bf16 tensor-core peak of one A100-SXM4-80GB, no sparsity.
A100_BF16_PEAK_TFLOPS = 312.0


@dataclass
class ParamStats:
    total: int
    embed: int
    lm_head: int
    mlp: int
    full_attn: int
    linear_attn: int
    other: int

    @property
    def compute(self) -> int:
        """Params whose forward is a matmul.

        Excludes the embedding lookup.  For tied embeddings the lm_head is not
        a separate checkpoint tensor but its matmul still costs
        ``6 * vocab * hidden`` per token, so add the embedding back.
        """
        return self.total - self.embed + (0 if self.lm_head else self.embed)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("total", "embed", "lm_head", "mlp", "full_attn", "linear_attn", "other")}
        d["compute"] = self.compute
        d = {k: (v / 1e9 if isinstance(v, int) else v) for k, v in d.items()}
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _header_shapes(path: str) -> dict[str, tuple]:
    """Return {tensor_name: shape} for a safetensors checkpoint directory."""
    index_path = os.path.join(path, "model.safetensors.index.json")
    shapes: dict[str, tuple] = {}
    if os.path.exists(index_path):
        files = sorted(set(json.load(open(index_path))["weight_map"].values()))
    else:
        files = [f for f in ("model.safetensors", "pytorch_model.safetensors")
                 if os.path.exists(os.path.join(path, f))]
    for fname in files:
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shapes[name] = tuple(meta["shape"])
    return shapes


def param_stats(model_path: str) -> ParamStats:
    shapes = _header_shapes(model_path)
    total = embed = lm_head = mlp = full = linear = other = 0
    for name, shape in shapes.items():
        # Text-only CausalLM: drop vision tower and the unused MTP head.
        if "visual" in name or name.startswith("mtp."):
            continue
        n = 1
        for d in shape:
            n *= d
        total += n
        if "embed_tokens" in name:
            embed += n
        elif name.startswith("lm_head"):
            lm_head += n
        elif ".mlp." in name:
            mlp += n
        elif ".self_attn." in name:
            full += n
        elif ".linear_attn." in name:
            linear += n
        else:
            other += n
    return ParamStats(total, embed, lm_head, mlp, full, linear, other)


def full_attention_dims(text_config) -> tuple[int, int]:
    n_layers = text_config.num_hidden_layers
    layer_types = getattr(text_config, "layer_types", None) or ["full_attention"] * n_layers
    n_full = sum(str(t) == "full_attention" for t in layer_types)
    qk_dim = int(text_config.num_attention_heads) * int(getattr(
        text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads))
    return n_full, qk_dim


def flops_per_token(params: ParamStats, text_config, seq_len: int,
                    causal_attention: bool = True) -> float:
    """Training FLOPs per *padded* token (fwd + bwd), causal full attention.

    ``6N`` for the matmul weights, plus ~6 * s * d per full-attention layer for
    the QK^T and AV matmuls (forward 2, backward 4; causal masking halves the
    forward and the corresponding backward work).
    """
    base = 6.0 * params.compute
    n_full, qk_dim = full_attention_dims(text_config)
    # Forward QK^T + AV = 4*s*d per layer-token; x3 for fwd+bwd; x0.5 causal.
    attn = n_full * (4.0 * seq_len * qk_dim) * 3.0 * (0.5 if causal_attention else 1.0)
    return base + attn


def mfu(nominal_flops: float, seconds: float, peak_tflops: float = A100_BF16_PEAK_TFLOPS) -> dict:
    if seconds <= 0:
        return {}
    tflops = nominal_flops / seconds / 1e12
    return {
        "tflops": tflops,
        "mfu": tflops / peak_tflops,
        "peak_tflops": peak_tflops,
    }


def _config(model_path: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    return getattr(cfg, "text_config", cfg)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/models/Qwen3.5-27B"
    stats = param_stats(path)
    cfg = _config(path)
    print(json.dumps(stats.as_dict(), indent=1))
    print(f"compute params: {stats.compute/1e9:.3f}B")
    for s in (1024, 2048):
        f = flops_per_token(stats, cfg, s)
        print(f"seq={s}: {f/1e9:.2f} GFLOP/token, "
              f"6N share={(6*stats.compute/f):.1%}")
