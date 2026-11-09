"""Load Qwen3.5 text backbones from multimodal checkpoints.

The local Qwen3.5 / Qwen3.6 checkpoints are ``*ForConditionalGeneration`` models with a
vision tower.  For speculative-decoding research we only need the text LM.  Passing the
text config to ``AutoModelForCausalLM.from_pretrained`` makes transformers instantiate
``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` and skip the vision weights.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from infra import model_registry


def _dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def load_tokenizer(model_or_path: str, trust_remote_code: bool = True):
    path = model_registry.resolve(model_or_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_text_model(
    model_or_path: str,
    dtype: str = "bfloat16",
    device_map: Any = "auto",
    max_memory: Optional[dict] = None,
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load only the language model from a (possibly multimodal) Qwen3.5 checkpoint."""
    path = model_registry.resolve(model_or_path)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    text_cfg = getattr(cfg, "text_config", cfg)
    kwargs = dict(
        config=text_cfg,
        dtype=_dtype(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


@torch.inference_mode()
def forward_logits(model, input_ids, attention_mask=None, **kwargs):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
    return out.logits


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def describe(model_or_path: str) -> dict:
    path = model_registry.resolve(model_or_path)
    cfg = AutoConfig.from_pretrained(path)
    tc = getattr(cfg, "text_config", cfg)
    return {
        "path": path,
        "model_type": getattr(cfg, "model_type", None),
        "text_model_type": getattr(tc, "model_type", None),
        "hidden_size": getattr(tc, "hidden_size", None),
        "num_hidden_layers": getattr(tc, "num_hidden_layers", None),
        "vocab_size": getattr(tc, "vocab_size", None),
        "num_experts": getattr(tc, "num_experts", None),
        "num_experts_per_tok": getattr(tc, "num_experts_per_tok", None),
        "tie_word_embeddings": getattr(tc, "tie_word_embeddings", None),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m infra.model_loader <model_key_or_path>")
        raise SystemExit(1)
    print(json.dumps(describe(sys.argv[1]), indent=1))
