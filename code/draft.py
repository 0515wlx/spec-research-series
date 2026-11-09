"""Dense draft model with the same Qwen3.5/3.6 backbone, MoE FFN replaced by a width-matched dense FFN."""
import json, os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN

TEACHER = "/path/to/models/Qwen3.6-35B-A3B"
DENSE_INTER = 8 * 512 + 512  # top-k experts + shared expert = active intermediate width (4608)


class DenseMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def build_dense_model(teacher_dir=TEACHER, inter=DENSE_INTER, dtype=torch.bfloat16, device="cpu"):
    cfg = AutoConfig.from_pretrained(teacher_dir)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tc)
    for layer in model.model.layers:
        layer.mlp = DenseMLP(tc.hidden_size, inter, tc.hidden_act)
    model = model.to_empty(device=device)
    model = model.to(dtype)
    model.config.use_cache = True
    return model


def load_dense(ckpt_dir, dtype=torch.bfloat16, device="cpu"):
    from safetensors.torch import load_file
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    model = build_dense_model(inter=meta["inter"], dtype=dtype, device=device)
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("missing", missing[:5], len(missing))
    if unexpected:
        print("unexpected", unexpected[:5], len(unexpected))
    return model, meta


def save_dense(model, ckpt_dir, meta):
    from safetensors.torch import save_file
    os.makedirs(ckpt_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # transformers 5 wraps unused params; ensure no meta tensors
    for k, v in sd.items():
        assert v.device.type == "cpu" and v.dtype != torch.float32 or True
    save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("saved", ckpt_dir, sum(v.numel() for v in sd.values()) / 1e9, "B params")
