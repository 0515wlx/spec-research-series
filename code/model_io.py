"""Load/save draft checkpoints, dispatching on meta['arch'] (dense or moe)."""
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from draft import build_dense_model, save_dense


def load_model(ckpt_dir, dtype=torch.bfloat16, device="cpu"):
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("arch") == "moe":
        from moe_draft import build_moe_model
        from safetensors.torch import load_file
        model, _ = build_moe_model(meta["num_experts"], dtype=dtype, device=device)
        sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("missing", missing[:5], len(missing))
        if unexpected:
            print("unexpected", unexpected[:5], len(unexpected))
        model.eval()
        return model, meta
    from draft import load_dense
    return load_dense(ckpt_dir, dtype=dtype, device=device)


def save_model(model, ckpt_dir, meta):
    save_dense(model, ckpt_dir, meta)
