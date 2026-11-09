"""Evaluate the target-native Qwen3.5 MTP head as a trained draft.

The Qwen3.5 checkpoint ships an MTP head (`mtp.*`) that transformers ignores when
loading the text CausalLM.  We load it manually and use the NEXTN formulation

    h'_{i,1} = M_1 [ RMSNorm(Emb(x_{i+1})) ; RMSNorm(h_i) ]
    p(x_{i+2} | x_{<=i+1}) = LMHead(h'_{i,1})

(note the checkpoint's concatenation order: embedding first, hidden second).
At decoding position p, the draft uses the target hidden at p-2 and the
embedding of the current token p-1, and predicts p.  This is a one-step-ahead
draft, directly comparable with a post-hoc next-token draft.

The hidden state is the post-final-norm target state (`hidden_states[-1]`), which
is the convention used by the vLLM-MLX Qwen3.5 MTP patch; the pre-norm variant is
also available for diagnosis but is not used in the paper.

Output schema matches the post-hoc scan NPZs (`match`, `margin_log`,
`d_margin_log`, `seq_id`) so the margin analysis scripts can consume it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from safetensors import safe_open
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm)

from infra import data
from infra.model_loader import load_text_model, model_device
from infra.model_registry import resolve


class MTPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = copy.deepcopy(config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, hidden, emb, position_embeddings, attention_mask):
        # checkpoint order: embedding first, hidden second
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb),
                               self.pre_fc_norm_hidden(hidden)], dim=-1))
        for layer in self.layers:
            x = layer(x, position_embeddings=position_embeddings,
                      attention_mask=attention_mask)
        return self.norm(x)


def load_mtp(model_path, model, dtype=torch.bfloat16):
    index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))
    wm = index["weight_map"]
    keys = [k for k in wm if k.startswith("mtp.")]
    state = {}
    for shard in sorted(set(wm[k] for k in keys)):
        with safe_open(os.path.join(model_path, shard), framework="pt") as f:
            for k in keys:
                if wm[k] == shard:
                    state[k[len("mtp."):]] = f.get_tensor(k)
    mtp = MTPHead(model.config).to(dtype=dtype, device=model_device(model))
    missing, unexpected = mtp.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MTP load mismatch: missing={missing} unexpected={unexpected}")
    mtp.eval()
    return mtp


@torch.inference_mode()
def _seq_arrays(model, mtp, seq, device, hidden_mode="last", pos_offset=1):
    full = seq["input_ids"] + seq["output_ids"]
    x = torch.tensor([full], dtype=torch.long, device=device)
    out = model(input_ids=x, use_cache=False, output_hidden_states=True)
    H = out.hidden_states[-1] if hidden_mode == "last" else out.hidden_states[-2]
    L = x.shape[1]
    if L < 3:
        return None
    h = H[:, :-2, :]
    emb = model.model.embed_tokens(x[:, 1:-1])
    pos = (torch.arange(L - 2, device=device) + pos_offset).unsqueeze(0)
    pos_emb = model.model.rotary_emb(h, pos)
    mask = create_causal_mask(config=model.config, inputs_embeds=h,
                              attention_mask=None, past_key_values=None,
                              position_ids=None)
    z = mtp(h, emb, pos_emb, mask)
    logits = model.lm_head(z).float()
    tgt_h = H[:, 1:-1, :]
    tgt_logits = model.lm_head(tgt_h).float()
    d_arg = logits.argmax(-1)
    t_arg = tgt_logits.argmax(-1)
    match = (d_arg == t_arg)
    t_top = tgt_logits.topk(2, dim=-1).indices
    z1 = tgt_logits.gather(-1, t_top[:, :, 0:1]).squeeze(-1)
    z2 = tgt_logits.gather(-1, t_top[:, :, 1:2]).squeeze(-1)
    d1 = logits.gather(-1, t_top[:, :, 0:1]).squeeze(-1)
    d2 = logits.gather(-1, t_top[:, :, 1:2]).squeeze(-1)
    m = (z1 - z2).cpu().numpy().ravel().astype(np.float32)
    dm = ((d1 - d2) - (z1 - z2)).cpu().numpy().ravel().astype(np.float32)
    return {
        "match": match.cpu().numpy().ravel(),
        "margin_log": m,
        "d_margin_log": dm,
        "abs_d_margin_log": np.abs(dm),
        "target_ids": x[:, 2:].cpu().numpy().ravel(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name:data.jsonl entries")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hidden-mode", default="last", choices=["last", "pre"])
    ap.add_argument("--pos-offset", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = resolve(a.model)
    model = load_text_model(a.model, dtype="bfloat16", device_map={"": a.device})
    mtp = load_mtp(path, model, dtype=model.dtype)
    dev = model_device(model)
    rows = []
    for spec in a.sets:
        name, dpath = spec.split(":", 1)
        seqs = data.load_target_sequences(dpath, limit=a.limit)
        print(f"set {name}: {len(seqs)} seqs", flush=True)
        all_out = []
        for i, s in enumerate(seqs):
            r = _seq_arrays(model, mtp, s, dev, a.hidden_mode, a.pos_offset)
            if r is not None:
                all_out.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(seqs)}", flush=True)
            torch.cuda.empty_cache()
        if not all_out:
            continue
        seq_len = np.array([len(o["match"]) for o in all_out], dtype=np.int32)
        seq_id = np.repeat(np.arange(len(all_out), dtype=np.int32), seq_len)
        cat = {k: np.concatenate([o[k] for o in all_out], axis=0)
               for k in ("match", "margin_log", "d_margin_log",
                         "abs_d_margin_log", "target_ids")}
        match = cat["match"].astype(bool)
        m = cat["margin_log"].astype(np.float64)
        d = cat["d_margin_log"].astype(np.float64)
        q = 1.0 - float(match.mean())
        q_lb = float(((m + d) < 0).mean())
        low = m <= np.quantile(m, 0.2)
        row = {
            "set": name, "model": a.model, "mtp": True,
            "hidden_mode": a.hidden_mode, "pos_offset": a.pos_offset,
            "seqs": len(all_out), "tokens": int(len(match)),
            "q": q, "q_lb_top2": q_lb,
            "E_margin": float(m.mean()), "median_m": float(np.median(m)),
            "E_abs_d_margin_log": float(np.abs(d).mean()),
            "low_mass": float((m < 0.5).mean()),
            "low_quintile_q": float(1.0 - match[low].mean()),
            "low_quintile_abs_d": float(np.abs(d)[low].mean()),
        }
        stem = os.path.join(a.out, name)
        os.makedirs(a.out, exist_ok=True)
        np.savez(stem + ".npz", **cat, gen_len=seq_len, seq_id=seq_id)
        with open(stem + ".json", "w") as f:
            json.dump(row, f, indent=1)
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {a.out}/summary.json")


if __name__ == "__main__":
    main()
