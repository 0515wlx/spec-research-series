"""Common helpers for the WikiText distribution-shift probe."""
import glob, json, os, sys
import numpy as np
import torch

TEACHER = "/path/to/models/Qwen3.6-35B-A3B"
WIKI_DEMO = "/path/to/sbh/infinitevl_repro/repo/data/wiki_demo.txt"
WIKI_PARQUET = "~/ .cache/modelscope/datasets/Salesforce--wikitext/snapshots/master/wikitext-2-raw-v1/test-00000-of-00001.parquet"
OUT = "/path/to/spec-research/out"


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEACHER)


def load_text():
    """Prefer the real WikiText-2 raw test set downloaded from ModelScope."""
    if os.path.exists(WIKI_PARQUET):
        import pandas as pd
        df = pd.read_parquet(WIKI_PARQUET)
        return "\n".join(str(x) for x in df["text"].tolist())
    return open(WIKI_DEMO, encoding="utf-8", errors="ignore").read()


def load_chunks(n=100, seq_len=1024):
    """Tokenize WikiText and return [n, seq_len] token ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER)
    text = load_text()
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    need = n * seq_len
    if len(ids) < need:
        raise RuntimeError(f"only {len(ids)} tokens, need {need}")
    return ids[:need].view(n, seq_len)


def metric_arrays(logp, targets, top_idx, top_logp):
    """logp [B,T,V], targets [B,T], teacher top_idx/top_logp [B,T,K]."""
    B, T, V = logp.shape
    K = top_idx.shape[-1]
    nll = -logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    pred = logp.argmax(-1)
    match = pred == targets
    dtop = logp.topk(5, dim=-1).indices
    t1 = top_idx[:, :, 0]
    in5 = (dtop == t1.unsqueeze(-1)).any(-1)
    t1logp = logp.gather(2, t1.unsqueeze(-1))
    rank = (logp > t1logp).sum(-1)
    p = top_logp.exp()
    q = logp.gather(2, top_idx).exp()
    p0 = (1 - p.sum(-1)).clamp(min=1e-8)
    q0 = (1 - q.sum(-1)).clamp(min=1e-8)
    kl_td = (p * (top_logp - q.clamp(min=1e-12).log())).sum(-1) + p0 * (p0.log() - q0.log())
    kl_dt = (q * (q.clamp(min=1e-12).log() - top_logp)).sum(-1) + q0 * (q0.log() - p0.log())
    tv = 0.5 * ((p - q).abs().sum(-1) + (p0 - q0).abs())
    m = 0.5 * (p + q); m0 = 0.5 * (p0 + q0)
    ml = m.clamp(min=1e-12).log()
    js = 0.5 * ((p * (top_logp - ml)).sum(-1) + p0 * (p0.log() - m0.log())) + \
         0.5 * ((q * (q.clamp(min=1e-12).log() - ml)).sum(-1) + q0 * (q0.log() - m0.log()))
    ent = -(logp.exp() * logp).sum(-1)
    return {"nll": nll, "match": match, "in5": in5, "rank": rank,
            "kl_td": kl_td, "kl_dt": kl_dt, "tv": tv, "js": js,
            "ent": ent, "p0": p0, "q0": q0, "pred": pred}
