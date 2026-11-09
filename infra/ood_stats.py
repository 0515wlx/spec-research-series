"""Operational OOD diagnostics: target NLL/perplexity and calibration n-gram overlap.

Qwen3.5 pretraining data are not public, so we cannot certify contamination.  We
report two operational checks that support the raw-vs-generated distinction used in
the paper:

* target NLL / perplexity on the scored continuation of each set;
* the fraction of evaluation word 8-grams that also occur in the calibration corpus.

The second check is a lower bound on overlap with *our* calibration data, not with the
target's pretraining data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from infra import data
from infra.model_loader import load_text_model, model_device


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


@torch.inference_mode()
def nll_set(model, seqs, device, batch_size=4):
    order = sorted(range(len(seqs)),
                   key=lambda i: len(seqs[i]["input_ids"]) + len(seqs[i]["output_ids"]))
    total_nll = 0.0
    total_tok = 0
    per_seq = []
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        fulls = [seqs[k]["input_ids"] + seqs[k]["output_ids"] for k in idxs]
        xs = [f[:-1] for f in fulls]
        maxlen = max(len(x) for x in xs)
        x = torch.zeros((len(xs), maxlen), dtype=torch.long)
        att = torch.zeros((len(xs), maxlen), dtype=torch.long)
        for j, xx in enumerate(xs):
            x[j, :len(xx)] = torch.tensor(xx, dtype=torch.long)
            att[j, :len(xx)] = 1
        x, att = x.to(device), att.to(device)
        logits = model(input_ids=x, attention_mask=att, use_cache=False).logits.float()
        for j, k in enumerate(idxs):
            s = seqs[k]
            start = len(s["input_ids"]) - 1
            L = len(s["output_ids"])
            lg = logits[j, start:start + L]
            tgt = torch.tensor(s["output_ids"], dtype=torch.long, device=device)
            nll = F.cross_entropy(lg, tgt, reduction="sum").item()
            total_nll += nll
            total_tok += L
            per_seq.append(nll / max(L, 1))
    mean_nll = total_nll / max(total_tok, 1)
    return {
        "tokens": int(total_tok),
        "mean_nll": float(mean_nll),
        "perplexity": float(np.exp(mean_nll)),
        "per_seq_nll_mean": float(np.mean(per_seq)) if per_seq else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name:data.jsonl entries")
    ap.add_argument("--calib", default="data/processed/distill_controlled.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    calib_ngrams = set()
    if os.path.exists(a.calib):
        with open(a.calib) as f:
            for line in f:
                r = json.loads(line)
                text = r.get("text") or r.get("prompt") or ""
                calib_ngrams |= _ngrams(_words(text), a.ngram)
        print(f"calibration {a.ngram}-grams: {len(calib_ngrams)}", flush=True)

    model = load_text_model(a.model, dtype=a.dtype, device_map=a.device_map)
    dev = model_device(model)
    rows = []
    for spec in a.sets:
        name, path = spec.split(":", 1)
        seqs = data.load_target_sequences(path, limit=a.limit)
        print(f"set {name}: {len(seqs)} seqs", flush=True)
        m = nll_set(model, seqs, dev, a.batch_size)
        eval_ngrams = set()
        for s in seqs:
            text = s.get("text") or s.get("prompt") or ""
            eval_ngrams |= _ngrams(_words(text), a.ngram)
        overlap = len(eval_ngrams & calib_ngrams) / max(len(eval_ngrams), 1)
        m.update({"set": name, "data": path, "seqs": len(seqs),
                  "eval_ngrams": len(eval_ngrams),
                  "calib_ngram_overlap": float(overlap)})
        rows.append(m)
        print(f"  NLL={m['mean_nll']:.4f} ppl={m['perplexity']:.2f} "
              f"overlap={overlap:.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
