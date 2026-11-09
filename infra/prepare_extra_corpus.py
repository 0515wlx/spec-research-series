"""Build extra raw-text OOD corpora from ModelScope (C4, Gutenberg, code).

Outputs the same format as ``infra/prepare_fineweb.py``:
  * ``{name}_target.jsonl``: 100 chunks of 1024 tokens (512 prompt / 512 scored)
  * ``{name}.jsonl``: the prompt half, for target generation
"""
from __future__ import annotations

import argparse
import json
import os

from infra import data
from infra.model_loader import load_tokenizer


CORPORA = {
    "c4": dict(dataset="allenai/c4", split="train", text_key="text", subset=None),
    "gutenberg": dict(dataset="common-pile/project_gutenberg", split="train",
                      text_key="text", subset=None),
    "code": dict(dataset="bigcode/the-stack-smol", split="train",
                 text_key="content", subset=None, lang="Python"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--n-chunks", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-chunks-per-doc", type=int, default=5)
    ap.add_argument("--tokenizer", default="qwen3.5-35b-a3b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts-out", default=None)
    a = ap.parse_args()

    from modelscope.msdatasets import MsDataset

    spec = CORPORA[a.corpus]
    tok = load_tokenizer(a.tokenizer)
    kw = dict(split=spec["split"], use_streaming=True, trust_remote_code=True)
    if spec.get("subset"):
        kw["subset_name"] = spec["subset"]
    ds = MsDataset.load(spec["dataset"], **kw)

    half = a.seq_len // 2
    rows, prompts = [], []
    scanned = 0
    for r in ds:
        scanned += 1
        if spec.get("lang") and r.get("lang") not in (spec["lang"], None):
            continue
        text = r.get(spec["text_key"]) or ""
        if len(text) < 500:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < a.seq_len:
            continue
        for i in range(min(len(ids) // a.seq_len, a.max_chunks_per_doc)):
            chunk = ids[i * a.seq_len:(i + 1) * a.seq_len]
            if len(chunk) < a.seq_len:
                break
            cid = f"{a.corpus}-{len(rows)}"
            rows.append({
                "id": cid, "domain": a.corpus,
                "prompt": tok.decode(chunk[:half]),
                "input_ids": chunk[:half], "output_ids": chunk[half:],
                "text": tok.decode(chunk[half:]),
                "meta": {"source": spec["dataset"], "split": spec["split"],
                         "seq_len": a.seq_len},
            })
            prompts.append({
                "id": cid, "domain": a.corpus,
                "prompt": tok.decode(chunk[:half]), "reference": None,
                "meta": {"source": spec["dataset"], "split": spec["split"]},
            })
            if len(rows) >= a.n_chunks:
                break
        if len(rows) >= a.n_chunks:
            break

    out = a.out or f"data/processed/{a.corpus}_target.jsonl"
    prompts_out = a.prompts_out or f"data/processed/{a.corpus}.jsonl"
    data.write_jsonl(out, rows)
    data.write_jsonl(prompts_out, prompts)
    print(f"scanned {scanned} docs; wrote {len(rows)} chunks -> {out}; prompts -> {prompts_out}")


if __name__ == "__main__":
    main()
