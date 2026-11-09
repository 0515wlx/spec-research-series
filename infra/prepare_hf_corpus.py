"""Build additional raw-text / raw-code corpora from Hugging Face streaming.

The output matches ``prepare_extra_corpus.py``: ``{corpus}_target.jsonl`` has
``input_ids``/``output_ids`` chunks of ``seq_len`` tokens (half prompt, half
scored continuation), and ``{corpus}.jsonl`` has the prompt half only.

Streaming keeps the disk footprint small (the cluster has limited free space).
Set ``HF_HOME``/``HF_ENDPOINT`` externally if the default cache is not usable.
"""
from __future__ import annotations

import argparse
import json
import os

from infra import data
from infra.model_loader import load_tokenizer


CORPORA = {
    # multilingual encyclopedic prose
    "wikipedia_zh": dict(dataset="wikimedia/wikipedia", name="20231101.zh",
                         split="train", text_key="text", streaming=True),
    "wikipedia_de": dict(dataset="wikimedia/wikipedia", name="20231101.de",
                         split="train", text_key="text", streaming=True),
    "wikipedia_fr": dict(dataset="wikimedia/wikipedia", name="20231101.fr",
                         split="train", text_key="text", streaming=True),
    "wikipedia_ja": dict(dataset="wikimedia/wikipedia", name="20231101.ja",
                         split="train", text_key="text", streaming=True),
    # multilingual web prose (FineWeb-2)
    "fineweb2_zh": dict(dataset="HuggingFaceFW/fineweb-2", name="cmn_Hani",
                        split="train", text_key="text", streaming=True),
    # code
    "code_raw": dict(dataset="codeparrot/codeparrot-clean", split="train",
                     text_key="content", streaming=True),
    # long-form books
    "pg19": dict(dataset="emozilla/pg19", split="test", text_key="text",
                 streaming=True),
    # scientific prose
    "arxiv": dict(dataset="ccdv/arxiv-summarization", split="train",
                  text_key="article", streaming=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--n-chunks", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-chunks-per-doc", type=int, default=5)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--tokenizer", default="qwen3.5-9b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts-out", default=None)
    a = ap.parse_args()

    from datasets import load_dataset

    spec = CORPORA[a.corpus]
    tok = load_tokenizer(a.tokenizer)
    kw = dict(split=spec["split"], streaming=spec.get("streaming", True))
    if spec.get("name"):
        kw["name"] = spec["name"]
    ds = load_dataset(spec["dataset"], **kw)

    half = a.seq_len // 2
    rows, prompts = [], []
    scanned = 0
    seen_text = set()
    for r in ds:
        scanned += 1
        text = r.get(spec["text_key"]) or ""
        if len(text) < a.min_chars:
            continue
        # cheap dedup on the first 200 characters
        key = text[:200]
        if key in seen_text:
            continue
        seen_text.add(key)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < a.seq_len:
            continue
        n_take = min(len(ids) // a.seq_len, a.max_chunks_per_doc)
        for i in range(n_take):
            chunk = ids[i * a.seq_len:(i + 1) * a.seq_len]
            if len(chunk) < a.seq_len:
                break
            cid = f"{a.corpus}-{len(rows)}"
            rows.append({
                "id": cid, "domain": a.corpus,
                "prompt": tok.decode(chunk[:half]),
                "input_ids": chunk[:half], "output_ids": chunk[half:],
                "text": tok.decode(chunk[half:]),
                "meta": {"source": spec["dataset"], "name": spec.get("name"),
                         "split": spec["split"], "seq_len": a.seq_len},
            })
            prompts.append({
                "id": cid, "domain": a.corpus,
                "prompt": tok.decode(chunk[:half]), "reference": None,
                "meta": {"source": spec["dataset"], "name": spec.get("name")},
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
