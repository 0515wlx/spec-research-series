"""Stream FineWeb-Edu from ModelScope and build a token-level OOD continuation set."""
from __future__ import annotations

import argparse
import json
import os

from infra import data
from infra.model_loader import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-chunks", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--mode", choices=["ood", "train"], default="ood",
                    help="ood: half prompt/half continuation; train: full chunk as output (self-supervised)")
    ap.add_argument("--tokenizer", default="qwen3.5-35b-a3b")
    ap.add_argument("--out", default="data/processed/fineweb_target.jsonl")
    ap.add_argument("--prompts-out", default="data/processed/fineweb.jsonl")
    a = ap.parse_args()

    from modelscope.msdatasets import MsDataset

    tok = load_tokenizer(a.tokenizer)
    ds = MsDataset.load(
        a.dataset, subset_name=a.subset, split=a.split,
        use_streaming=True, trust_remote_code=True,
    )
    half = a.seq_len // 2
    rows, prompts = [], []
    scanned = 0
    for r in ds:
        scanned += 1
        text = r.get("text") or ""
        if len(text) < 200:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < a.seq_len:
            continue
        for i in range(len(ids) // a.seq_len):
            chunk = ids[i * a.seq_len:(i + 1) * a.seq_len]
            if len(chunk) < a.seq_len:
                break
            cid = f"fineweb-{len(rows)}"
            if a.mode == "train":
                rows.append({
                    "id": cid, "domain": "fineweb", "prompt": "",
                    "input_ids": [], "output_ids": chunk,
                    "text": tok.decode(chunk),
                    "meta": {"source": a.dataset, "subset": a.subset, "seq_len": a.seq_len,
                             "mode": "self_supervised"},
                })
            else:
                rows.append({
                    "id": cid,
                    "domain": "fineweb",
                    "prompt": tok.decode(chunk[:half]),
                    "input_ids": chunk[:half],
                    "output_ids": chunk[half:],
                    "text": tok.decode(chunk[half:]),
                    "meta": {"source": a.dataset, "subset": a.subset, "seq_len": a.seq_len},
                })
                prompts.append({
                    "id": cid, "domain": "fineweb",
                    "prompt": tok.decode(chunk[:half]), "reference": None,
                    "meta": {"source": a.dataset, "subset": a.subset},
                })
            if len(rows) >= a.n_chunks:
                break
        if len(rows) >= a.n_chunks:
            break

    data.write_jsonl(a.out, rows)
    if a.mode == "ood":
        data.write_jsonl(a.prompts_out, prompts)
        print(f"scanned {scanned} docs; wrote {len(rows)} OOD chunks -> {a.out}; prompts -> {a.prompts_out}")
    else:
        print(f"scanned {scanned} docs; wrote {len(rows)} train chunks -> {a.out}")


if __name__ == "__main__":
    main()
