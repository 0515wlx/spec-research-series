"""Synthetic high-predictability raw-text corpus for a screen positive control.

The main raw corpora are natural prose.  To test whether the margin-mass screen
can ever predict a window (q < q*), we build a distribution with deliberately
high predictability: templated JSON records, repetitive code lines, and
key-value logs.  The scored continuation is still raw text (not a target
response); it is a boundary-mapping positive control, not a deployment corpus.

Run the normal pipeline afterwards:
  python -m infra.cache_target --data data/processed/template_target.jsonl ...
  python -m infra.joint_scan --model qwen3.5-9b --mode width ...
"""
from __future__ import annotations

import argparse
import json
import os

from infra.model_loader import load_tokenizer


def records(n: int = 200) -> str:
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    cities = ["Paris", "Berlin", "Tokyo", "Nairobi", "Lima", "Oslo"]
    out = []
    for i in range(n):
        out.append(json.dumps({
            "id": i,
            "name": names[i % len(names)],
            "age": 20 + (i % 40),
            "city": cities[i % len(cities)],
            "status": "active" if i % 3 else "inactive",
            "score": round((i % 100) / 100.0, 2),
        }))
    return "\n".join(out)


def code(n: int = 200) -> str:
    out = []
    for i in range(n):
        out.append(f"def f_{i}(x):\n    return x + {i}\n")
    return "\n".join(out)


def logs(n: int = 200) -> str:
    levels = ["INFO", "WARN", "ERROR"]
    services = ["api", "worker", "scheduler"]
    messages = ["request completed", "cache hit", "retry scheduled",
                "connection closed", "task finished"]
    out = []
    for i in range(n):
        out.append(
            f"timestamp=2026-01-01T00:00:{i % 60:02d}Z level={levels[i % 3]} "
            f"service={services[i % 3]} message=\"{messages[i % 5]}\"")
    return "\n".join(out)


BLOCKS = [records, code, logs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--tokenizer", default="qwen3.5-9b")
    ap.add_argument("--out", default="data/processed/template_target.jsonl")
    ap.add_argument("--prompts-out", default="data/processed/template.jsonl")
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    half = a.seq_len // 2
    rows, prompts = [], []
    for i in range(a.n_chunks):
        block = BLOCKS[i % len(BLOCKS)]()
        # repeat the block until it is long enough, then truncate by tokens
        text = block
        while len(tok(text, add_special_tokens=False)["input_ids"]) < a.seq_len:
            text = text + "\n" + block
        ids = tok(text, add_special_tokens=False)["input_ids"][:a.seq_len]
        cid = f"template-{i}"
        rows.append({
            "id": cid, "domain": "template",
            "prompt": tok.decode(ids[:half]),
            "input_ids": ids[:half], "output_ids": ids[half:],
            "text": tok.decode(ids[half:]),
            "meta": {"source": "synthetic", "kind": ["records", "code", "logs"][i % 3],
                     "seq_len": a.seq_len},
        })
        prompts.append({
            "id": cid, "domain": "template",
            "prompt": tok.decode(ids[:half]), "reference": None,
            "meta": {"source": "synthetic", "kind": ["records", "code", "logs"][i % 3]},
        })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(a.prompts_out, "w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} chunks -> {a.out}; prompts -> {a.prompts_out}")


if __name__ == "__main__":
    main()
