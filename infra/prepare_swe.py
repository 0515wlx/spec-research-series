"""Build an agentic issue -> patch corpus from SWE-bench-Lite.

The deployment prefix is an agent-style instruction plus the GitHub issue and
its hints; the scored continuation is the human patch.  This is a static
single-turn proxy for an agentic/RAG prefix distribution (tool traces are not
included because the public Lite split does not ship repository snapshots).

Example:
  HF_ENDPOINT=https://hf-mirror.com python -m infra.prepare_swe \
    --n 100 --prefix-tokens 1024 --cont-tokens 512 \
    --out data/processed/swe_lite_target.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from infra import data
from infra.model_loader import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix-tokens", type=int, default=1024)
    ap.add_argument("--cont-tokens", type=int, default=512)
    ap.add_argument("--min-cont-tokens", type=int, default=32)
    ap.add_argument("--tokenizer", default="qwen3.5-9b")
    ap.add_argument("--out", default="data/processed/swe_lite_target.jsonl")
    ap.add_argument("--prompts-out", default="data/processed/swe_lite.jsonl")
    a = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    tok = load_tokenizer(a.tokenizer)
    rows, prompts = [], []
    seen = set()
    scanned = 0
    for r in ds:
        scanned += 1
        iid = r.get("instance_id", f"swe-{scanned}")
        if iid in seen:
            continue
        seen.add(iid)
        issue = (r.get("problem_statement") or "").strip()
        hints = (r.get("hints_text") or "").strip()
        patch = (r.get("patch") or "").strip()
        if len(issue) < 200 or len(patch) < 100:
            continue
        header = (
            "You are a software engineer. Resolve the issue in the repository "
            f"{r.get('repo', 'unknown')}. Read the report and the hints, then "
            "write the patch.\n\n"
        )
        prefix_text = header + issue
        if hints:
            prefix_text += "\n\nHints:\n" + hints
        prefix_text += "\n\nPatch:\n"
        p_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
        c_ids = tok(patch, add_special_tokens=False)["input_ids"]
        if len(c_ids) < a.min_cont_tokens:
            continue
        p_ids = p_ids[-a.prefix_tokens:]
        c_ids = c_ids[:a.cont_tokens]
        cid = f"swe-{len(rows)}"
        row = {
            "id": cid, "domain": "swe_lite",
            "prompt": tok.decode(p_ids),
            "input_ids": p_ids, "output_ids": c_ids,
            "text": tok.decode(c_ids),
            "meta": {"source": a.dataset, "instance_id": iid,
                     "repo": r.get("repo"), "base_commit": r.get("base_commit"),
                     "prefix_tokens": len(p_ids), "cont_tokens": len(c_ids)},
        }
        rows.append(row)
        prompts.append({
            "id": cid, "domain": "swe_lite", "prompt": row["prompt"],
            "reference": None,
            "meta": {"source": a.dataset, "instance_id": iid, "repo": r.get("repo")},
        })
        if len(rows) >= a.n:
            break

    data.write_jsonl(a.out, rows)
    data.write_jsonl(a.prompts_out, prompts)
    print(f"scanned {scanned}; wrote {len(rows)} -> {a.out}; prompts -> {a.prompts_out}")


if __name__ == "__main__":
    main()
