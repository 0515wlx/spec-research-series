"""Normalize ModelScope datasets into data/processed/{domain}.jsonl."""
from __future__ import annotations

import argparse
import json
import os

from infra import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["math500", "gsm8k", "humaneval", "wikitext", "mtbench"])
    ap.add_argument("--out", default=data.PROCESSED)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for d in a.domains:
        try:
            rows = data.load_prompts(d, limit=a.limit, processed_dir=a.out, force=True)
            path = os.path.join(a.out, f"{d}.jsonl")
            n = data.write_jsonl(path, rows)
            summary[d] = {"path": path, "rows": n}
            print(f"{d:10s} {n:6d} -> {path}")
        except Exception as e:
            summary[d] = {"error": repr(e)}
            print(f"{d:10s} ERROR {e!r}")
    man_path = os.path.join(a.out, "manifest.json")
    existing = {}
    if os.path.exists(man_path):
        try:
            existing = json.load(open(man_path))
        except Exception:
            existing = {}
    existing.update(summary)
    with open(man_path, "w") as f:
        json.dump(existing, f, indent=1)
    print("manifest ->", man_path)


if __name__ == "__main__":
    main()
