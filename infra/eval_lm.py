"""Thin wrapper around lm-eval.

Example:
  python -m infra.eval_lm --model qwen3.5-9b \
    --tasks gsm8k,minerva_math,humaneval,wikitext \
    --batch-size 8 --apply-chat-template --out out/lm_eval/qwen3.5-9b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

from infra import model_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry key or path (prefer text-only export)")
    ap.add_argument("--tasks", default="gsm8k,minerva_math,humaneval,wikitext")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--gen-kwargs", default=None, help="e.g. 'max_gen_toks=1024'")
    ap.add_argument("--log-samples", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    path = model_registry.resolve(a.model) if a.model in model_registry.MODELS else a.model
    os.makedirs(a.out, exist_ok=True)
    model_args = f"pretrained={path},dtype={a.dtype},trust_remote_code=True"
    if a.backend == "vllm":
        model_args = f"pretrained={path},trust_remote_code=True"
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", a.backend,
        "--model_args", model_args,
        "--tasks", a.tasks,
        "--batch_size", str(a.batch_size),
        "--device", a.device,
        "--output_path", a.out,
    ]
    if a.apply_chat_template:
        cmd.append("--apply_chat_template")
    if a.gen_kwargs:
        cmd += ["--gen_kwargs", a.gen_kwargs]
    if a.log_samples:
        cmd.append("--log_samples")
    if a.limit is not None:
        cmd += ["--limit", str(a.limit)]
    if a.num_fewshot is not None:
        cmd += ["--num_fewshot", str(a.num_fewshot)]
    print(" ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        print("lm-eval failed.  If the model is multimodal, export a text-only checkpoint:\n"
              f"  python -m infra.export_text_model --model {a.model} --out ckpt/text/{a.model}\n"
              f"  python -m infra.eval_lm --model ckpt/text/{a.model} ...", file=sys.stderr)
        raise SystemExit(p.returncode)

    js = sorted(glob.glob(os.path.join(a.out, "**", "*.json"), recursive=True), key=os.path.getmtime)
    if js:
        res = json.load(open(js[-1]))
        summary = {"model": a.model, "path": path, "tasks": a.tasks,
                   "results": res.get("results", {}), "config": res.get("config", {})}
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("summary ->", os.path.join(a.out, "summary.json"))
        for task, metrics in summary["results"].items():
            keep = {k: v for k, v in metrics.items() if "acc" in k or "exact" in k or "ppl" in k or "word_perplexity" in k}
            print(task, keep)


if __name__ == "__main__":
    main()
