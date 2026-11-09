"""Export a text-only HF checkpoint from a multimodal Qwen3.5 checkpoint.

Useful for lm-eval, vLLM, GPTQ/AWQ, and any tool that expects a standard causal LM.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from infra.model_loader import load_text_model, load_tokenizer
from infra import model_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-shard-size", default="5GB")
    ap.add_argument("--no-think", action="store_true",
                    help="rewrite the chat template to always use the empty-think (no-thinking) branch")
    a = ap.parse_args()

    path = model_registry.resolve(a.model)
    os.makedirs(a.out, exist_ok=True)
    print(f"loading text model from {path}", flush=True)
    model = load_text_model(path, dtype=a.dtype, device_map="cpu")
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size=a.max_shard_size)
    tok = load_tokenizer(path)
    if a.no_think:
        ct = tok.chat_template
        if isinstance(ct, str):
            old = "{%- if enable_thinking is defined and enable_thinking is false %}"
            if old in ct:
                ct = ct.replace(old, "{%- if true %}")
            else:
                print("warning: no-think marker not found in chat template; leaving unchanged")
            tok.chat_template = ct
        else:
            print("warning: chat_template is not a string; --no-think skipped")
    tok.save_pretrained(a.out)
    # copy tokenizer extras if present
    for fn in ["chat_template.jinja", "generation_config.json", "preprocessor_config.json"]:
        src = os.path.join(path, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, fn))
    with open(os.path.join(a.out, "README_TEXT_ONLY.md"), "w") as f:
        f.write(f"# Text-only export of {a.model}\n\n"
                f"Source: `{path}`\n\n"
                "The vision tower and multimodal projector were dropped.  The language model is a\n"
                "standard `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` and can be loaded with\n"
                "`AutoModelForCausalLM.from_pretrained`.\n")
    print(f"saved text-only checkpoint -> {a.out}")


if __name__ == "__main__":
    main()
