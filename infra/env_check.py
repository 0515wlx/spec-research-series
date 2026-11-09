"""Environment / model / data sanity check."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import sys

from infra import model_registry


def _pkg(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"ERROR: {e!r}"


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False, "n": 0, "gpus": []}
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "total_gb": round(p.total_memory / 1e9, 1)})
        return {"cuda": True, "n": torch.cuda.device_count(), "gpus": gpus}
    except Exception as e:
        return {"error": repr(e)}


def disk_info(path="/path/to"):
    try:
        u = shutil.disk_usage(path)
        return {"path": path, "total_gb": round(u.total / 1e9, 1),
                "used_gb": round(u.used / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
    except Exception as e:
        return {"error": repr(e)}


def modelscope_status():
    try:
        import requests
        r = requests.get("https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B", timeout=10)
        return {"http": r.status_code, "ok": r.status_code == 200}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def tokenizer_check():
    try:
        from transformers import AutoTokenizer
        out = {}
        for k in ["qwen3.5-4b", "qwen3.5-9b"]:
            m = model_registry.MODELS[k]
            if not m.present:
                out[k] = {"present": False}
                continue
            tok = AutoTokenizer.from_pretrained(m.path, trust_remote_code=True)
            out[k] = {"present": True, "vocab_size": len(tok),
                      "bos": tok.bos_token_id, "eos": tok.eos_token_id,
                      "pad": tok.pad_token_id}
        if all(v.get("vocab_size") for v in out.values()):
            out["consistent"] = len({v["vocab_size"] for v in out.values() if v.get("present")}) <= 1
        return out
    except Exception as e:
        return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rep = {
        "python": sys.version.split()[0],
        "packages": {k: _pkg(k) for k in
                     ["torch", "transformers", "lm_eval", "modelscope", "accelerate",
                      "scipy", "matplotlib", "pandas", "pyarrow", "safetensors"]},
        "gpu": gpu_info(),
        "disk": disk_info(),
        "modelscope": modelscope_status(),
        "models": model_registry.list_models(),
        "capacity_series": model_registry.capacity_table(),
        "tokenizer": tokenizer_check(),
    }
    print(json.dumps(rep, indent=1))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
