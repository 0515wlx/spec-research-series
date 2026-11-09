"""Download missing Qwen3.5 models, with ModelScope and HF-mirror fallback.

Usage:
  python -m infra.download --models qwen3.5-0.8b qwen3.5-2b qwen3.5-27b qwen3.5-35b-a3b

On this machine HuggingFace itself is blocked but ``https://hf-mirror.com`` works, and
ModelScope's API has been intermittently returning HTTP 500.  ``--backend auto`` tries
ModelScope first, then the HF mirror.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from infra import model_registry

LOG_DIR = "/path/to/spec-research/out/logs"
HF_MIRROR = "https://hf-mirror.com"


def _run(cmd: list[str], log: str, env: dict | None = None) -> int:
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)} ===\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def download_modelscope(key: str, workers: int, force: bool, log: str) -> bool:
    spec = model_registry.get(key)
    cmd = ["modelscope", "download", "--model", spec.repo,
           "--local-dir", spec.path, "--max-workers", str(workers)]
    if force:
        cmd.append("--force")
    return _run(cmd, log) == 0 and spec.present


def download_hf(key: str, workers: int, force: bool, log: str) -> bool:
    spec = model_registry.get(key)
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", HF_MIRROR)
    # new huggingface_hub CLI is `hf`; fall back to huggingface-cli
    exe = "hf" if subprocess.run(["which", "hf"], capture_output=True).returncode == 0 else "huggingface-cli"
    cmd = [exe, "download", spec.repo, "--local-dir", spec.path, "--max-workers", str(workers)]
    if force:
        cmd.append("--force")
    return _run(cmd, log, env=env) == 0 and spec.present


def download_one(key: str, backend: str, retry: int, sleep: float, workers: int, force: bool) -> bool:
    spec = model_registry.get(key)
    if spec.present and not force:
        print(f"[skip] {key} already present at {spec.path}")
        return True
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"download_{key}.log")
    backends = ["modelscope", "hf"] if backend == "auto" else [backend]
    for attempt in range(1, retry + 1):
        for b in backends:
            print(f"[{key}] attempt {attempt}/{retry} via {b} -> {spec.path}", flush=True)
            ok = download_modelscope(key, workers, force, log) if b == "modelscope" else \
                 download_hf(key, workers, force, log)
            if ok:
                print(f"[ok] {key} ({b})")
                return True
        wait = min(sleep * (2 ** (attempt - 1)), 300)
        print(f"[retry] {key} failed; sleeping {wait:.0f}s (log: {log})", flush=True)
        time.sleep(wait)
    print(f"[fail] {key}; see {log}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-27b", "qwen3.5-35b-a3b"])
    ap.add_argument("--backend", default="auto", choices=["auto", "modelscope", "hf"])
    ap.add_argument("--retry", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=20.0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    ok = {}
    for k in a.models:
        ok[k] = download_one(k, a.backend, a.retry, a.sleep, a.max_workers, a.force)
    print("summary:", ok)
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
