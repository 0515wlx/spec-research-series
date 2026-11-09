"""Model registry for the Route A capacity series."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional

MODEL_ROOT = "/path/to/models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    params_b: float
    kind: str            # dense | moe
    active_b: Optional[float] = None
    note: str = ""

    @property
    def path(self) -> str:
        # Qwen/Qwen3.5-2B -> /path/to/models/Qwen3.5-2B
        return os.path.join(MODEL_ROOT, self.repo.split("/")[-1])

    @property
    def present(self) -> bool:
        return os.path.isdir(self.path) and any(
            f.endswith((".safetensors", ".bin")) for f in os.listdir(self.path)
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(path=self.path, present=self.present)
        return d


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-0.8b": ModelSpec("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", 0.8, "dense"),
    "qwen3.5-2b": ModelSpec("qwen3.5-2b", "Qwen/Qwen3.5-2B", 2.0, "dense"),
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", 4.0, "dense"),
    "qwen3.5-9b": ModelSpec("qwen3.5-9b", "Qwen/Qwen3.5-9B", 9.0, "dense"),
    "qwen3.5-27b": ModelSpec("qwen3.5-27b", "Qwen/Qwen3.5-27B", 27.0, "dense"),
    "qwen3.5-35b-a3b": ModelSpec(
        "qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B", 35.0, "moe", active_b=3.0,
        note="target; 256 experts, top-8 active",
    ),
    # legacy target used in REPORT.md, kept for cross-checking
    "qwen3.6-35b-a3b": ModelSpec(
        "qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B", 35.0, "moe", active_b=3.4,
        note="legacy teacher; same architecture family",
    ),
    "qwen3.6-27b": ModelSpec("qwen3.6-27b", "Qwen/Qwen3.6-27B", 27.0, "dense"),
    "qwen3.8-27b": ModelSpec("qwen3.8-27b", "Qwen/Qwen3.8-27B", 27.0, "dense"),
    # Original Qwen3 family (different tokenizer from Qwen3.5/3.6; use its own target).
    "qwen3-0.6b": ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", 0.6, "dense"),
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", 1.7, "dense"),
    "qwen3-4b": ModelSpec("qwen3-4b", "Qwen/Qwen3-4B-Instruct-2507", 4.0, "dense"),
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", 8.0, "dense"),
    "qwen3-14b": ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", 14.0, "dense"),
    "qwen3-32b": ModelSpec("qwen3-32b", "Qwen/Qwen3-32B", 32.0, "dense"),
    # IFM K2 Horizon family: same recipe, ~20T tokens, 512K context, Apache-2.0.
    # 0.9B has a smaller vocabulary; keep it out of the shared-tokenizer capacity axis.
    "k2-horizon-3.7b": ModelSpec("k2-horizon-3.7b", "IFM/K2-Horizon-3.7B", 3.7, "dense"),
    "k2-horizon-7b": ModelSpec("k2-horizon-7b", "IFM/K2-Horizon-7B", 7.0, "dense"),
    "k2-horizon-32b": ModelSpec("k2-horizon-32b", "IFM/K2-Horizon-32B", 32.0, "dense",
                                note="34.78B stored incl. embeddings"),
    "k2-horizon-mova-36b-a4b": ModelSpec(
        "k2-horizon-mova-36b-a4b", "IFM/K2-Horizon-MoVA-36B-A4B", 36.0, "moe", active_b=4.0,
        note="K2 Horizon MoVA + MoE target",
    ),
}

CAPACITY_SERIES = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]


def get(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}")
    return MODELS[key]


def resolve(key_or_path: str) -> str:
    """Accept a registry key or an existing directory."""
    if key_or_path in MODELS:
        return MODELS[key_or_path].path
    if os.path.isdir(key_or_path):
        return key_or_path
    raise FileNotFoundError(f"not a registry key and not a directory: {key_or_path}")


def list_models(present_only: bool = False) -> list[dict]:
    out = [m.to_dict() for m in MODELS.values()]
    if present_only:
        out = [m for m in out if m["present"]]
    return out


def capacity_table() -> list[dict]:
    rows = []
    for k in CAPACITY_SERIES:
        m = MODELS[k]
        rows.append({"key": k, "params_b": m.params_b, "path": m.path, "present": m.present})
    return rows


if __name__ == "__main__":
    import json

    print(json.dumps({"models": list_models(), "capacity_series": capacity_table()}, indent=1))
