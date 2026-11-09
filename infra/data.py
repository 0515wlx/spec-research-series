"""Dataset registry and prompt normalization."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

DATA_DIR = "/path/to/spec-research/data"
PROCESSED = os.path.join(DATA_DIR, "processed")
WIKITEXT_PARQUET = (
    "~/ .cache/modelscope/datasets/Salesforce--wikitext/snapshots/master/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    source: str
    split: str
    subset: Optional[str] = None
    chat: bool = True
    kind: str = "qa"      # qa | code | prose | chat
    trust_remote_code: bool = False
    note: str = ""


DOMAINS: dict[str, DomainSpec] = {
    "math500": DomainSpec("math500", "HuggingFaceH4/MATH-500", "test", chat=True, kind="qa"),
    "gsm8k": DomainSpec("gsm8k", "opencompass/gsm8k", "test", chat=True, kind="qa"),
    "humaneval": DomainSpec("humaneval", "opencompass/humaneval", "test", chat=False, kind="code"),
    "wikitext": DomainSpec("wikitext", "Salesforce/wikitext", "test",
                           subset="wikitext-2-raw-v1", chat=False, kind="prose"),
    "mtbench": DomainSpec("mtbench", "lmsys/mt_bench_human_judgments", "gpt4_pair",
                          chat=True, kind="chat", trust_remote_code=True),
    "nq": DomainSpec("nq", "google-research-datasets/natural_questions", "validation",
                     chat=True, kind="qa"),
    "fineweb": DomainSpec("fineweb", "HuggingFaceFW/fineweb-edu", "train",
                          chat=False, kind="prose", trust_remote_code=True),
    "c4": DomainSpec("c4", "allenai/c4", "train", chat=False, kind="prose",
                     trust_remote_code=True),
    "gutenberg": DomainSpec("gutenberg", "common-pile/project_gutenberg", "train",
                            chat=False, kind="prose", trust_remote_code=True),
    "code": DomainSpec("code", "bigcode/the-stack-smol", "train", chat=False,
                       kind="code", trust_remote_code=True),
}


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; known: {sorted(DOMAINS)}")
    return DOMAINS[name]


def _first_user_content(conversation) -> Optional[str]:
    if not conversation:
        return None
    for turn in conversation:
        if isinstance(turn, dict) and turn.get("role") == "user":
            return str(turn.get("content", ""))
    return None


def _record_from_row(domain: str, row: dict) -> Optional[dict]:
    spec = get_domain(domain)
    if domain == "math500":
        prompt, ref = row.get("problem"), row.get("answer")
    elif domain == "gsm8k":
        prompt, ref = row.get("question"), row.get("answer")
    elif domain == "humaneval":
        prompt, ref = row.get("prompt"), row.get("canonical_solution")
    elif domain == "mtbench":
        prompt = _first_user_content(row.get("conversation_a"))
        ref = None
    elif domain == "nq":
        prompt, ref = row.get("question"), row.get("answer")
    else:
        prompt, ref = row.get("text"), None
    if not prompt or len(str(prompt).strip()) < 4:
        return None
    return {
        "id": str(row.get("id") or row.get("unique_id") or row.get("task_id") or row.get("question_id") or ""),
        "domain": domain,
        "prompt": str(prompt),
        "reference": None if ref is None else str(ref),
        "meta": {"source": spec.source, "split": spec.split, "kind": spec.kind},
    }


def _load_modelscope(domain: str, limit: int) -> list[dict]:
    from modelscope.msdatasets import MsDataset

    spec = get_domain(domain)
    kw = dict(split=spec.split, use_streaming=True, trust_remote_code=spec.trust_remote_code)
    if spec.subset:
        kw["subset_name"] = spec.subset
    ds = MsDataset.load(spec.source, **kw)
    rows = []
    seen = set()
    for i, r in enumerate(ds):
        rec = _record_from_row(domain, r)
        if rec:
            if rec["id"] and rec["id"] in seen:
                continue
            seen.add(rec["id"])
            rows.append(rec)
        if len(rows) >= limit:
            break
    return rows


def load_text() -> str:
    """Full WikiText-2 raw test text (joined)."""
    import pandas as pd

    if not os.path.exists(WIKITEXT_PARQUET):
        raise FileNotFoundError(
            f"wikitext parquet not found at {WIKITEXT_PARQUET}; run `python -m infra.prepare_data --domains wikitext`"
        )
    df = pd.read_parquet(WIKITEXT_PARQUET)
    return "\n".join(str(x) for x in df["text"].tolist())


def load_wikitext_chunks(n_chunks: int = 100, seq_len: int = 1024, min_chars: int = 200) -> list[dict]:
    """Raw continuation chunks: first half prompt, second half scored continuation."""
    import pandas as pd

    if not os.path.exists(WIKITEXT_PARQUET):
        raise FileNotFoundError(
            f"wikitext parquet not found at {WIKITEXT_PARQUET}; run `python -m infra.prepare_data --domains wikitext`"
        )
    df = pd.read_parquet(WIKITEXT_PARQUET)
    text = "\n".join(str(x) for x in df["text"].tolist())
    # character-level chunks are only a rough split; tokenizer truncation happens later
    target_chars = seq_len * 4
    out = []
    for i in range(n_chunks):
        chunk = text[i * target_chars:(i + 1) * target_chars]
        if len(chunk) < min_chars:
            break
        out.append({
            "id": f"wikitext-{i}",
            "domain": "wikitext",
            "prompt": chunk,
            "reference": None,
            "meta": {"source": "Salesforce/wikitext", "split": "test", "kind": "prose"},
        })
    return out


def load_prompts(domain: str, limit: Optional[int] = None, processed_dir: str = PROCESSED,
                 force: bool = False) -> list[dict]:
    """Prefer a processed JSONL; otherwise load from ModelScope."""
    limit = limit or 10**9
    path = os.path.join(processed_dir, f"{domain}.jsonl")
    if os.path.exists(path) and not force:
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        return rows
    if domain == "wikitext":
        return load_wikitext_chunks(n_chunks=min(limit, 1000))
    return _load_modelscope(domain, limit)


def chat_prompt(tokenizer, prompt: str, system: Optional[str] = None, thinking: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except TypeError:
        # template does not expose enable_thinking
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_target_sequences(path: str, limit: Optional[int] = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "input_ids" not in r or "output_ids" not in r:
                raise ValueError(f"target sequence missing input_ids/output_ids: {path}")
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def manifest(path: str = PROCESSED) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        try:
            n = sum(1 for _ in open(p))
        except OSError:
            n = None
        out[name] = {"path": p, "lines": n, "bytes": os.path.getsize(p)}
    return out
