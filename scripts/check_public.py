#!/usr/bin/env python3
"""Fail if the public tree leaks submission/venue information or breaks the series contract."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

BANNED = [
    r"\bTMLR\b",
    r"under review",
    r"camera[- ]ready",
    r"CCF[- ]?A",
    r"target_venue",
    r"submission_status",
    r"paper/venues",
    r"paper/tmlr",
    r"paper/arxiv",
    r"companion technical report",
    r"expanded technical report",
    r"conference submission",
    r"submission to TMLR",
]

PRIVATE_KEYS = {
    "target_venue",
    "submission_status",
    "review_round",
    "candidate_venues",
    "venue_manifest",
    "harness",
}

SCAN_PATHS = [
    "README.md",
    "SERIES.md",
    "CITATION.cff",
    "CHANGELOG.md",
    ".zenodo.json",
    "series",
    "docs",
    "site",
    "artifacts",
]

ALLOWED_STATUS = {"preparation", "drafting", "stable", "revision", "archived"}


def iter_files(root: Path):
    for rel in SCAN_PATHS:
        path = root / rel
        if path.is_file():
            yield path
        elif path.is_dir():
            for p in path.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".md", ".yaml", ".yml", ".json", ".html", ".cff", ".txt"}:
                    yield p


def check_text(root: Path) -> list[str]:
    errors = []
    patterns = [(p, re.compile(p, re.IGNORECASE)) for p in BANNED]
    for path in iter_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, rx in patterns:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                errors.append(f"{path.relative_to(root)}:{line}: banned term {pattern!r}: {m.group(0)!r}")
    return errors


def walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, f"{path}.{k}" if path else k
            yield from walk_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_keys(v, f"{path}[{i}]")


def check_series(root: Path) -> list[str]:
    errors = []
    series_path = root / "series" / "series.yaml"
    if not series_path.exists():
        return [f"missing {series_path}"]
    data = yaml.safe_load(series_path.read_text(encoding="utf-8"))
    for key, where in walk_keys(data):
        if key in PRIVATE_KEYS:
            errors.append(f"{series_path}: private key {key!r} at {where}")
    for study in data.get("studies", []):
        sid = study.get("id", "?")
        status = study.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{series_path}: {sid} has invalid status {status!r}")
        if status in {"drafting", "stable", "revision"}:
            draft = study.get("draft")
            if not draft or not (root / draft).exists():
                errors.append(f"{series_path}: {sid} status={status} but draft missing: {draft!r}")
    return errors


def check_no_source(root: Path) -> list[str]:
    errors = []
    if (root / "paper").exists():
        errors.append("public tree must not contain paper/ (LaTeX source stays private)")
    return errors


def check_pdfs(root: Path) -> list[str]:
    errors = []
    if not shutil_which("pdftotext"):
        print("warning: pdftotext not found; skipping PDF text scan", file=sys.stderr)
        return errors
    for pdf in (root / "series").rglob("*.pdf"):
        out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
        text = out.stdout
        for pattern in [r"\bTMLR\b", r"under review", r"conference submission"]:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"{pdf.relative_to(root)}: PDF text contains {pattern!r}")
    return errors


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    errors = []
    errors += check_no_source(root)
    errors += check_text(root)
    errors += check_series(root)
    errors += check_pdfs(root)

    if errors:
        print("public-tree check failed:")
        for e in errors:
            print("  -", e)
        return 1
    print("public tree clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
