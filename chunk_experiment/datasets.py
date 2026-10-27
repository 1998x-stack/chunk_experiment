"""Deterministic manifest-based corpus and gold-label loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval import EvidenceSpan, QueryCase


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    text: str
    split: str
    language: str
    domain: str
    relative_path: str
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def load_corpus(manifest: Path, root: Path) -> list[Document]:
    """No implicit globbing, silent UTF-8 replacement, or cross-split duplicates."""
    root = root.resolve()
    output = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for record in _jsonl_rows(manifest):
        doc_id = record["document_id"]
        split = record["split"]
        relative_path = record["path"]
        if not all(isinstance(value, str) and value for value in (doc_id, split, relative_path)):
            raise ValueError("document_id, split and path must be nonempty strings")
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"invalid corpus path: {relative_path}")
        if doc_id in seen_ids:
            raise ValueError(f"duplicate document id: {doc_id}")
        raw = candidate.read_bytes()
        digest = sha256_bytes(raw)
        expected_sha = record.get("sha256")
        if expected_sha is not None and digest != expected_sha:
            raise ValueError(f"dataset checksum mismatch: {doc_id}")
        old_split = seen_hashes.get(digest)
        if old_split is not None and old_split != split:
            raise ValueError(f"identical document appears in different splits: {doc_id}")
        seen_ids.add(doc_id)
        seen_hashes[digest] = split
        output.append(
            Document(
                document_id=doc_id,
                text=raw.decode("utf-8"),
                split=split,
                language=record.get("language", "unknown"),
                domain=record.get("domain", "unknown"),
                relative_path=relative_path,
                sha256=digest,
            )
        )
    if not output:
        raise ValueError("empty dataset manifest")
    return sorted(output, key=lambda doc: doc.document_id)


def load_gold(path: Path, documents: list[Document], *, split: str = "test") -> list[QueryCase]:
    """Validate source-relative evidence and limit evaluation to a named split."""
    by_id = {doc.document_id: doc for doc in documents}
    cases = []
    seen_ids: set[str] = set()
    for row in _jsonl_rows(path):
        case_id = row["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        spans = tuple(EvidenceSpan(**item) for item in row["relevant_spans"])
        for span in spans:
            doc = by_id.get(span.document_id)
            if doc is None or doc.split != split:
                raise ValueError(
                    f"gold references absent or non-{split} document: {span.document_id}"
                )
            if span.end > len(doc.text):
                raise ValueError(f"gold span exceeds document boundary: {case_id}")
        cases.append(QueryCase(case_id=case_id, query=row["query"], relevant_spans=spans))
    if not cases:
        raise ValueError("empty retrieval gold labels")
    return sorted(cases, key=lambda case: case.case_id)
