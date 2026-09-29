from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .retrieval import Document
from .retrieval_eval import QueryCase, RelevantSpan


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    dataset_id: str
    documents: tuple[Document, ...]
    queries: tuple[QueryCase, ...]
    fingerprint: str


def _canonical_fingerprint(
    documents: list[Document],
    queries: list[QueryCase],
) -> str:
    payload = {
        "documents": [
            {"id": document.document_id, "text": document.text}
            for document in sorted(documents, key=lambda item: item.document_id)
        ],
        "queries": [
            {
                "id": case.query_id,
                "query": case.query,
                "relevant_spans": [
                    {
                        "document_id": span.document_id,
                        "start": span.start,
                        "end": span.end,
                        "relevance": span.relevance,
                    }
                    for span in case.relevant_spans
                ],
            }
            for case in sorted(queries, key=lambda item: item.query_id)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("evaluation dataset schema_version must be 1")
    dataset_id = str(raw.get("dataset_id") or path.stem)
    base = path.parent

    documents: list[Document] = []
    document_map: dict[str, Document] = {}
    for item in raw.get("documents", []):
        document_id = str(item["id"])
        if document_id in document_map:
            raise ValueError(f"duplicate document id: {document_id}")
        if "text" in item:
            text = str(item["text"])
        elif "path" in item:
            text = (base / str(item["path"])).read_text(encoding="utf-8")
        else:
            raise ValueError(f"document {document_id!r} needs text or path")
        document = Document(
            document_id,
            text,
            dict(item.get("metadata") or {}),
        )
        documents.append(document)
        document_map[document_id] = document

    queries: list[QueryCase] = []
    query_ids: set[str] = set()
    for item in raw.get("queries", []):
        query_id = str(item["id"])
        if query_id in query_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        query_ids.add(query_id)
        spans: list[RelevantSpan] = []
        for span_raw in item.get("relevant_spans", []):
            span = RelevantSpan(
                document_id=str(span_raw["document_id"]),
                start=int(span_raw["start"]),
                end=int(span_raw["end"]),
                relevance=int(span_raw.get("relevance", 1)),
            )
            document = document_map.get(span.document_id)
            if document is None:
                raise ValueError(
                    f"query {query_id!r} references unknown document {span.document_id!r}"
                )
            if span.end > len(document.text):
                raise ValueError(
                    f"query {query_id!r} span {span.start}:{span.end} exceeds "
                    f"document {span.document_id!r} length {len(document.text)}"
                )
            spans.append(span)
        queries.append(
            QueryCase(
                query_id,
                str(item["query"]),
                tuple(spans),
            )
        )

    if not documents:
        raise ValueError("evaluation dataset must contain documents")
    if not queries:
        raise ValueError("evaluation dataset must contain queries")

    return EvaluationDataset(
        dataset_id=dataset_id,
        documents=tuple(documents),
        queries=tuple(queries),
        fingerprint=_canonical_fingerprint(documents, queries),
    )
