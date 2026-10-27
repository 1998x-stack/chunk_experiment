"""Source-span retrieval evaluation, independent of chunk boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .embeddings import EmbeddingProvider, l2_normalize
from .models import Chunk


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    document_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.document_id or self.start < 0 or self.end <= self.start:
            raise ValueError("gold evidence must have document_id and nonempty valid offsets")


@dataclass(frozen=True, slots=True)
class QueryCase:
    case_id: str
    query: str
    relevant_spans: tuple[EvidenceSpan, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.query.strip() or not self.relevant_spans:
            raise ValueError("gold query requires case_id, nonempty query and evidence")
        by_doc: dict[str, list[EvidenceSpan]] = {}
        for span in self.relevant_spans:
            by_doc.setdefault(span.document_id, []).append(span)
        for spans in by_doc.values():
            ordered = sorted(spans, key=lambda span: span.start)
            if any(
                left.end > right.start
                for left, right in zip(ordered, ordered[1:], strict=False)
            ):
                raise ValueError("overlapping gold spans must be merged before evaluation")


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    chunk: Chunk
    score: float

    def __post_init__(self) -> None:
        if not self.document_id or not isfinite(self.score):
            raise ValueError("search hit must contain a document id and finite score")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_at_k: float
    reciprocal_rank_at_k: float
    evidence_char_recall_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _merged_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0
    for left, right in ordered[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def evaluate_retrieval(
    case: QueryCase, ranked_hits: Sequence[SearchHit], *, k: int = 5
) -> RetrievalMetrics:
    """Span-char recall measures covered *evidence*, not arbitrary chunk overlap.

    Gold spans are labelled against the source document and stay unchanged when
    chunk boundaries or numbers of chunks change. Multiple retrieved overlapping
    chunks cannot double-count covered evidence.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    hits = ranked_hits[:k]
    first_relevant_rank = 0
    relevant_characters = sum(span.end - span.start for span in case.relevant_spans)
    covered_characters = 0
    for gold in case.relevant_spans:
        intersections = []
        for rank, hit in enumerate(hits, start=1):
            if hit.document_id != gold.document_id:
                continue
            start = max(gold.start, hit.chunk.start)
            end = min(gold.end, hit.chunk.end)
            if end > start:
                intersections.append((start, end))
                if first_relevant_rank == 0:
                    first_relevant_rank = rank
                else:
                    first_relevant_rank = min(first_relevant_rank, rank)
        covered_characters += _merged_length(intersections)
    return RetrievalMetrics(
        hit_at_k=float(first_relevant_rank > 0),
        reciprocal_rank_at_k=(1.0 / first_relevant_rank if first_relevant_rank else 0.0),
        evidence_char_recall_at_k=covered_characters / relevant_characters,
    )


class DenseIndex:
    """Exact cosine baseline; no ANN approximation or reranking confounders."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        items: Iterable[tuple[str, Chunk]],
    ) -> None:
        self.provider = provider
        self.items = list(items)
        if not self.items:
            raise ValueError("cannot build retrieval index with zero chunks")
        if any(not doc_id or not chunk.text for doc_id, chunk in self.items):
            raise ValueError("index requires nonempty documents and chunks")
        vectors = np.asarray(provider.embed([c.text for _, c in self.items]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.items) or vectors.shape[1] < 1:
            raise ValueError("embedding provider returned invalid index dimensions")
        self.vectors = l2_normalize(vectors)

    def search(self, query: str, *, k: int = 5) -> list[SearchHit]:
        if k <= 0 or not query.strip():
            raise ValueError("search requires nonempty query and positive k")
        vectors = np.asarray(self.provider.embed([query]), dtype=np.float32)
        if vectors.shape != (1, self.vectors.shape[1]):
            raise ValueError("query embedding dimension does not match index")
        scores = self.vectors @ l2_normalize(vectors)[0]
        order = sorted(
            range(len(scores)),
            key=lambda idx: (-float(scores[idx]), self.items[idx][0],
                             self.items[idx][1].start, self.items[idx][1].end),
        )
        return [
            SearchHit(self.items[idx][0], self.items[idx][1], float(scores[idx]))
            for idx in order[:k]
        ]
