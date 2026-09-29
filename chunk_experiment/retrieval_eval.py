from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from .retrieval import DenseRetriever, SearchResult


@dataclass(frozen=True, slots=True)
class RelevantSpan:
    document_id: str
    start: int
    end: int
    relevance: int = 1

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("relevant span must satisfy 0 <= start < end")
        if self.relevance <= 0:
            raise ValueError("relevance must be > 0")


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    query: str
    relevant_spans: tuple[RelevantSpan, ...]

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id is required")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.relevant_spans:
            raise ValueError("at least one relevant span is required")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    query_id: str
    k: int
    hit_rate: float
    precision: float
    span_recall: float
    reciprocal_rank: float
    ndcg: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AggregateRetrievalMetrics:
    k: int
    queries: int
    hit_rate: float
    precision: float
    span_recall: float
    mrr: float
    ndcg: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _overlaps(result: SearchResult, span: RelevantSpan) -> bool:
    chunk = result.item.chunk
    if result.item.document_id != span.document_id:
        return False
    return max(chunk.start, span.start) < min(chunk.end, span.end)


def _matching_span_indexes(
    result: SearchResult, spans: Sequence[RelevantSpan]
) -> list[int]:
    return [index for index, span in enumerate(spans) if _overlaps(result, span)]


def evaluate_query(
    case: QueryCase,
    results: Sequence[SearchResult],
    *,
    k: int,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("k must be > 0")
    top = list(results[:k])
    matched_unique: set[int] = set()
    relevant_results = 0
    first_rank: int | None = None
    gains: list[int] = []

    for result in top:
        matches = _matching_span_indexes(result, case.relevant_spans)
        if matches:
            relevant_results += 1
            if first_rank is None:
                first_rank = result.rank
        new_matches = [index for index in matches if index not in matched_unique]
        if new_matches:
            gains.append(max(case.relevant_spans[index].relevance for index in new_matches))
            matched_unique.update(new_matches)
        else:
            gains.append(0)

    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_gains = sorted(
        (span.relevance for span in case.relevant_spans),
        reverse=True,
    )[:k]
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, 1)
    )
    denominator = len(top) if top else k
    return RetrievalMetrics(
        query_id=case.query_id,
        k=k,
        hit_rate=1.0 if matched_unique else 0.0,
        precision=relevant_results / denominator,
        span_recall=len(matched_unique) / len(case.relevant_spans),
        reciprocal_rank=(1.0 / first_rank) if first_rank is not None else 0.0,
        ndcg=(dcg / idcg) if idcg else 0.0,
    )


def evaluate_retriever(
    retriever: DenseRetriever,
    cases: Sequence[QueryCase],
    *,
    top_k: Sequence[int] = (1, 3, 5),
) -> tuple[dict[int, AggregateRetrievalMetrics], list[RetrievalMetrics]]:
    if not cases:
        raise ValueError("at least one query case is required")
    ks = tuple(sorted(set(top_k)))
    if not ks or ks[0] <= 0:
        raise ValueError("top_k must contain positive integers")

    per_query: list[RetrievalMetrics] = []
    max_k = ks[-1]
    for case in cases:
        results = retriever.search(case.query, top_k=max_k)
        for k in ks:
            per_query.append(evaluate_query(case, results, k=k))

    aggregates: dict[int, AggregateRetrievalMetrics] = {}
    for k in ks:
        rows = [row for row in per_query if row.k == k]
        aggregates[k] = AggregateRetrievalMetrics(
            k=k,
            queries=len(rows),
            hit_rate=mean(row.hit_rate for row in rows),
            precision=mean(row.precision for row in rows),
            span_recall=mean(row.span_recall for row in rows),
            mrr=mean(row.reciprocal_rank for row in rows),
            ndcg=mean(row.ndcg for row in rows),
        )
    return aggregates, per_query


@dataclass(frozen=True, slots=True)
class RetrievalGate:
    """Optional regression thresholds for one aggregate K value."""

    min_hit_rate: float | None = None
    min_span_recall: float | None = None
    min_mrr: float | None = None
    min_ndcg: float | None = None

    def failures(self, metrics: AggregateRetrievalMetrics) -> list[str]:
        checks = (
            ("hit_rate", metrics.hit_rate, self.min_hit_rate),
            ("span_recall", metrics.span_recall, self.min_span_recall),
            ("mrr", metrics.mrr, self.min_mrr),
            ("ndcg", metrics.ndcg, self.min_ndcg),
        )
        failures: list[str] = []
        for name, actual, minimum in checks:
            if minimum is not None and actual < minimum:
                failures.append(f"{name}={actual:.4f} < required {minimum:.4f}")
        return failures
