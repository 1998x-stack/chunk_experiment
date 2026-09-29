from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from .retrieval import Retriever, SearchResult
from .retrieval_eval import (
    AggregateRetrievalMetrics,
    QueryCase,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    evaluate_query,
)


@dataclass(frozen=True, slots=True)
class QueryCostMetrics:
    query_id: str
    latency_ms: float
    returned_chunks: int
    context_chars: int
    unique_context_chars: int
    duplication_ratio: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AggregateCostMetrics:
    queries: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_context_chars: float
    mean_unique_context_chars: float
    mean_duplication_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _unique_context_chars(results: Sequence[SearchResult]) -> int:
    by_document: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for result in results:
        by_document[result.item.document_id].append(
            (
                result.item.chunk.start,
                result.item.chunk.end,
            )
        )

    total = 0
    for intervals in by_document.values():
        cursor = -1
        for start, end in sorted(intervals):
            if end <= cursor:
                continue
            total += end - max(start, cursor)
            cursor = max(cursor, end)
    return total


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(0.95 * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def evaluate_retriever_profiled(
    retriever: Retriever,
    cases: Sequence[QueryCase],
    *,
    top_k: Sequence[int] = (1, 3, 5),
    cost_k: int | None = None,
) -> tuple[
    dict[int, AggregateRetrievalMetrics],
    list[RetrievalMetrics],
    AggregateCostMetrics,
    list[QueryCostMetrics],
]:
    if not cases:
        raise ValueError("at least one query case is required")
    ks = tuple(sorted(set(top_k)))
    if not ks or ks[0] <= 0:
        raise ValueError("top_k must contain positive integers")

    resolved_cost_k = cost_k if cost_k is not None else ks[-1]
    if resolved_cost_k <= 0:
        raise ValueError("cost_k must be > 0")
    search_k = max(ks[-1], resolved_cost_k)

    rows: list[RetrievalMetrics] = []
    costs: list[QueryCostMetrics] = []
    for case in cases:
        started = time.perf_counter()
        results = retriever.search(
            case.query,
            top_k=search_k,
        )
        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0
        for k in ks:
            rows.append(
                evaluate_query(
                    case,
                    results,
                    k=k,
                )
            )

        cost_results = results[:resolved_cost_k]
        context_chars = sum(
            result.item.chunk.char_length
            for result in cost_results
        )
        unique_chars = _unique_context_chars(cost_results)
        duplicate_chars = max(
            0,
            context_chars - unique_chars,
        )
        costs.append(
            QueryCostMetrics(
                query_id=case.query_id,
                latency_ms=latency_ms,
                returned_chunks=len(cost_results),
                context_chars=context_chars,
                unique_context_chars=unique_chars,
                duplication_ratio=(
                    duplicate_chars / context_chars
                    if context_chars
                    else 0.0
                ),
            )
        )

    aggregates = aggregate_retrieval_metrics(
        rows,
        ks=ks,
    )
    cost_summary = AggregateCostMetrics(
        queries=len(costs),
        mean_latency_ms=mean(
            row.latency_ms for row in costs
        ),
        p95_latency_ms=_p95(
            [row.latency_ms for row in costs]
        ),
        mean_context_chars=mean(
            row.context_chars for row in costs
        ),
        mean_unique_context_chars=mean(
            row.unique_context_chars for row in costs
        ),
        mean_duplication_ratio=mean(
            row.duplication_ratio for row in costs
        ),
    )
    return aggregates, rows, cost_summary, costs
