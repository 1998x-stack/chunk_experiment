from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from .rerank import TwoStageRetriever
from .retrieval_eval import QueryCase, evaluate_query


@dataclass(frozen=True, slots=True)
class TwoStageQueryMetrics:
    query_id: str
    candidate_k: int
    final_k: int
    candidate_hit_rate: float
    candidate_span_recall: float
    prefix_mrr: float
    prefix_ndcg: float
    final_hit_rate: float
    final_span_recall: float
    final_mrr: float
    final_ndcg: float
    mrr_lift: float
    ndcg_lift: float
    candidate_latency_ms: float
    rerank_latency_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AggregateTwoStageMetrics:
    queries: int
    candidate_k: int
    final_k: int
    candidate_hit_rate: float
    candidate_span_recall: float
    prefix_mrr: float
    prefix_ndcg: float
    final_hit_rate: float
    final_span_recall: float
    final_mrr: float
    final_ndcg: float
    mean_mrr_lift: float
    mean_ndcg_lift: float
    mean_candidate_latency_ms: float
    mean_rerank_latency_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TwoStageGate:
    min_candidate_span_recall: float | None = None
    min_final_ndcg: float | None = None
    min_mean_ndcg_lift: float | None = None

    def failures(
        self,
        metrics: AggregateTwoStageMetrics,
    ) -> list[str]:
        checks = (
            (
                "candidate_span_recall",
                metrics.candidate_span_recall,
                self.min_candidate_span_recall,
            ),
            (
                "final_ndcg",
                metrics.final_ndcg,
                self.min_final_ndcg,
            ),
            (
                "mean_ndcg_lift",
                metrics.mean_ndcg_lift,
                self.min_mean_ndcg_lift,
            ),
        )
        failures: list[str] = []
        for name, actual, minimum in checks:
            if minimum is not None and actual < minimum:
                failures.append(
                    f"{name}={actual:.4f} < required {minimum:.4f}"
                )
        return failures


def evaluate_two_stage_retriever(
    retriever: TwoStageRetriever,
    cases: Sequence[QueryCase],
    *,
    final_k: int,
) -> tuple[
    AggregateTwoStageMetrics,
    list[TwoStageQueryMetrics],
]:
    if not cases:
        raise ValueError("at least one query case is required")
    if final_k <= 0:
        raise ValueError("final_k must be > 0")
    if final_k > retriever.candidate_k:
        raise ValueError(
            "final_k cannot exceed retriever candidate_k"
        )

    rows: list[TwoStageQueryMetrics] = []
    for case in cases:
        trace = retriever.search_with_trace(
            case.query,
            top_k=final_k,
        )
        candidate_ceiling = evaluate_query(
            case,
            trace.candidates,
            k=trace.candidate_k,
        )
        prefix = evaluate_query(
            case,
            trace.candidates,
            k=final_k,
        )
        final = evaluate_query(
            case,
            trace.final_results,
            k=final_k,
        )
        rows.append(
            TwoStageQueryMetrics(
                query_id=case.query_id,
                candidate_k=trace.candidate_k,
                final_k=final_k,
                candidate_hit_rate=candidate_ceiling.hit_rate,
                candidate_span_recall=candidate_ceiling.span_recall,
                prefix_mrr=prefix.reciprocal_rank,
                prefix_ndcg=prefix.ndcg,
                final_hit_rate=final.hit_rate,
                final_span_recall=final.span_recall,
                final_mrr=final.reciprocal_rank,
                final_ndcg=final.ndcg,
                mrr_lift=(
                    final.reciprocal_rank
                    - prefix.reciprocal_rank
                ),
                ndcg_lift=final.ndcg - prefix.ndcg,
                candidate_latency_ms=trace.candidate_latency_ms,
                rerank_latency_ms=trace.rerank_latency_ms,
            )
        )

    aggregate = AggregateTwoStageMetrics(
        queries=len(rows),
        candidate_k=retriever.candidate_k,
        final_k=final_k,
        candidate_hit_rate=mean(
            row.candidate_hit_rate for row in rows
        ),
        candidate_span_recall=mean(
            row.candidate_span_recall for row in rows
        ),
        prefix_mrr=mean(row.prefix_mrr for row in rows),
        prefix_ndcg=mean(row.prefix_ndcg for row in rows),
        final_hit_rate=mean(
            row.final_hit_rate for row in rows
        ),
        final_span_recall=mean(
            row.final_span_recall for row in rows
        ),
        final_mrr=mean(row.final_mrr for row in rows),
        final_ndcg=mean(row.final_ndcg for row in rows),
        mean_mrr_lift=mean(row.mrr_lift for row in rows),
        mean_ndcg_lift=mean(row.ndcg_lift for row in rows),
        mean_candidate_latency_ms=mean(
            row.candidate_latency_ms for row in rows
        ),
        mean_rerank_latency_ms=mean(
            row.rerank_latency_ms for row in rows
        ),
    )
    return aggregate, rows
