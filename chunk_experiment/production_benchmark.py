from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from .retrieval import Retriever


@dataclass(frozen=True, slots=True)
class BenchmarkPhase:
    phase: str
    passes: int
    queries: int
    elapsed_seconds: float
    qps: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductionBenchmark:
    cold: BenchmarkPhase
    warm: BenchmarkPhase

    def to_dict(self) -> dict[str, dict[str, str | int | float]]:
        return {
            "cold": self.cold.to_dict(),
            "warm": self.warm.to_dict(),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def run_benchmark_phase(
    retriever: Retriever,
    queries: Sequence[str],
    *,
    top_k: int,
    phase: str,
    passes: int,
) -> BenchmarkPhase:
    if not queries:
        raise ValueError("at least one benchmark query is required")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if passes <= 0:
        raise ValueError("passes must be > 0")

    latencies: list[float] = []
    started_phase = time.perf_counter()
    for _ in range(passes):
        for query in queries:
            started = time.perf_counter()
            retriever.search(query, top_k=top_k)
            latencies.append(
                (time.perf_counter() - started) * 1000.0
            )
    elapsed = time.perf_counter() - started_phase
    total_queries = len(queries) * passes
    return BenchmarkPhase(
        phase=phase,
        passes=passes,
        queries=total_queries,
        elapsed_seconds=elapsed,
        qps=total_queries / elapsed if elapsed > 0 else 0.0,
        mean_latency_ms=mean(latencies) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
    )


def benchmark_cold_warm(
    retriever: Retriever,
    queries: Sequence[str],
    *,
    top_k: int,
    reset_caches: Callable[[], None],
    cold_passes: int = 1,
    warm_passes: int = 3,
) -> ProductionBenchmark:
    reset_caches()
    cold = run_benchmark_phase(
        retriever,
        queries,
        top_k=top_k,
        phase="cold",
        passes=cold_passes,
    )
    warm = run_benchmark_phase(
        retriever,
        queries,
        top_k=top_k,
        phase="warm",
        passes=warm_passes,
    )
    return ProductionBenchmark(cold=cold, warm=warm)


def pareto_frontier(
    rows: Sequence[dict[str, float | str | int]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[dict[str, float | str | int]]:
    """Return non-dominated rows without inventing a single overall winner."""

    if not rows:
        return []
    if not maximize and not minimize:
        raise ValueError("at least one Pareto dimension is required")

    def dominates(
        left: dict[str, float | str | int],
        right: dict[str, float | str | int],
    ) -> bool:
        no_worse = True
        strictly_better = False
        for key in maximize:
            left_value = float(left[key])
            right_value = float(right[key])
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
        for key in minimize:
            left_value = float(left[key])
            right_value = float(right[key])
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        return no_worse and strictly_better

    frontier: list[dict[str, float | str | int]] = []
    for index, candidate in enumerate(rows):
        if any(
            dominates(other, candidate)
            for other_index, other in enumerate(rows)
            if other_index != index
        ):
            continue
        frontier.append(dict(candidate))
    return frontier
