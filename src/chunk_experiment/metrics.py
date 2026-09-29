from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, median, pstdev
from typing import Iterable, Sequence

from .models import Chunk


@dataclass(frozen=True, slots=True)
class ChunkingMetrics:
    chunks: int
    mean_size: float
    median_size: float
    std_size: float
    min_size: int
    max_size: int
    coverage_ratio: float
    duplication_ratio: float
    boundary_alignment_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    relevant_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.relevant_spans:
            raise ValueError("relevant_spans must not be empty")
        for start, end in self.relevant_spans:
            if start < 0 or end <= start:
                raise ValueError("relevant spans must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    mean_span_recall_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_chunking(
    source: str,
    chunks: Sequence[Chunk],
    *,
    preferred_boundaries: tuple[str, ...] = (
        "\n",
        "。",
        "！",
        "？",
        ".",
        "!",
        "?",
        ";",
        "；",
    ),
) -> ChunkingMetrics:
    if not chunks:
        return ChunkingMetrics(
            0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            1.0 if not source else 0.0,
            0.0,
            1.0,
        )

    for chunk in chunks:
        if chunk.end > len(source):
            raise ValueError("chunk offset exceeds source length")
        if source[chunk.start : chunk.end] != chunk.text:
            raise ValueError("chunk text does not match its source offsets")

    sizes = [chunk.size for chunk in chunks]
    covered = _covered_length((chunk.start, chunk.end) for chunk in chunks)
    total_chunk_chars = sum(sizes)
    source_length = len(source)
    coverage_ratio = covered / source_length if source_length else 1.0
    duplication_ratio = (
        max(0, total_chunk_chars - covered) / covered if covered else 0.0
    )

    boundary_candidates = [chunk for chunk in chunks if chunk.end < source_length]
    if boundary_candidates:
        aligned = sum(
            chunk.text.endswith(preferred_boundaries) for chunk in boundary_candidates
        )
        boundary_alignment = aligned / len(boundary_candidates)
    else:
        boundary_alignment = 1.0

    return ChunkingMetrics(
        chunks=len(chunks),
        mean_size=fmean(sizes),
        median_size=float(median(sizes)),
        std_size=pstdev(sizes) if len(sizes) > 1 else 0.0,
        min_size=min(sizes),
        max_size=max(sizes),
        coverage_ratio=coverage_ratio,
        duplication_ratio=duplication_ratio,
        boundary_alignment_ratio=boundary_alignment,
    )


def evaluate_rankings(
    cases: Sequence[RetrievalCase],
    rankings: dict[str, Sequence[Chunk]],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    """Evaluate retrieval against source spans rather than chunk IDs."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if not cases:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    hits = 0
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []

    for case in cases:
        ranked = list(rankings.get(case.case_id, ()))
        top_k = ranked[:k]
        first_hit_rank = next(
            (
                rank
                for rank, chunk in enumerate(ranked, start=1)
                if any(
                    _overlaps((chunk.start, chunk.end), span)
                    for span in case.relevant_spans
                )
            ),
            None,
        )
        reciprocal_ranks.append(
            0.0 if first_hit_rank is None else 1.0 / first_hit_rank
        )

        matched_spans = sum(
            any(
                _overlaps((chunk.start, chunk.end), span)
                for chunk in top_k
            )
            for span in case.relevant_spans
        )
        recall = matched_spans / len(case.relevant_spans)
        recalls.append(recall)
        hits += int(matched_spans > 0)

    return RetrievalMetrics(
        hit_rate_at_k=hits / len(cases),
        mean_reciprocal_rank=fmean(reciprocal_ranks),
        mean_span_recall_at_k=fmean(recalls),
    )


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _covered_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0

    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total
