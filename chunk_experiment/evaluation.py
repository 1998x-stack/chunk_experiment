from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev

from .length import LengthMetric, character_length
from .models import Chunk


@dataclass(frozen=True, slots=True)
class ChunkingMetrics:
    chunks: int
    source_chars: int
    covered_chars: int
    duplicated_chars: int
    coverage_ratio: float
    duplication_ratio: float
    mean_length: float
    std_length: float
    max_length: int
    size_compliance_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_chunks(
    source: str,
    chunks: list[Chunk],
    *,
    max_chunk_size: int,
    length_metric: LengthMetric = character_length,
) -> ChunkingMetrics:
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")
    intervals = sorted((max(0, c.start), min(len(source), c.end)) for c in chunks)
    covered = 0
    cursor = 0
    for start, end in intervals:
        if end <= cursor:
            continue
        covered += end - max(start, cursor)
        cursor = max(cursor, end)

    lengths = [length_metric(c.text) for c in chunks]
    total_span = sum(max(0, end - start) for start, end in intervals)
    duplicated = max(0, total_span - covered)
    source_chars = len(source)
    return ChunkingMetrics(
        chunks=len(chunks),
        source_chars=source_chars,
        covered_chars=covered,
        duplicated_chars=duplicated,
        coverage_ratio=(covered / source_chars) if source_chars else 1.0,
        duplication_ratio=(duplicated / source_chars) if source_chars else 0.0,
        mean_length=mean(lengths) if lengths else 0.0,
        std_length=pstdev(lengths) if len(lengths) > 1 else 0.0,
        max_length=max(lengths, default=0),
        size_compliance_ratio=(
            sum(length <= max_chunk_size for length in lengths) / len(lengths)
            if lengths
            else 1.0
        ),
    )
