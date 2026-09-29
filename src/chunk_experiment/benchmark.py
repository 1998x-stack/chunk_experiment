from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter

from .chunkers import Chunker
from .metrics import ChunkingMetrics, evaluate_chunking
from .models import Chunk


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    algorithm: str
    latency_ms: float
    deterministic: bool
    metrics: ChunkingMetrics

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


def run_benchmark(
    text: str,
    chunker: Chunker,
    *,
    repeats: int = 3,
) -> tuple[BenchmarkResult, list[Chunk]]:
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    timings: list[float] = []
    signatures: list[tuple[tuple[int, int, str], ...]] = []
    latest_chunks: list[Chunk] = []

    for _ in range(repeats):
        started = perf_counter()
        latest_chunks = chunker.split(text)
        timings.append((perf_counter() - started) * 1000.0)
        signatures.append(
            tuple((chunk.start, chunk.end, chunk.text) for chunk in latest_chunks)
        )

    algorithm = getattr(chunker, "name", chunker.__class__.__name__)
    result = BenchmarkResult(
        algorithm=algorithm,
        latency_ms=float(median(timings)),
        deterministic=all(
            signature == signatures[0] for signature in signatures[1:]
        ),
        metrics=evaluate_chunking(text, latest_chunks),
    )
    return result, latest_chunks
