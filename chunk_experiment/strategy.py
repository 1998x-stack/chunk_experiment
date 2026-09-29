from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .length import LengthMetric, approximate_token_length
from .markdown import MarkdownChunker
from .recursive import RecursiveChunker
from .retrieval import Chunker


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    name: str
    chunk_size: int = 300
    overlap: int = 0
    min_chunk_size: int = 40
    breakpoint_percentile: float = 20.0

    def __post_init__(self) -> None:
        if self.name not in {"recursive", "markdown", "semantic"}:
            raise ValueError(f"unknown chunking strategy: {self.name}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")


def build_strategy(
    config: StrategyConfig,
    *,
    length_metric: LengthMetric = approximate_token_length,
    embedding_provider: EmbeddingProvider | None = None,
) -> Chunker:
    """Build chunkers behind one comparable length-unit contract."""

    if config.name == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_metric=length_metric,
        )
    if config.name == "markdown":
        return MarkdownChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_metric=length_metric,
        )

    if embedding_provider is None:
        raise ValueError("semantic strategy requires an embedding_provider")

    from .semantic import SemanticChunker

    return SemanticChunker(
        embedding_provider,
        chunk_size=config.chunk_size,
        min_chunk_size=config.min_chunk_size,
        breakpoint_percentile=config.breakpoint_percentile,
        length_metric=length_metric,
    )
