"""Explicit, validated factory for reproducible chunker experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from .embeddings import EmbeddingProvider, HashEmbeddingProvider, HttpEmbeddingProvider
from .length import LengthMetric, approximate_token_length, character_length
from .models import Chunk
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

Algorithm = Literal["recursive", "semantic-hash", "semantic-http", "semantic-provider"]
LengthUnit = Literal["characters", "approx_tokens"]


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    algorithm: Algorithm
    chunk_size: int = 500
    length_unit: LengthUnit = "characters"
    chunk_overlap: int = 0
    min_chunk_size: int = 0
    breakpoint_percentile: float = 20.0
    similarity_threshold: float | None = None
    embedding_url: str | None = None
    embedding_model: str = "m3e"
    embedding_version: str = "m3e"
    hash_dimension: int = 256

    def __post_init__(self) -> None:
        if self.algorithm not in (
            "recursive", "semantic-hash", "semantic-http", "semantic-provider"
        ):
            raise ValueError(f"unsupported chunker algorithm: {self.algorithm}")
        if self.length_unit not in ("characters", "approx_tokens"):
            raise ValueError(f"unsupported length unit: {self.length_unit}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if not 0 <= self.min_chunk_size <= self.chunk_size:
            raise ValueError("min_chunk_size must be in [0, chunk_size]")
        if not 0 <= self.breakpoint_percentile <= 100:
            raise ValueError("breakpoint_percentile must be in [0, 100]")
        if self.similarity_threshold is not None and not -1 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if self.hash_dimension < 8:
            raise ValueError("hash_dimension must be at least 8")
        if self.algorithm == "semantic-http" and not self.embedding_url:
            raise ValueError("semantic-http requires embedding_url")
        if self.algorithm != "recursive" and self.chunk_overlap:
            raise ValueError("semantic algorithms do not support chunk_overlap")

    def to_dict(self) -> dict:
        config = asdict(self)
        # Endpoint credentials/hosts must not become part of published experiment reports.
        config["embedding_url"] = "[configured]" if self.embedding_url else None
        return config


def make_length_metric(unit: LengthUnit) -> LengthMetric:
    if unit == "characters":
        return character_length
    if unit == "approx_tokens":
        return approximate_token_length
    raise ValueError(f"unsupported length unit: {unit}")


def build_chunker(
    config: ChunkerConfig, *, embedding_provider: EmbeddingProvider | None = None
) -> Chunker:
    """Create a fresh chunker; no mutable instance is reused across runs."""
    metric = make_length_metric(config.length_unit)
    if config.algorithm == "recursive":
        if embedding_provider is not None:
            raise ValueError("recursive chunker does not use an embedding provider")
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_metric=metric,
        )
    if config.algorithm == "semantic-hash":
        if embedding_provider is not None:
            raise ValueError("semantic-hash uses a fixed lexical provider")
        embedding_provider = HashEmbeddingProvider(dimension=config.hash_dimension)
    elif config.algorithm == "semantic-http":
        if embedding_provider is not None:
            raise ValueError("semantic-http uses its configured HTTP provider")
        embedding_provider = HttpEmbeddingProvider(
            config.embedding_url or "",
            model=config.embedding_model,
            version=config.embedding_version,
        )
    elif embedding_provider is None:
        raise ValueError("semantic-provider requires an explicitly supplied embedding provider")
    return SemanticChunker(
        embedding_provider,
        chunk_size=config.chunk_size,
        min_chunk_size=config.min_chunk_size,
        breakpoint_percentile=config.breakpoint_percentile,
        similarity_threshold=config.similarity_threshold,
        length_metric=metric,
    )
