from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .hybrid import HybridRetriever
from .retrieval import (
    ChunkRenderer,
    DenseRetriever,
    IndexedChunk,
    Retriever,
    plain_chunk_text,
)
from .sparse import BM25Retriever


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    mode: str = "dense"
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    candidate_k: int = 50

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retrieval mode: {self.mode}")
        if self.dense_weight <= 0:
            raise ValueError("dense_weight must be > 0")
        if self.sparse_weight <= 0:
            raise ValueError("sparse_weight must be > 0")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")


def build_retriever(
    config: RetrieverConfig,
    chunks: Sequence[IndexedChunk],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    renderer: ChunkRenderer = plain_chunk_text,
) -> Retriever:
    """Build dense, sparse or RRF-hybrid retrieval behind one interface."""

    if config.mode == "bm25":
        return BM25Retriever(chunks, renderer=renderer)

    if embedding_provider is None:
        raise ValueError(
            f"{config.mode} retrieval requires an embedding_provider"
        )

    dense = DenseRetriever(
        embedding_provider,
        chunks,
        renderer=renderer,
    )
    if config.mode == "dense":
        return dense

    sparse = BM25Retriever(
        chunks,
        renderer=renderer,
    )
    return HybridRetriever(
        (dense, sparse),
        weights=(
            config.dense_weight,
            config.sparse_weight,
        ),
        rrf_k=config.rrf_k,
        candidate_k=config.candidate_k,
    )
