from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .embeddings import EmbeddingProvider, l2_normalize
from .models import Chunk


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    chunk: Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    score: float
    item: IndexedChunk


ChunkRenderer = Callable[[IndexedChunk], str]


def plain_chunk_text(item: IndexedChunk) -> str:
    return item.chunk.text


def contextual_heading_text(item: IndexedChunk) -> str:
    """Add Markdown heading breadcrumbs for embedding without changing source text."""

    path = item.chunk.metadata.get("heading_path")
    if not path:
        return item.chunk.text
    breadcrumb = " > ".join(str(part) for part in path)
    return f"{breadcrumb}\n{item.chunk.text}"


def chunk_documents(documents: Sequence[Document], chunker: Chunker) -> list[IndexedChunk]:
    indexed: list[IndexedChunk] = []
    for document in documents:
        for position, chunk in enumerate(chunker.split(document.text)):
            if chunk.text != document.text[chunk.start : chunk.end]:
                raise ValueError(
                    f"chunker violated source alignment for document {document.document_id!r}"
                )
            indexed.append(
                IndexedChunk(
                    chunk_id=f"{document.document_id}:{position}",
                    document_id=document.document_id,
                    chunk=chunk,
                )
            )
    return indexed


class DenseRetriever:
    """Exact dense retriever for controlled chunking experiments."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunks: Sequence[IndexedChunk],
        *,
        renderer: ChunkRenderer = plain_chunk_text,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chunks = tuple(chunks)
        self.renderer = renderer
        texts = [renderer(chunk) for chunk in self.chunks]
        if texts:
            vectors = np.asarray(embedding_provider.embed(texts), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts):
                raise ValueError(
                    f"embedding provider returned shape {vectors.shape}; "
                    f"expected ({len(texts)}, dimension)"
                )
            self.vectors = l2_normalize(vectors)
        else:
            self.vectors = np.empty((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.chunks:
            return []
        query_vectors = np.asarray(self.embedding_provider.embed([query]), dtype=np.float32)
        if query_vectors.ndim != 2 or query_vectors.shape[0] != 1:
            raise ValueError("embedding provider must return exactly one vector for a query")
        query_vector = l2_normalize(query_vectors)[0]
        if query_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query dimension {query_vector.shape[0]} does not match "
                f"index dimension {self.vectors.shape[1]}"
            )
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(rank=rank, score=float(scores[index]), item=self.chunks[index])
            for rank, index in enumerate(order, start=1)
        ]
