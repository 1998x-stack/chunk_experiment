from __future__ import annotations

from collections.abc import Sequence

from .retrieval import IndexedChunk, Retriever, SearchResult


class HybridRetriever:
    """Fuse retriever rankings with weighted Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")

        resolved_weights = (
            tuple(weights)
            if weights is not None
            else tuple(1.0 for _ in retrievers)
        )
        if len(resolved_weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        if any(weight <= 0 for weight in resolved_weights):
            raise ValueError("all weights must be > 0")

        self.retrievers = tuple(retrievers)
        self.weights = resolved_weights
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        candidate_limit = max(top_k, self.candidate_k)
        fused_scores: dict[str, float] = {}
        items: dict[str, IndexedChunk] = {}
        first_seen: dict[str, int] = {}
        sequence = 0

        for retriever, weight in zip(
            self.retrievers,
            self.weights,
            strict=True,
        ):
            results = retriever.search(
                query,
                top_k=candidate_limit,
            )
            for result in results:
                chunk_id = result.item.chunk_id
                if chunk_id not in first_seen:
                    first_seen[chunk_id] = sequence
                    sequence += 1
                items[chunk_id] = result.item
                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + weight / (self.rrf_k + result.rank)
                )

        ordered = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                first_seen[chunk_id],
                chunk_id,
            ),
        )
        return [
            SearchResult(
                rank=rank,
                score=fused_scores[chunk_id],
                item=items[chunk_id],
            )
            for rank, chunk_id in enumerate(
                ordered[:top_k],
                start=1,
            )
        ]
