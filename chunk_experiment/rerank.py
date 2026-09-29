from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .retrieval import (
    ChunkRenderer,
    Retriever,
    SearchResult,
    plain_chunk_text,
)
from .sparse import lexical_terms


class RerankScoreProvider(Protocol):
    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> Sequence[float]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]: ...


class LexicalOverlapScoreProvider:
    """Deterministic lexical rerank baseline for CI and pipeline validation."""

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> Sequence[float]:
        query_terms = tuple(dict.fromkeys(lexical_terms(query)))
        if not query_terms:
            return [0.0 for _ in texts]

        query_set = set(query_terms)
        scores: list[float] = []
        for text in texts:
            document_terms = set(lexical_terms(text))
            matched = len(query_set & document_terms)
            coverage = matched / len(query_set)
            precision = (
                matched / len(document_terms)
                if document_terms
                else 0.0
            )
            scores.append(0.8 * coverage + 0.2 * precision)
        return scores


class ScoreProviderReranker:
    """Rerank a candidate set with a batched score provider."""

    def __init__(
        self,
        score_provider: RerankScoreProvider,
        *,
        renderer: ChunkRenderer = plain_chunk_text,
    ) -> None:
        self.score_provider = score_provider
        self.renderer = renderer

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not candidates:
            return []

        texts = [
            self.renderer(candidate.item)
            for candidate in candidates
        ]
        scores = list(
            self.score_provider.score(
                query,
                texts,
            )
        )
        if len(scores) != len(candidates):
            raise ValueError(
                "rerank score provider returned "
                f"{len(scores)} scores for {len(candidates)} candidates"
            )

        ranked = sorted(
            zip(
                candidates,
                scores,
                strict=True,
            ),
            key=lambda item: (
                -float(item[1]),
                item[0].rank,
                item[0].item.chunk_id,
            ),
        )
        return [
            SearchResult(
                rank=rank,
                score=float(score),
                item=candidate.item,
            )
            for rank, (candidate, score) in enumerate(
                ranked[:top_k],
                start=1,
            )
        ]


@dataclass(frozen=True, slots=True)
class TwoStageTrace:
    query: str
    candidate_k: int
    final_k: int
    candidates: tuple[SearchResult, ...]
    final_results: tuple[SearchResult, ...]
    candidate_latency_ms: float
    rerank_latency_ms: float


class TwoStageRetriever:
    """Candidate retrieval followed by explicit reranking."""

    def __init__(
        self,
        candidate_retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search_with_trace(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> TwoStageTrace:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > self.candidate_k:
            raise ValueError(
                "top_k cannot exceed candidate_k in two-stage retrieval"
            )

        started = time.perf_counter()
        candidates = self.candidate_retriever.search(
            query,
            top_k=self.candidate_k,
        )
        candidate_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        final_results = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
        rerank_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return TwoStageTrace(
            query=query,
            candidate_k=self.candidate_k,
            final_k=top_k,
            candidates=tuple(candidates),
            final_results=tuple(final_results),
            candidate_latency_ms=candidate_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return list(
            self.search_with_trace(
                query,
                top_k=top_k,
            ).final_results
        )
