from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from chunk_experiment.http_rerank import HttpRerankScoreProvider
from chunk_experiment.models import Chunk
from chunk_experiment.production import (
    CachedEmbeddingProvider,
    CachedRerankScoreProvider,
    ModelIdentity,
    UnitPricing,
)
from chunk_experiment.production_benchmark import (
    pareto_frontier,
    run_benchmark_phase,
)
from chunk_experiment.retrieval import IndexedChunk, SearchResult


class CountingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [
                [float(len(text)), 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class CountingRerankProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> Sequence[float]:
        self.calls.append((query, list(texts)))
        return [
            float(text.count(query))
            for text in texts
        ]


@dataclass
class StubRetriever:
    calls: int = 0

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls += 1
        text = query[:5].ljust(5, "_")
        item = IndexedChunk(
            chunk_id=f"q:{self.calls}",
            document_id="doc",
            chunk=Chunk(text, 0, 5),
        )
        return [SearchResult(1, 1.0, item)][:top_k]


def test_model_identity_fingerprint_is_stable_and_revision_sensitive() -> None:
    first = ModelIdentity(
        kind="embedding",
        provider="provider",
        model="model",
        revision="r1",
        endpoint_label="prod-us",
    )
    second = ModelIdentity(
        kind="embedding",
        provider="provider",
        model="model",
        revision="r1",
        endpoint_label="prod-us",
    )
    changed = ModelIdentity(
        kind="embedding",
        provider="provider",
        model="model",
        revision="r2",
        endpoint_label="prod-us",
    )
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != changed.fingerprint()
    assert len(first.fingerprint()) == 64


def test_cached_embedding_provider_deduplicates_backend_work() -> None:
    raw = CountingEmbeddingProvider()
    cached = CachedEmbeddingProvider(
        raw,
        identity=ModelIdentity(
            kind="embedding",
            provider="test",
            model="counting",
        ),
    )
    first = cached.embed(["alpha", "alpha", "beta"])
    second = cached.embed(["beta", "alpha"])
    assert first.shape == (3, 2)
    assert second.shape == (2, 2)
    assert raw.calls == [["alpha", "beta"]]
    usage = cached.usage()
    assert usage.requested_items == 5
    assert usage.backend_items == 2
    assert usage.cache_hits == 2
    assert usage.cache_misses == 3
    assert usage.backend_calls == 1


def test_cached_rerank_provider_caches_query_document_pairs() -> None:
    raw = CountingRerankProvider()
    cached = CachedRerankScoreProvider(
        raw,
        identity=ModelIdentity(
            kind="rerank",
            provider="test",
            model="counting",
        ),
    )
    first = cached.score(
        "gpu",
        ["gpu guide", "cpu guide", "gpu guide"],
    )
    second = cached.score(
        "gpu",
        ["gpu guide", "cpu guide"],
    )
    assert first == [1.0, 0.0, 1.0]
    assert second == [1.0, 0.0]
    assert raw.calls == [
        ("gpu", ["gpu guide", "cpu guide"])
    ]
    usage = cached.usage()
    assert usage.backend_items == 2
    assert usage.cache_hits == 2
    assert usage.cache_misses == 3


def test_unit_pricing_uses_backend_items_not_requested_items() -> None:
    embedding_raw = CountingEmbeddingProvider()
    embedding = CachedEmbeddingProvider(
        embedding_raw,
        identity=ModelIdentity(
            kind="embedding",
            provider="test",
            model="embedding",
        ),
    )
    embedding.embed(["same", "same"])
    rerank_raw = CountingRerankProvider()
    rerank = CachedRerankScoreProvider(
        rerank_raw,
        identity=ModelIdentity(
            kind="rerank",
            provider="test",
            model="rerank",
        ),
    )
    rerank.score("q", ["a", "b"])

    pricing = UnitPricing(
        embedding_usd_per_1k_items=2.0,
        rerank_usd_per_1k_pairs=3.0,
    )
    expected = 1 / 1000 * 2.0 + 2 / 1000 * 3.0
    assert pricing.estimate_usd(
        embedding_usage=embedding.usage(),
        rerank_usage=rerank.usage(),
    ) == pytest.approx(expected)


def test_http_rerank_parser_supports_scores_and_indexed_results() -> None:
    assert HttpRerankScoreProvider._parse_scores(
        [0.1, "0.9"],
        expected=2,
    ) == [0.1, 0.9]
    assert HttpRerankScoreProvider._parse_ranked_results(
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "score": 0.1},
        ],
        expected=2,
    ) == [0.1, 0.9]


def test_http_rerank_parser_rejects_duplicate_indexes() -> None:
    with pytest.raises(RuntimeError, match="duplicated"):
        HttpRerankScoreProvider._parse_ranked_results(
            [
                {"index": 0, "score": 0.1},
                {"index": 0, "score": 0.9},
            ],
            expected=2,
        )


def test_pareto_frontier_keeps_tradeoffs_and_removes_dominated_rows() -> None:
    rows = [
        {"id": "fast", "quality": 0.8, "latency": 5.0},
        {"id": "quality", "quality": 0.9, "latency": 10.0},
        {"id": "dominated", "quality": 0.7, "latency": 12.0},
    ]
    frontier = pareto_frontier(
        rows,
        maximize=("quality",),
        minimize=("latency",),
    )
    assert {row["id"] for row in frontier} == {
        "fast",
        "quality",
    }


def test_benchmark_phase_reports_qps_and_query_count() -> None:
    retriever = StubRetriever()
    phase = run_benchmark_phase(
        retriever,
        ("alpha", "beta"),
        top_k=1,
        phase="warm",
        passes=2,
    )
    assert phase.queries == 4
    assert retriever.calls == 4
    assert phase.qps > 0.0
    assert phase.mean_latency_ms >= 0.0
