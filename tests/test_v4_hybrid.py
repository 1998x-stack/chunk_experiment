from __future__ import annotations

from dataclasses import dataclass

from chunk_experiment.benchmark import evaluate_retriever_profiled
from chunk_experiment.embeddings import HashEmbeddingProvider
from chunk_experiment.hybrid import HybridRetriever
from chunk_experiment.length import character_length
from chunk_experiment.models import Chunk
from chunk_experiment.parent_child import (
    ParentChildRetriever,
    build_parent_child_index,
)
from chunk_experiment.recursive import RecursiveChunker
from chunk_experiment.retrieval import (
    Document,
    IndexedChunk,
    SearchResult,
)
from chunk_experiment.retrieval_eval import (
    QueryCase,
    RelevantSpan,
)
from chunk_experiment.retriever_factory import (
    RetrieverConfig,
    build_retriever,
)
from chunk_experiment.sparse import BM25Retriever, lexical_terms


@dataclass
class StubRetriever:
    results: list[SearchResult]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        del query
        return self.results[:top_k]


def _item(
    chunk_id: str,
    text: str,
    start: int,
    end: int,
) -> IndexedChunk:
    return IndexedChunk(
        chunk_id,
        "doc",
        Chunk(text, start, end),
    )


def test_lexical_terms_handles_mixed_cjk_and_english() -> None:
    assert lexical_terms("CUDA-compatible GPU 中文检索") == [
        "cuda",
        "compatible",
        "gpu",
        "中",
        "文",
        "检",
        "索",
    ]


def test_bm25_retriever_ranks_exact_lexical_match_first() -> None:
    chunks = (
        _item("a", "Apples are red.", 0, 15),
        _item("b", "Mangoes are tropical.", 15, 36),
        _item("c", "Database indexes help queries.", 36, 66),
    )
    results = BM25Retriever(chunks).search(
        "database indexes",
        top_k=3,
    )
    assert results[0].item.chunk_id == "c"
    assert results[0].score > results[1].score


def test_hybrid_retriever_uses_rrf_and_deduplicates() -> None:
    a = _item("a", "alpha", 0, 5)
    b = _item("b", "bravo", 5, 10)
    c = _item("c", "charl", 10, 15)
    left = StubRetriever(
        [
            SearchResult(1, 10.0, a),
            SearchResult(2, 9.0, b),
        ]
    )
    right = StubRetriever(
        [
            SearchResult(1, 3.0, b),
            SearchResult(2, 2.0, c),
        ]
    )
    results = HybridRetriever(
        (left, right),
        rrf_k=0,
        candidate_k=3,
    ).search("query", top_k=3)
    assert [result.item.chunk_id for result in results] == [
        "b",
        "a",
        "c",
    ]
    assert len({result.item.chunk_id for result in results}) == 3


def test_parent_child_retrieval_returns_larger_source_parent() -> None:
    text = (
        "Alpha details about astronomy. "
        "Telescope optics and mirrors are discussed here. "
        "Database indexing is a separate topic."
    )
    document = Document("doc", text)
    parent_chunker = RecursiveChunker(
        chunk_size=70,
        chunk_overlap=0,
        length_metric=character_length,
    )
    child_chunker = RecursiveChunker(
        chunk_size=30,
        chunk_overlap=0,
        length_metric=character_length,
    )
    index = build_parent_child_index(
        (document,),
        parent_chunker=parent_chunker,
        child_chunker=child_chunker,
    )
    child_retriever = BM25Retriever(index.children)
    retriever = ParentChildRetriever(
        child_retriever,
        index,
    )
    results = retriever.search(
        "telescope optics mirrors",
        top_k=1,
    )
    assert len(results) == 1
    parent = results[0].item.chunk
    assert "Telescope" in parent.text
    assert parent.text == text[parent.start : parent.end]
    assert parent.char_length > 30


def test_profiled_evaluation_reports_context_duplication() -> None:
    first = _item("a", "abcdefghij", 0, 10)
    second = _item("b", "fghijklmno", 5, 15)
    retriever = StubRetriever(
        [
            SearchResult(1, 1.0, first),
            SearchResult(2, 0.9, second),
        ]
    )
    case = QueryCase(
        "q",
        "answer",
        (RelevantSpan("doc", 0, 5),),
    )
    aggregates, rows, costs, per_query_cost = (
        evaluate_retriever_profiled(
            retriever,
            (case,),
            top_k=(1, 2),
            cost_k=2,
        )
    )
    assert aggregates[1].hit_rate == 1.0
    assert len(rows) == 2
    assert costs.mean_context_chars == 20.0
    assert costs.mean_unique_context_chars == 15.0
    assert costs.mean_duplication_ratio == 0.25
    assert per_query_cost[0].returned_chunks == 2


def test_bm25_factory_does_not_require_embedding_provider() -> None:
    chunks = (
        _item("a", "alpha database", 0, 14),
        _item("b", "beta telescope", 14, 28),
    )
    retriever = build_retriever(
        RetrieverConfig(mode="bm25"),
        chunks,
    )
    assert retriever.search(
        "telescope",
        top_k=1,
    )[0].item.chunk_id == "b"


def test_hybrid_factory_uses_dense_and_sparse_paths() -> None:
    chunks = (
        _item("a", "alpha database", 0, 14),
        _item("b", "beta telescope", 14, 28),
    )
    retriever = build_retriever(
        RetrieverConfig(
            mode="hybrid",
            candidate_k=2,
        ),
        chunks,
        embedding_provider=HashEmbeddingProvider(
            dimension=128,
        ),
    )
    results = retriever.search(
        "telescope",
        top_k=2,
    )
    assert results[0].item.chunk_id == "b"
