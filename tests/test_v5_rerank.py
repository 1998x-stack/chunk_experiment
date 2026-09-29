from __future__ import annotations

from dataclasses import dataclass

import pytest

from chunk_experiment.models import Chunk
from chunk_experiment.rerank import (
    LexicalOverlapScoreProvider,
    ScoreProviderReranker,
    TwoStageRetriever,
)
from chunk_experiment.retrieval import IndexedChunk, SearchResult
from chunk_experiment.retrieval_eval import QueryCase, RelevantSpan
from chunk_experiment.two_stage_eval import evaluate_two_stage_retriever


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


class BadScoreProvider:
    def score(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        del query, texts
        return [1.0]


def _item(
    chunk_id: str,
    text: str,
    start: int,
    end: int,
) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk=Chunk(text, start, end),
    )


def test_lexical_overlap_score_prefers_query_coverage() -> None:
    provider = LexicalOverlapScoreProvider()
    scores = provider.score(
        "cuda gpu acceleration",
        (
            "General installation notes.",
            "CUDA GPU acceleration guide.",
        ),
    )
    assert scores[1] > scores[0]


def test_score_provider_reranker_reorders_and_resets_rank() -> None:
    irrelevant = _item(
        "irrelevant",
        "General package installation.",
        0,
        29,
    )
    relevant = _item(
        "relevant",
        "CUDA GPU acceleration guide.",
        29,
        57,
    )
    reranker = ScoreProviderReranker(
        LexicalOverlapScoreProvider()
    )
    reranked = reranker.rerank(
        "cuda gpu acceleration",
        (
            SearchResult(1, 9.0, irrelevant),
            SearchResult(2, 8.0, relevant),
        ),
        top_k=2,
    )
    assert reranked[0].item.chunk_id == "relevant"
    assert [row.rank for row in reranked] == [1, 2]


def test_score_provider_cardinality_is_validated() -> None:
    reranker = ScoreProviderReranker(BadScoreProvider())
    candidates = (
        SearchResult(
            1,
            1.0,
            _item("a", "alpha", 0, 5),
        ),
        SearchResult(
            2,
            0.9,
            _item("b", "beta", 5, 9),
        ),
    )
    with pytest.raises(ValueError, match="scores for"):
        reranker.rerank(
            "query",
            candidates,
            top_k=2,
        )


def test_two_stage_retriever_rejects_final_k_above_candidate_k() -> None:
    retriever = TwoStageRetriever(
        StubRetriever([]),
        ScoreProviderReranker(
            LexicalOverlapScoreProvider()
        ),
        candidate_k=2,
    )
    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search("query", top_k=3)


def test_two_stage_metrics_separate_candidate_recall_and_ranking_lift() -> None:
    irrelevant = _item(
        "irrelevant",
        "General installation notes.",
        0,
        27,
    )
    relevant = _item(
        "relevant",
        "CUDA GPU acceleration guide.",
        27,
        55,
    )
    candidate_retriever = StubRetriever(
        [
            SearchResult(1, 9.0, irrelevant),
            SearchResult(2, 8.0, relevant),
        ]
    )
    retriever = TwoStageRetriever(
        candidate_retriever,
        ScoreProviderReranker(
            LexicalOverlapScoreProvider()
        ),
        candidate_k=2,
    )
    case = QueryCase(
        "gpu",
        "cuda gpu acceleration",
        (
            RelevantSpan(
                "doc",
                27,
                55,
            ),
        ),
    )
    aggregate, rows = evaluate_two_stage_retriever(
        retriever,
        (case,),
        final_k=1,
    )
    assert aggregate.candidate_span_recall == 1.0
    assert aggregate.prefix_mrr == 0.0
    assert aggregate.final_mrr == 1.0
    assert aggregate.mean_mrr_lift == 1.0
    assert aggregate.mean_ndcg_lift == 1.0
    assert rows[0].candidate_latency_ms >= 0.0
    assert rows[0].rerank_latency_ms >= 0.0


def test_reranker_cannot_recover_missing_candidate_evidence() -> None:
    candidate_retriever = StubRetriever(
        [
            SearchResult(
                1,
                1.0,
                _item(
                    "irrelevant",
                    "General installation notes.",
                    0,
                    27,
                ),
            )
        ]
    )
    retriever = TwoStageRetriever(
        candidate_retriever,
        ScoreProviderReranker(
            LexicalOverlapScoreProvider()
        ),
        candidate_k=1,
    )
    case = QueryCase(
        "gpu",
        "cuda gpu acceleration",
        (
            RelevantSpan(
                "doc",
                40,
                50,
            ),
        ),
    )
    aggregate, _ = evaluate_two_stage_retriever(
        retriever,
        (case,),
        final_k=1,
    )
    assert aggregate.candidate_span_recall == 0.0
    assert aggregate.final_span_recall == 0.0
    assert aggregate.final_mrr == 0.0
