"""Reproducible chunking and retrieval evaluation primitives for RAG experiments."""

from .benchmark import (
    AggregateCostMetrics,
    QueryCostMetrics,
    evaluate_retriever_profiled,
)
from .dataset import EvaluationDataset, load_evaluation_dataset
from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .evaluation import ChunkingMetrics, evaluate_chunks
from .http_rerank import HttpRerankScoreProvider
from .hybrid import HybridRetriever
from .length import approximate_token_length, character_length
from .markdown import MarkdownChunker, MarkdownSection, markdown_sections
from .models import Chunk
from .parent_child import (
    ParentChildIndex,
    ParentChildRetriever,
    build_parent_child_index,
)
from .production import (
    CachedEmbeddingProvider,
    CachedRerankScoreProvider,
    MemoryCache,
    ModelIdentity,
    ProviderUsage,
    UnitPricing,
)
from .production_benchmark import (
    BenchmarkPhase,
    ProductionBenchmark,
    benchmark_cold_warm,
    pareto_frontier,
    run_benchmark_phase,
)
from .recursive import RecursiveChunker
from .rerank import (
    LexicalOverlapScoreProvider,
    Reranker,
    RerankScoreProvider,
    ScoreProviderReranker,
    TwoStageRetriever,
    TwoStageTrace,
)
from .rerank_factory import RerankerConfig, build_reranker
from .retrieval import (
    DenseRetriever,
    Document,
    IndexedChunk,
    Retriever,
    SearchResult,
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from .retrieval_eval import (
    AggregateRetrievalMetrics,
    QueryCase,
    RelevantSpan,
    RetrievalGate,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    evaluate_query,
    evaluate_retriever,
)
from .retriever_factory import RetrieverConfig, build_retriever
from .semantic import SemanticChunker
from .sparse import BM25Retriever, lexical_terms
from .strategy import StrategyConfig, build_strategy
from .two_stage_eval import (
    AggregateTwoStageMetrics,
    TwoStageGate,
    TwoStageQueryMetrics,
    evaluate_two_stage_retriever,
)

__all__ = [
    "AggregateCostMetrics",
    "AggregateRetrievalMetrics",
    "AggregateTwoStageMetrics",
    "BM25Retriever",
    "BenchmarkPhase",
    "CachedEmbeddingProvider",
    "CachedRerankScoreProvider",
    "Chunk",
    "ChunkingMetrics",
    "DenseRetriever",
    "Document",
    "EvaluationDataset",
    "HashEmbeddingProvider",
    "HttpEmbeddingProvider",
    "HttpRerankScoreProvider",
    "HybridRetriever",
    "IndexedChunk",
    "LexicalOverlapScoreProvider",
    "MarkdownChunker",
    "MarkdownSection",
    "MemoryCache",
    "ModelIdentity",
    "ParentChildIndex",
    "ParentChildRetriever",
    "ProductionBenchmark",
    "ProviderUsage",
    "QueryCase",
    "QueryCostMetrics",
    "RecursiveChunker",
    "RelevantSpan",
    "Reranker",
    "RerankerConfig",
    "RerankScoreProvider",
    "Retriever",
    "RetrieverConfig",
    "RetrievalGate",
    "RetrievalMetrics",
    "ScoreProviderReranker",
    "SearchResult",
    "SemanticChunker",
    "StrategyConfig",
    "TwoStageGate",
    "TwoStageQueryMetrics",
    "TwoStageRetriever",
    "TwoStageTrace",
    "UnitPricing",
    "aggregate_retrieval_metrics",
    "approximate_token_length",
    "benchmark_cold_warm",
    "build_parent_child_index",
    "build_reranker",
    "build_retriever",
    "build_strategy",
    "character_length",
    "chunk_documents",
    "contextual_heading_text",
    "evaluate_chunks",
    "evaluate_query",
    "evaluate_retriever",
    "evaluate_retriever_profiled",
    "evaluate_two_stage_retriever",
    "lexical_terms",
    "load_evaluation_dataset",
    "markdown_sections",
    "pareto_frontier",
    "plain_chunk_text",
    "run_benchmark_phase",
]
