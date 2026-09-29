"""Reproducible chunking and retrieval evaluation primitives for RAG experiments."""

from .benchmark import (
    AggregateCostMetrics,
    QueryCostMetrics,
    evaluate_retriever_profiled,
)
from .dataset import EvaluationDataset, load_evaluation_dataset
from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .evaluation import ChunkingMetrics, evaluate_chunks
from .hybrid import HybridRetriever
from .length import approximate_token_length, character_length
from .markdown import MarkdownChunker, MarkdownSection, markdown_sections
from .models import Chunk
from .parent_child import (
    ParentChildIndex,
    ParentChildRetriever,
    build_parent_child_index,
)
from .recursive import RecursiveChunker
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

__all__ = [
    "AggregateCostMetrics",
    "AggregateRetrievalMetrics",
    "BM25Retriever",
    "Chunk",
    "ChunkingMetrics",
    "DenseRetriever",
    "Document",
    "EvaluationDataset",
    "HashEmbeddingProvider",
    "HttpEmbeddingProvider",
    "HybridRetriever",
    "IndexedChunk",
    "MarkdownChunker",
    "MarkdownSection",
    "ParentChildIndex",
    "ParentChildRetriever",
    "QueryCase",
    "QueryCostMetrics",
    "RecursiveChunker",
    "RelevantSpan",
    "Retriever",
    "RetrieverConfig",
    "RetrievalGate",
    "RetrievalMetrics",
    "SearchResult",
    "SemanticChunker",
    "StrategyConfig",
    "aggregate_retrieval_metrics",
    "approximate_token_length",
    "build_parent_child_index",
    "build_retriever",
    "build_strategy",
    "character_length",
    "chunk_documents",
    "contextual_heading_text",
    "evaluate_chunks",
    "evaluate_query",
    "evaluate_retriever",
    "evaluate_retriever_profiled",
    "lexical_terms",
    "load_evaluation_dataset",
    "markdown_sections",
    "plain_chunk_text",
]
