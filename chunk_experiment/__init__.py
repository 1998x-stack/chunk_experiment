"""Reproducible chunking and retrieval evaluation primitives for RAG experiments."""

from .dataset import EvaluationDataset, load_evaluation_dataset
from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .evaluation import ChunkingMetrics, evaluate_chunks
from .length import approximate_token_length, character_length
from .markdown import MarkdownChunker, MarkdownSection, markdown_sections
from .models import Chunk
from .recursive import RecursiveChunker
from .retrieval import (
    DenseRetriever,
    Document,
    IndexedChunk,
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
    evaluate_query,
    evaluate_retriever,
)
from .semantic import SemanticChunker
from .strategy import StrategyConfig, build_strategy

__all__ = [
    "AggregateRetrievalMetrics",
    "Chunk",
    "ChunkingMetrics",
    "DenseRetriever",
    "Document",
    "EvaluationDataset",
    "HashEmbeddingProvider",
    "HttpEmbeddingProvider",
    "IndexedChunk",
    "MarkdownChunker",
    "MarkdownSection",
    "QueryCase",
    "RecursiveChunker",
    "RelevantSpan",
    "RetrievalGate",
    "RetrievalMetrics",
    "SearchResult",
    "SemanticChunker",
    "StrategyConfig",
    "approximate_token_length",
    "build_strategy",
    "character_length",
    "chunk_documents",
    "contextual_heading_text",
    "evaluate_chunks",
    "evaluate_query",
    "evaluate_retriever",
    "load_evaluation_dataset",
    "markdown_sections",
    "plain_chunk_text",
]
