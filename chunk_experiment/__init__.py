"""Reproducible chunking primitives for RAG experiments."""

from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .evaluation import ChunkingMetrics, evaluate_chunks
from .factory import ChunkerConfig, build_chunker, make_length_metric
from .retrieval import DenseIndex, EvidenceSpan, QueryCase, RetrievalMetrics, evaluate_retrieval
from .length import approximate_token_length, character_length
from .models import Chunk
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

__all__ = [
    "Chunk",
    "ChunkerConfig",
    "DenseIndex",
    "EvidenceSpan",
    "QueryCase",
    "RetrievalMetrics",
    "build_chunker",
    "make_length_metric",
    "evaluate_retrieval",
    "ChunkingMetrics",
    "HashEmbeddingProvider",
    "HttpEmbeddingProvider",
    "RecursiveChunker",
    "SemanticChunker",
    "approximate_token_length",
    "character_length",
    "evaluate_chunks",
]
