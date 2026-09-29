from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import EmbeddingProvider, cosine_similarity
from .length import LengthMetric, approximate_token_length
from .models import Chunk
from .recursive import RecursiveChunker


_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    text: str
    start: int
    end: int


def sentence_spans(text: str, max_sentence_chars: int = 400) -> list[SentenceSpan]:
    """Split mixed Chinese/English text while preserving exact source offsets."""

    if not text:
        return []
    spans: list[SentenceSpan] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        boundary = char in _SENTENCE_ENDINGS or char == "."
        too_long = i + 1 - start >= max_sentence_chars
        if boundary or too_long:
            end = i + 1
            if boundary:
                while end < len(text) and text[end].isspace():
                    end += 1
            if text[start:end].strip():
                spans.append(SentenceSpan(text[start:end], start, end))
            start = end
            i = end
            continue
        i += 1
    if start < len(text) and text[start:].strip():
        spans.append(SentenceSpan(text[start:], start, len(text)))
    return spans


class SemanticChunker:
    """Embedding-driven chunker with explicit, reproducible dependencies.

    Unlike the legacy implementation, this class never invents random vectors,
    never mutates its configured threshold while processing a document and never
    equates whitespace-separated words with tokens.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        chunk_size: int = 200,
        min_chunk_size: int = 40,
        similarity_threshold: float | None = None,
        breakpoint_percentile: float = 20.0,
        length_metric: LengthMetric = approximate_token_length,
        max_sentence_chars: int = 400,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if min_chunk_size < 0 or min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size must be in [0, chunk_size]")
        if similarity_threshold is not None and not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if not 0.0 <= breakpoint_percentile <= 100.0:
            raise ValueError("breakpoint_percentile must be in [0, 100]")
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.breakpoint_percentile = breakpoint_percentile
        self.length_metric = length_metric
        self.max_sentence_chars = max_sentence_chars

    def split(self, text: str) -> list[Chunk]:
        sentences = sentence_spans(text, self.max_sentence_chars)
        if not sentences:
            return []

        embeddings = np.asarray(
            self.embedding_provider.embed([s.text for s in sentences]), dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
            raise ValueError(
                f"embedding provider returned shape {embeddings.shape}; "
                f"expected ({len(sentences)}, dimension)"
            )

        similarities = [
            cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]
        threshold = self._threshold(similarities)
        raw = self._build_chunks(text, sentences, similarities, threshold)
        return self._merge_small_tail(text, raw)

    def _threshold(self, similarities: Sequence[float]) -> float:
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        if not similarities:
            return -1.0
        return float(np.percentile(np.asarray(similarities), self.breakpoint_percentile))

    def _build_chunks(
        self,
        text: str,
        sentences: list[SentenceSpan],
        similarities: list[float],
        threshold: float,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        group_start = 0
        for idx in range(1, len(sentences)):
            candidate_start = sentences[group_start].start
            candidate_end = sentences[idx].end
            over_budget = self.length_metric(text[candidate_start:candidate_end]) > self.chunk_size
            semantic_break = similarities[idx - 1] <= threshold
            if over_budget or semantic_break:
                chunks.extend(self._emit_group(text, sentences[group_start:idx], threshold))
                group_start = idx
        chunks.extend(self._emit_group(text, sentences[group_start:], threshold))
        return chunks

    def _emit_group(
        self, text: str, group: list[SentenceSpan], threshold: float
    ) -> list[Chunk]:
        if not group:
            return []
        start, end = group[0].start, group[-1].end
        group_text = text[start:end]
        if self.length_metric(group_text) <= self.chunk_size:
            return [
                Chunk(
                    group_text,
                    start,
                    end,
                    {
                        "algorithm": "semantic",
                        "sentence_count": len(group),
                        "length": self.length_metric(group_text),
                        "similarity_threshold": threshold,
                    },
                )
            ]

        fallback = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            length_metric=self.length_metric,
        )
        pieces = fallback.split(group_text)
        return [
            Chunk(
                piece.text,
                start + piece.start,
                start + piece.end,
                {
                    "algorithm": "semantic",
                    "fallback": "size-boundary",
                    "length": self.length_metric(piece.text),
                    "similarity_threshold": threshold,
                },
            )
            for piece in pieces
        ]

    def _merge_small_tail(self, text: str, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) < 2 or self.min_chunk_size == 0:
            return chunks
        last = chunks[-1]
        if self.length_metric(last.text) >= self.min_chunk_size:
            return chunks
        previous = chunks[-2]
        if previous.end != last.start:
            return chunks
        merged_text = text[previous.start:last.end]
        if self.length_metric(merged_text) > self.chunk_size:
            return chunks
        merged = Chunk(
            merged_text,
            previous.start,
            last.end,
            {
                "algorithm": "semantic",
                "merged_small_tail": True,
                "length": self.length_metric(merged_text),
                "similarity_threshold": previous.metadata.get("similarity_threshold"),
            },
        )
        return [*chunks[:-2], merged]
