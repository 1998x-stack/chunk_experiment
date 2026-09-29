from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .retrieval import ChunkRenderer, IndexedChunk, SearchResult, plain_chunk_text

_LEXICAL_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+",
    re.UNICODE,
)


def lexical_terms(text: str) -> list[str]:
    """Tokenize text for a deterministic dependency-free lexical baseline."""

    return [match.group(0).lower() for match in _LEXICAL_RE.finditer(text)]


class BM25Retriever:
    """Small exact BM25 implementation for controlled retrieval experiments."""

    def __init__(
        self,
        chunks: Sequence[IndexedChunk],
        *,
        renderer: ChunkRenderer = plain_chunk_text,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")

        self.chunks = tuple(chunks)
        self.renderer = renderer
        self.k1 = k1
        self.b = b
        self.term_counts = [
            Counter(lexical_terms(renderer(chunk)))
            for chunk in self.chunks
        ]
        self.doc_lengths = [
            sum(counts.values())
            for counts in self.term_counts
        ]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        self.document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        documents = len(self.chunks)
        frequency = self.document_frequency.get(term, 0)
        return math.log(
            1.0
            + (documents - frequency + 0.5)
            / (frequency + 0.5)
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.chunks:
            return []

        query_terms = tuple(dict.fromkeys(lexical_terms(query)))
        scores: list[tuple[float, int]] = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            length = self.doc_lengths[index]
            normalization = (
                1.0 - self.b
                + self.b * length / self.avg_doc_length
                if self.avg_doc_length
                else 1.0
            )
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * normalization
                score += self._idf(term) * numerator / denominator
            scores.append((score, index))

        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                rank=rank,
                score=float(score),
                item=self.chunks[index],
            )
            for rank, (score, index) in enumerate(
                scores[:top_k],
                start=1,
            )
        ]
