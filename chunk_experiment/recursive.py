from __future__ import annotations

from collections.abc import Sequence

from .length import LengthMetric, character_length
from .models import Chunk

DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ")


class RecursiveChunker:
    """Boundary-aware splitter with exact offsets and deterministic overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        *,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        length_metric: LengthMetric = character_length,
        boundary_search_ratio: float = 0.5,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        if not 0.0 <= boundary_search_ratio <= 1.0:
            raise ValueError("boundary_search_ratio must be in [0, 1]")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(s for s in separators if s)
        self.length_metric = length_metric
        self.boundary_search_ratio = boundary_search_ratio

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            hard_end = self._max_end(text, start, self.chunk_size)
            if hard_end >= len(text):
                end = len(text)
            else:
                end = self._prefer_boundary(text, start, hard_end)
                if end <= start:
                    end = hard_end

            chunks.append(
                Chunk(
                    text=text[start:end],
                    start=start,
                    end=end,
                    metadata={
                        "algorithm": "recursive",
                        "length": self.length_metric(text[start:end]),
                    },
                )
            )
            if end >= len(text):
                break

            next_start = self._overlap_start(text, start, end)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _max_end(self, text: str, start: int, budget: int) -> int:
        if self.length_metric(text[start:]) <= budget:
            return len(text)
        lo, hi = start + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.length_metric(text[start:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _prefer_boundary(self, text: str, start: int, hard_end: int) -> int:
        window_chars = max(1, hard_end - start)
        min_end = start + int(window_chars * self.boundary_search_ratio)
        best = -1
        for sep in self.separators:
            idx = text.rfind(sep, min_end, hard_end)
            if idx >= 0:
                candidate = idx + len(sep)
                if candidate > best:
                    best = candidate
        return best if best > start else hard_end

    def _overlap_start(self, text: str, chunk_start: int, chunk_end: int) -> int:
        if self.chunk_overlap == 0:
            return chunk_end
        lo, hi = chunk_start, chunk_end
        # Find the earliest suffix whose measured length is <= overlap.
        while lo < hi:
            mid = (lo + hi) // 2
            if self.length_metric(text[mid:chunk_end]) <= self.chunk_overlap:
                hi = mid
            else:
                lo = mid + 1
        return lo
