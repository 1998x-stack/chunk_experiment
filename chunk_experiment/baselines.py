"""Source-aligned deterministic baselines for chunking ablations."""

from __future__ import annotations

import re

from .length import LengthMetric, character_length
from .models import Chunk
from .recursive import RecursiveChunker

_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+[^\n]+")


class FixedWindowChunker:
    """Fixed measured-budget windows with optional measured overlap."""

    def __init__(
        self, chunk_size: int, chunk_overlap: int = 0,
        *, length_metric: LengthMetric = character_length,
    ) -> None:
        self._splitter = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=(),
            length_metric=length_metric,
        )

    def split(self, text: str) -> list[Chunk]:
        return [
            Chunk(chunk.text, chunk.start, chunk.end,
                  {"algorithm": "fixed", "length": chunk.metadata["length"]})
            for chunk in self._splitter.split(text)
        ]


class MarkdownHeadingChunker:
    """Respect ATX Markdown heading boundaries; fall back on recursive size splits.

    Operates on raw Markdown to keep exact source offsets. It deliberately does
    not claim to parse HTML, tables, fenced code or full CommonMark syntax.
    """

    def __init__(
        self, chunk_size: int, *, length_metric: LengthMetric = character_length,
    ) -> None:
        self.chunk_size = chunk_size
        self.length_metric = length_metric
        self._fallback = RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=0, length_metric=length_metric
        )

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []
        offsets = [0]
        offsets.extend(
            match.start() for match in _HEADING.finditer(text) if match.start() > 0
        )
        offsets.append(len(text))
        chunks = []
        for start, end in zip(offsets, offsets[1:]):
            section = text[start:end]
            if self.length_metric(section) <= self.chunk_size:
                pieces = [Chunk(section, 0, len(section))]
            else:
                pieces = self._fallback.split(section)
            for piece in pieces:
                chunks.append(
                    Chunk(piece.text, start + piece.start, start + piece.end,
                          {"algorithm": "markdown-heading",
                           "length": self.length_metric(piece.text)})
                )
        return chunks
