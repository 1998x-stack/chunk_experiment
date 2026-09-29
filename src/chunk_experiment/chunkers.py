from __future__ import annotations

from typing import Protocol

from .models import Chunk, ChunkingConfig


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


class BoundaryAwareChunker:
    """Deterministic source-faithful chunking with prioritized natural boundaries."""

    name = "boundary-aware"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(self, text: str) -> list[Chunk]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        ordinal = 0
        text_length = len(text)

        while start < text_length:
            end = self._choose_end(text, start)
            chunks.append(
                Chunk(
                    text=text[start:end],
                    start=start,
                    end=end,
                    ordinal=ordinal,
                    metadata={"chunker": self.name},
                )
            )
            ordinal += 1
            if end >= text_length:
                break

            next_start = end - self.config.chunk_overlap
            start = max(start + 1, next_start)

        return chunks

    def _choose_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.config.chunk_size, len(text))
        if hard_end >= len(text):
            return len(text)

        minimum_width = max(self.config.min_chunk_size, self.config.chunk_overlap + 1)
        search_start = min(start + minimum_width, hard_end)

        for separator in self.config.separators:
            index = text.rfind(separator, search_start, hard_end + 1)
            if index >= 0:
                candidate = index + len(separator)
                if candidate > start:
                    return candidate

        return hard_end
