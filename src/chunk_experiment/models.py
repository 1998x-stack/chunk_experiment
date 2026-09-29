from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A text chunk with source-faithful offsets."""

    text: str
    start: int
    end: int
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid chunk offsets")
        if len(self.text) != self.end - self.start:
            raise ValueError("chunk text length must match [start, end) offsets")

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 80
    separators: tuple[str, ...] = (
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ". ",
        "! ",
        "? ",
        "；",
        "; ",
        "，",
        ", ",
        " ",
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be > 0")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")
        if any(not separator for separator in self.separators):
            raise ValueError("separators must not contain empty strings")
