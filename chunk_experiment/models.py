from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Chunk:
    """A source-aligned text chunk.

    ``start`` and ``end`` are character offsets into the original document.
    Keeping offsets first-class makes chunking auditable and lets callers
    reconstruct coverage/overlap without fuzzy string matching.
    """

    text: str
    start: int
    end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                "Chunk text length must equal end-start so offsets remain source-aligned"
            )

    @property
    def char_length(self) -> int:
        return self.end - self.start
