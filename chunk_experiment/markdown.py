from __future__ import annotations

import re
from dataclasses import dataclass

from .length import LengthMetric, approximate_token_length
from .models import Chunk
from .recursive import RecursiveChunker

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$")
_FENCE_RE = re.compile(r"^[ \t]*(```+|~~~+)")


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    start: int
    end: int
    level: int
    title: str | None
    heading_path: tuple[str, ...]


def markdown_sections(text: str) -> list[MarkdownSection]:
    """Parse heading-delimited Markdown sections while preserving source offsets.

    Headings inside fenced code blocks are ignored. Preamble text before the first
    heading is represented as a section with an empty heading path.
    """

    if not text:
        return []

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    sections: list[MarkdownSection] = []
    stack: list[tuple[int, str]] = []
    current_start = 0
    current_level = 0
    current_title: str | None = None
    current_path: tuple[str, ...] = ()
    seen_heading = False
    fence_token: str | None = None

    for line, line_start in zip(lines, offsets, strict=True):
        fence = _FENCE_RE.match(line)
        if fence:
            token = fence.group(1)
            if fence_token is None:
                fence_token = token[0] * 3
            elif token.startswith(fence_token[0]):
                fence_token = None
            continue
        if fence_token is not None:
            continue

        heading = _HEADING_RE.match(line)
        if not heading:
            continue

        level = len(heading.group(1))
        title = heading.group(2).strip()
        if line_start > current_start:
            sections.append(
                MarkdownSection(
                    start=current_start,
                    end=line_start,
                    level=current_level,
                    title=current_title,
                    heading_path=current_path,
                )
            )
        elif seen_heading and line_start == current_start:
            pass

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current_start = line_start
        current_level = level
        current_title = title
        current_path = tuple(item[1] for item in stack)
        seen_heading = True

    if current_start < len(text):
        sections.append(
            MarkdownSection(
                start=current_start,
                end=len(text),
                level=current_level,
                title=current_title,
                heading_path=current_path,
            )
        )

    return [section for section in sections if section.end > section.start]


class MarkdownChunker:
    """Structure-aware Markdown chunker with exact source alignment."""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 0,
        *,
        length_metric: LengthMetric = approximate_token_length,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_metric = length_metric

    def split(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in markdown_sections(text):
            section_text = text[section.start : section.end]
            metadata = {
                "algorithm": "markdown",
                "heading_path": section.heading_path,
                "heading_level": section.level,
                "heading": section.title,
            }
            if self.length_metric(section_text) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        section_text,
                        section.start,
                        section.end,
                        {**metadata, "length": self.length_metric(section_text)},
                    )
                )
                continue

            fallback = RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_metric=self.length_metric,
            )
            for piece in fallback.split(section_text):
                chunks.append(
                    Chunk(
                        piece.text,
                        section.start + piece.start,
                        section.start + piece.end,
                        {
                            **metadata,
                            "length": self.length_metric(piece.text),
                            "fallback": "recursive-within-section",
                        },
                    )
                )
        return chunks
