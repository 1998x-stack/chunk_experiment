from __future__ import annotations

from dataclasses import dataclass

from .rerank import (
    LexicalOverlapScoreProvider,
    Reranker,
    ScoreProviderReranker,
)
from .retrieval import ChunkRenderer, plain_chunk_text


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    mode: str = "none"
    candidate_k: int = 20

    def __post_init__(self) -> None:
        if self.mode not in {"none", "lexical"}:
            raise ValueError(f"unknown reranker mode: {self.mode}")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")


def build_reranker(
    config: RerankerConfig,
    *,
    renderer: ChunkRenderer = plain_chunk_text,
) -> Reranker | None:
    if config.mode == "none":
        return None
    return ScoreProviderReranker(
        LexicalOverlapScoreProvider(),
        renderer=renderer,
    )
