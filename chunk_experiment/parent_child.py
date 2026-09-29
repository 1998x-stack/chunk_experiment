from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .models import Chunk
from .retrieval import (
    Chunker,
    Document,
    IndexedChunk,
    Retriever,
    SearchResult,
)


@dataclass(frozen=True, slots=True)
class ParentChildIndex:
    parents: tuple[IndexedChunk, ...]
    children: tuple[IndexedChunk, ...]
    child_to_parent: Mapping[str, str]
    parent_by_id: Mapping[str, IndexedChunk]


def build_parent_child_index(
    documents: Sequence[Document],
    *,
    parent_chunker: Chunker,
    child_chunker: Chunker,
) -> ParentChildIndex:
    """Build child retrieval units mapped to larger source-aligned parent context."""

    parents: list[IndexedChunk] = []
    children: list[IndexedChunk] = []
    child_to_parent: dict[str, str] = {}
    parent_by_id: dict[str, IndexedChunk] = {}

    for document in documents:
        for parent_position, parent in enumerate(
            parent_chunker.split(document.text)
        ):
            if parent.text != document.text[parent.start : parent.end]:
                raise ValueError(
                    "parent chunker violated source alignment for "
                    f"document {document.document_id!r}"
                )
            parent_id = f"{document.document_id}:parent:{parent_position}"
            parent_item = IndexedChunk(
                chunk_id=parent_id,
                document_id=document.document_id,
                chunk=parent,
            )
            parents.append(parent_item)
            parent_by_id[parent_id] = parent_item

            for child_position, relative_child in enumerate(
                child_chunker.split(parent.text)
            ):
                absolute_start = parent.start + relative_child.start
                absolute_end = parent.start + relative_child.end
                child_text = document.text[
                    absolute_start:absolute_end
                ]
                if child_text != relative_child.text:
                    raise ValueError(
                        "child chunker violated source alignment inside "
                        f"parent {parent_id!r}"
                    )
                child_id = f"{parent_id}:child:{child_position}"
                child = Chunk(
                    text=child_text,
                    start=absolute_start,
                    end=absolute_end,
                    metadata={
                        **dict(relative_child.metadata),
                        "parent_chunk_id": parent_id,
                    },
                )
                children.append(
                    IndexedChunk(
                        chunk_id=child_id,
                        document_id=document.document_id,
                        chunk=child,
                    )
                )
                child_to_parent[child_id] = parent_id

    return ParentChildIndex(
        parents=tuple(parents),
        children=tuple(children),
        child_to_parent=MappingProxyType(child_to_parent),
        parent_by_id=MappingProxyType(parent_by_id),
    )


class ParentChildRetriever:
    """Retrieve small child chunks, then return deduplicated parent contexts."""

    def __init__(
        self,
        child_retriever: Retriever,
        index: ParentChildIndex,
        *,
        candidate_k: int | None = None,
    ) -> None:
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        self.child_retriever = child_retriever
        self.index = index
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.index.children:
            return []

        candidate_k = (
            self.candidate_k
            if self.candidate_k is not None
            else len(self.index.children)
        )
        candidate_k = max(candidate_k, top_k)
        child_results = self.child_retriever.search(
            query,
            top_k=candidate_k,
        )

        best_score: dict[str, float] = {}
        first_rank: dict[str, int] = {}
        for result in child_results:
            parent_id = self.index.child_to_parent.get(
                result.item.chunk_id
            )
            if parent_id is None:
                raise ValueError(
                    "child retriever returned chunk not present in "
                    "the parent-child index"
                )
            if parent_id not in first_rank:
                first_rank[parent_id] = result.rank
                best_score[parent_id] = result.score
            else:
                best_score[parent_id] = max(
                    best_score[parent_id],
                    result.score,
                )

        ordered = sorted(
            best_score,
            key=lambda parent_id: (
                first_rank[parent_id],
                -best_score[parent_id],
                parent_id,
            ),
        )
        return [
            SearchResult(
                rank=rank,
                score=best_score[parent_id],
                item=self.index.parent_by_id[parent_id],
            )
            for rank, parent_id in enumerate(
                ordered[:top_k],
                start=1,
            )
        ]
