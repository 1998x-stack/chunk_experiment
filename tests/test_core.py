from __future__ import annotations

import math

import pytest

from chunk_experiment import (
    BoundaryAwareChunker,
    Chunk,
    ChunkingConfig,
    RetrievalCase,
    evaluate_chunking,
    evaluate_rankings,
    run_benchmark,
)


def test_multilingual_chunks_are_exact_source_slices_and_respect_hard_limit() -> None:
    text = (
        "第一段讨论检索增强生成。它需要稳定的文本边界。\n\n"
        "Second paragraph explains retrieval evaluation. It also checks overlap. "
        "最后一段回到中文，并验证中英文混合文本。"
    )
    config = ChunkingConfig(chunk_size=44, chunk_overlap=8, min_chunk_size=18)
    chunks = BoundaryAwareChunker(config).split(text)

    assert len(chunks) >= 2
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert all(chunk.size <= config.chunk_size for chunk in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert all(left.start < right.start for left, right in zip(chunks, chunks[1:]))


def test_overlap_is_represented_by_offsets_without_text_rewriting() -> None:
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    config = ChunkingConfig(
        chunk_size=12,
        chunk_overlap=4,
        min_chunk_size=8,
        separators=("|",),
    )
    chunks = BoundaryAwareChunker(config).split(text)

    assert chunks[1].start == chunks[0].end - 4
    assert chunks[0].text[-4:] == chunks[1].text[:4]
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)


def test_chunking_metrics_measure_full_coverage_and_duplication() -> None:
    text = "a" * 30
    chunks = BoundaryAwareChunker(
        ChunkingConfig(
            chunk_size=12,
            chunk_overlap=3,
            min_chunk_size=8,
            separators=("|",),
        )
    ).split(text)
    metrics = evaluate_chunking(text, chunks)

    assert metrics.coverage_ratio == 1.0
    assert metrics.duplication_ratio > 0.0
    assert metrics.max_size <= 12


def test_span_based_retrieval_metrics_do_not_depend_on_chunk_ids() -> None:
    cases = [
        RetrievalCase("q1", ((10, 14),)),
        RetrievalCase("q2", ((40, 45), (60, 63))),
    ]
    rankings = {
        "q1": [Chunk("abcd", 10, 14, 99)],
        "q2": [Chunk("xxxxx", 0, 5, 1), Chunk("zzzzz", 40, 45, 2)],
    }

    metrics = evaluate_rankings(cases, rankings, k=2)
    assert metrics.hit_rate_at_k == 1.0
    assert math.isclose(metrics.mean_reciprocal_rank, 0.75)
    assert math.isclose(metrics.mean_span_recall_at_k, 0.75)


def test_benchmark_checks_determinism() -> None:
    text = "A sentence. Another sentence. 最后一句。"
    result, chunks = run_benchmark(
        text,
        BoundaryAwareChunker(
            ChunkingConfig(chunk_size=18, chunk_overlap=3, min_chunk_size=8)
        ),
        repeats=3,
    )

    assert chunks
    assert result.deterministic is True
    assert result.metrics.coverage_ratio == 1.0
    assert result.latency_ms >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"chunk_size": 10, "chunk_overlap": -1},
        {"chunk_size": 10, "min_chunk_size": 11},
        {"separators": ("",)},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
