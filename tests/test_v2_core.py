from __future__ import annotations

import numpy as np

from chunk_experiment import (
    HashEmbeddingProvider,
    RecursiveChunker,
    SemanticChunker,
    approximate_token_length,
    evaluate_chunks,
)
from chunk_experiment.embeddings import cosine_similarity


def test_approximate_token_length_handles_unsegmented_chinese() -> None:
    assert approximate_token_length("这是中文文本。") >= 6
    assert approximate_token_length("This is English.") == 4


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimension=64)
    first = provider.embed(["alpha beta", "alpha beta"])
    second = provider.embed(["alpha beta", "gamma"])
    assert np.allclose(first[0], first[1])
    assert np.allclose(first[0], second[0])
    assert np.isclose(np.linalg.norm(first[0]), 1.0)
    assert cosine_similarity(first[0], first[1]) > 0.99


def test_recursive_chunks_are_source_aligned_and_bounded() -> None:
    text = "第一段内容。第二段内容继续。\n\nThird paragraph has several words and details. " * 8
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(chunk.text == text[chunk.start:chunk.end] for chunk in chunks)
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    metrics = evaluate_chunks(text, chunks, max_chunk_size=80)
    assert metrics.coverage_ratio == 1.0
    assert metrics.duplicated_chars > 0
    assert metrics.size_compliance_ratio == 1.0


def test_semantic_chunks_are_reproducible_and_bounded() -> None:
    text = (
        "Cats like naps. Cats enjoy warm windows. Cats purr quietly. "
        "Databases store rows. SQL queries retrieve records. Indexes speed queries. "
        "猫喜欢晒太阳。猫也喜欢安静地睡觉。数据库用于保存结构化数据。"
    )
    provider = HashEmbeddingProvider(dimension=128)
    chunker = SemanticChunker(
        provider,
        chunk_size=20,
        min_chunk_size=0,
        breakpoint_percentile=40,
    )
    first = chunker.split(text)
    second = chunker.split(text)
    assert [(c.start, c.end, c.text) for c in first] == [
        (c.start, c.end, c.text) for c in second
    ]
    assert all(approximate_token_length(c.text) <= 20 for c in first)
    assert all(c.text == text[c.start:c.end] for c in first)


def test_evaluation_detects_no_overlap_for_semantic_chunks() -> None:
    text = "One sentence. Another sentence. 第三句。第四句。"
    chunks = SemanticChunker(
        HashEmbeddingProvider(),
        chunk_size=50,
        min_chunk_size=0,
        similarity_threshold=-1.0,
    ).split(text)
    metrics = evaluate_chunks(
        text,
        chunks,
        max_chunk_size=50,
        length_metric=approximate_token_length,
    )
    assert metrics.coverage_ratio == 1.0
    assert metrics.duplicated_chars == 0


def test_recursive_boundary_search_never_exceeds_budget() -> None:
    text = "第一段。第二段继续。\n\nThis is an English paragraph with several words. Another sentence."
    chunks = RecursiveChunker(chunk_size=40, chunk_overlap=5).split(text)
    assert all(len(chunk.text) <= 40 for chunk in chunks)
