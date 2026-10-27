"""Regression tests for offset integrity and unsafe embedding outputs."""

import numpy as np
import pytest

from chunk_experiment.embeddings import cosine_similarity, l2_normalize
from chunk_experiment.evaluation import evaluate_chunks
from chunk_experiment.models import Chunk
from chunk_experiment.semantic import SemanticChunker, sentence_spans
from chunk_experiment.embeddings import HashEmbeddingProvider


@pytest.mark.parametrize("text", [
    "\nFirst sentence. Next sentence.",
    "  \n  中文段落。\n",
    "\n\n",
    "A. \n \nB.",
    "No punctuation with trailing spaces   ",
])
def test_sentence_spans_preserve_all_source_characters(text):
    spans = sentence_spans(text, max_sentence_chars=8)
    assert "".join(s.text for s in spans) == text
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert all(left.end == right.start for left, right in zip(spans, spans[1:]))


def test_semantic_chunking_keeps_whitespace_prefix():
    text = "\n\nHello world. Another sentence.\n"
    chunks = SemanticChunker(HashEmbeddingProvider(), min_chunk_size=0).split(text)
    assert "".join(c.text for c in chunks) == text
    assert evaluate_chunks(text, chunks, max_chunk_size=200).coverage_ratio == 1.0


@pytest.mark.parametrize("chunk", [
    Chunk("bc", 1, 3),
    Chunk("ab", 0, 2),
    Chunk("", 1, 1),
])
def test_evaluator_rejects_invalid_or_misaligned_chunks(chunk):
    if chunk.text == "ab":
        source = "xy"
    else:
        source = "a"
    with pytest.raises(ValueError):
        evaluate_chunks(source, [chunk], max_chunk_size=10)


def test_evaluator_detects_partial_coverage_without_falsely_passing():
    score = evaluate_chunks("abcdef", [Chunk("ab", 0, 2)], max_chunk_size=2)
    assert score.coverage_ratio == pytest.approx(2 / 6)


@pytest.mark.parametrize("bad", [
    np.array([[np.nan, 1.0]]),
    np.array([[np.inf, 1.0]]),
    np.array([[[1.0]]]),
])
def test_embedding_normalizer_rejects_invalid_arrays(bad):
    with pytest.raises(ValueError):
        l2_normalize(bad)


def test_cosine_rejects_nonfinite():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_invalid_sentence_size_rejected():
    with pytest.raises(ValueError):
        sentence_spans("test", max_sentence_chars=0)
    with pytest.raises(ValueError):
        SemanticChunker(HashEmbeddingProvider(), max_sentence_chars=0)
