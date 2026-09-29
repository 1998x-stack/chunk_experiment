from __future__ import annotations

import json

import pytest

from chunk_experiment.dataset import load_evaluation_dataset
from chunk_experiment.embeddings import HashEmbeddingProvider
from chunk_experiment.markdown import MarkdownChunker, markdown_sections
from chunk_experiment.models import Chunk
from chunk_experiment.retrieval import (
    DenseRetriever,
    Document,
    IndexedChunk,
    SearchResult,
    chunk_documents,
    contextual_heading_text,
)
from chunk_experiment.retrieval_eval import (
    QueryCase,
    RelevantSpan,
    RetrievalGate,
    evaluate_query,
    evaluate_retriever,
)
from chunk_experiment.strategy import StrategyConfig, build_strategy


def test_markdown_sections_ignore_headings_inside_fences() -> None:
    text = "# Root\nIntro.\n```python\n# not a heading\n```\n## Child\nDetails.\n"
    sections = markdown_sections(text)
    assert [section.heading_path for section in sections] == [
        ("Root",),
        ("Root", "Child"),
    ]
    assert all(text[section.start : section.end] for section in sections)


def test_markdown_chunker_preserves_offsets_and_heading_context() -> None:
    text = (
        "# Install\nUse pip to install the package.\n\n"
        "## GPU\nGPU installation needs a CUDA-compatible build. "
        "Check the compatibility table before installation.\n"
    )
    chunker = MarkdownChunker(chunk_size=12, chunk_overlap=0)
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    assert all(
        chunk.text == text[chunk.start : chunk.end]
        for chunk in chunks
    )
    gpu = next(
        chunk
        for chunk in chunks
        if "GPU" in chunk.metadata.get("heading_path", ())
    )
    item = IndexedChunk("doc:0", "doc", gpu)
    rendered = contextual_heading_text(item)
    assert rendered.startswith("Install > GPU\n")
    assert gpu.text in rendered


def test_dense_retriever_finds_matching_chunk() -> None:
    documents = (
        Document(
            "fruit",
            "Apples are red. Mangoes are tropical fruit.",
        ),
        Document(
            "db",
            "Databases store rows. Indexes accelerate database queries.",
        ),
    )
    chunker = build_strategy(
        StrategyConfig(
            "recursive",
            chunk_size=8,
            overlap=0,
        )
    )
    chunks = chunk_documents(documents, chunker)
    retriever = DenseRetriever(
        HashEmbeddingProvider(dimension=1024),
        chunks,
    )
    results = retriever.search("mangoes", top_k=3)
    assert results[0].item.document_id == "fruit"
    assert "Mangoes" in results[0].item.chunk.text


def test_retrieval_metrics_deduplicate_gold_spans_for_ndcg() -> None:
    span = RelevantSpan("doc", 10, 20, relevance=2)
    case = QueryCase("q", "answer", (span,))
    first = IndexedChunk(
        "doc:0",
        "doc",
        Chunk("abcdefghij", 10, 20),
    )
    duplicate = IndexedChunk(
        "doc:1",
        "doc",
        Chunk("fghijklmno", 15, 25),
    )
    results = [
        SearchResult(1, 1.0, first),
        SearchResult(2, 0.9, duplicate),
    ]
    metrics = evaluate_query(case, results, k=2)
    assert metrics.hit_rate == 1.0
    assert metrics.span_recall == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.ndcg == 1.0


def test_evaluate_retriever_reports_multiple_k_values() -> None:
    text = (
        "Alpha topic contains telescope details. "
        "Beta topic explains databases."
    )
    document = Document("doc", text)
    telescope_start = text.index("telescope")
    database_start = text.index("databases")
    cases = (
        QueryCase(
            "q1",
            "telescope",
            (
                RelevantSpan(
                    "doc",
                    telescope_start,
                    telescope_start + 9,
                ),
            ),
        ),
        QueryCase(
            "q2",
            "databases",
            (
                RelevantSpan(
                    "doc",
                    database_start,
                    database_start + 9,
                ),
            ),
        ),
    )
    chunks = chunk_documents(
        (document,),
        build_strategy(
            StrategyConfig(
                "recursive",
                chunk_size=7,
            )
        ),
    )
    retriever = DenseRetriever(
        HashEmbeddingProvider(dimension=1024),
        chunks,
    )
    aggregates, rows = evaluate_retriever(
        retriever,
        cases,
        top_k=(1, 3),
    )
    assert set(aggregates) == {1, 3}
    assert len(rows) == 4
    assert aggregates[3].hit_rate >= aggregates[1].hit_rate
    assert aggregates[3].span_recall >= aggregates[1].span_recall


def test_dataset_manifest_is_fingerprinted_and_validated(
    tmp_path,
) -> None:
    text = "# Topic\nThe answer is cobalt.\n"
    doc = tmp_path / "doc.md"
    doc.write_text(text, encoding="utf-8")
    start = text.index("cobalt")
    manifest = {
        "schema_version": 1,
        "dataset_id": "mini",
        "documents": [
            {
                "id": "doc",
                "path": "doc.md",
            }
        ],
        "queries": [
            {
                "id": "q1",
                "query": "What is the answer? cobalt",
                "relevant_spans": [
                    {
                        "document_id": "doc",
                        "start": start,
                        "end": start + len("cobalt"),
                    }
                ],
            }
        ],
    }
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    first = load_evaluation_dataset(path)
    second = load_evaluation_dataset(path)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64

    manifest["queries"][0]["relevant_spans"][0]["end"] = 999
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds"):
        load_evaluation_dataset(path)


def test_retrieval_gate_reports_regressions() -> None:
    text = "One telescope fact. Another unrelated fact."
    document = Document("doc", text)
    start = text.index("telescope")
    case = QueryCase(
        "q",
        "telescope",
        (
            RelevantSpan(
                "doc",
                start,
                start + 9,
            ),
        ),
    )
    chunks = chunk_documents(
        (document,),
        build_strategy(
            StrategyConfig(
                "recursive",
                chunk_size=5,
            )
        ),
    )
    retriever = DenseRetriever(
        HashEmbeddingProvider(dimension=1024),
        chunks,
    )
    aggregates, _ = evaluate_retriever(
        retriever,
        (case,),
        top_k=(1,),
    )
    assert RetrievalGate(
        min_hit_rate=1.0,
    ).failures(aggregates[1]) == []

    failures = RetrievalGate(
        min_ndcg=1.01,
    ).failures(aggregates[1])
    assert failures
    assert failures[0].startswith("ndcg=")
