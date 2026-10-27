"""Tests for fair, span-grounded chunking experiments."""

from __future__ import annotations

import hashlib
import json

import pytest

from chunk_experiment.datasets import load_corpus, load_gold
from chunk_experiment.embeddings import HashEmbeddingProvider
from chunk_experiment.factory import ChunkerConfig, build_chunker
from chunk_experiment.models import Chunk
from chunk_experiment.retrieval import (
    DenseIndex,
    EvidenceSpan,
    QueryCase,
    SearchHit,
    evaluate_retrieval,
)
from experiments.run_factory import main as run_factory_main


def test_factory_validates_algorithm_and_shared_budget():
    config = ChunkerConfig(algorithm="recursive", chunk_size=12, length_unit="characters")
    chunks = build_chunker(config).split("alpha beta gamma delta")
    assert chunks
    assert all(len(chunk.text) <= 12 for chunk in chunks)
    with pytest.raises(ValueError):
        ChunkerConfig(algorithm="semantic-hash", chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError):
        ChunkerConfig(algorithm="semantic-http", chunk_size=10)
    with pytest.raises(ValueError):
        ChunkerConfig(algorithm="invalid", chunk_size=10)  # type: ignore[arg-type]


def test_evidence_recall_is_union_based_and_does_not_double_count():
    case = QueryCase("q1", "find numbers", (EvidenceSpan("doc1", 2, 8),))
    hits = [
        SearchHit("doc1", Chunk("2345", 2, 6), 0.9),
        SearchHit("doc1", Chunk("4567", 4, 8), 0.8),
    ]
    result = evaluate_retrieval(case, hits, k=2)
    assert result.hit_at_k == 1.0
    assert result.reciprocal_rank_at_k == 1.0
    assert result.evidence_char_recall_at_k == 1.0
    assert evaluate_retrieval(case, hits, k=1).evidence_char_recall_at_k == pytest.approx(
        4 / 6
    )
    with pytest.raises(ValueError):
        QueryCase("q1", "test", (EvidenceSpan("doc1", 0, 3),
                                 EvidenceSpan("doc1", 2, 5)))


def test_exact_dense_baseline_finds_lexical_match():
    provider = HashEmbeddingProvider(dimension=1024)
    items = [
        ("fruit", Chunk("apple banana", 0, 12)),
        ("car", Chunk("diesel engine", 0, 13)),
    ]
    result = DenseIndex(provider, items).search("apple banana", k=1)
    assert result[0].document_id == "fruit"


def _write_fixture(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    sources = {
        "doc1": "apple banana orange. Fruit market prices.",
        "doc2": "diesel engine car. Traffic information.",
    }
    manifest = tmp_path / "manifest.jsonl"
    records = []
    for document_id, text in sources.items():
        (root / f"{document_id}.txt").write_text(text, encoding="utf-8")
        records.append({
            "document_id": document_id, "path": f"{document_id}.txt",
            "split": "test", "language": "en", "domain": "fixture",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({
        "case_id": "q1", "query": "apple banana",
        "relevant_spans": [{"document_id": "doc1", "start": 0, "end": 12}],
    }) + "\n", encoding="utf-8")
    return root, manifest, gold


def test_corpus_checksum_and_gold_bounds(tmp_path):
    root, manifest, gold = _write_fixture(tmp_path)
    documents = load_corpus(manifest, root)
    assert len(documents) == 2
    assert load_gold(gold, documents)[0].case_id == "q1"
    (root / "doc1.txt").write_text("tampered data", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_corpus(manifest, root)


def test_factory_runner_generates_metrics_and_loguru_traces(tmp_path):
    root, manifest, gold = _write_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "seed": 42, "top_k": 2,
        "retriever": {"kind": "hash", "dimension": 1024},
        "chunkers": [
            {"algorithm": "recursive", "chunk_size": 30,
             "length_unit": "characters", "chunk_overlap": 5},
            {"algorithm": "semantic-hash", "chunk_size": 30,
             "length_unit": "characters", "min_chunk_size": 0},
        ],
    }), encoding="utf-8")
    output = tmp_path / "output"
    assert run_factory_main([
        "--manifest", str(manifest), "--dataset-root", str(root),
        "--gold", str(gold), "--config", str(config_path),
        "--output-dir", str(output),
    ]) == 0
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert len(report["results"]) == 2
    assert all(len(row["queries"]) == 1 for row in report["results"])
    assert all(row["summary"]["mean_hit_at_k"] >= 0 for row in report["results"])
    assert (output / "run.jsonl").is_file()
