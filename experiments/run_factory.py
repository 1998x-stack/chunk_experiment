"""Reproducible source-span chunking benchmark (v3)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
from loguru import logger

from chunk_experiment.datasets import load_corpus, load_gold, sha256_bytes
from chunk_experiment.embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from chunk_experiment.evaluation import evaluate_chunks
from chunk_experiment.factory import ChunkerConfig, build_chunker, make_length_metric
from chunk_experiment.retrieval import DenseIndex, evaluate_retrieval


def _git_sha() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: dict) -> None:
    target = path.with_suffix(path.suffix + ".tmp")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    target.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk factory retrieval benchmark")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL corpus manifest")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path, help="JSONL span-labelled queries")
    parser.add_argument("--config", required=True, type=Path, help="JSON experiment matrix")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    return parser


def run_experiment(args: argparse.Namespace) -> dict:
    raw_config = args.config.read_bytes()
    settings = json.loads(raw_config)
    configs = [ChunkerConfig(**entry) for entry in settings["chunkers"]]
    if not configs:
        raise ValueError("at least one chunker configuration is required")
    if len({config.length_unit for config in configs}) != 1:
        raise ValueError("length units differ; use one length unit for a fair experiment")
    k = settings.get("top_k", 5)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("top_k must be a positive integer")
    seed = settings.get("seed", 42)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    np.random.seed(seed)
    docs = [doc for doc in load_corpus(args.manifest, args.dataset_root)
            if doc.split == args.split]
    if not docs:
        raise ValueError(f"no documents for split {args.split}")
    gold = load_gold(args.gold, docs, split=args.split)
    retriever_settings = settings.get("retriever", {"kind": "hash", "dimension": 256})
    if retriever_settings["kind"] == "hash":
        retrieval_provider = HashEmbeddingProvider(dimension=retriever_settings.get(
            "dimension", 256
        ))
        retrieval_identity = {"kind": "lexical-hash", "dimension": retrieval_provider.dimension,
                              "semantic_quality_claims": False}
    elif retriever_settings["kind"] == "http":
        url = os.getenv("CHUNK_RETRIEVER_EMBEDDING_URL")
        if not url:
            raise ValueError("set CHUNK_RETRIEVER_EMBEDDING_URL for http retrieval")
        retrieval_provider = HttpEmbeddingProvider(
            url,
            model=retriever_settings["model"],
            version=retriever_settings["version"],
        )
        retrieval_identity = {"kind": "http", "model": retriever_settings["model"],
                              "version": retriever_settings["version"]}
    else:
        raise ValueError("unsupported retriever kind")
    manifest_hash = sha256_bytes(args.manifest.read_bytes())
    gold_hash = sha256_bytes(args.gold.read_bytes())
    config_hash = hashlib.sha256(raw_config).hexdigest()
    run_id = f"factory-{config_hash[:12]}-{manifest_hash[:8]}-{gold_hash[:8]}"
    logger.bind(run_id=run_id).info(
        "benchmark_started documents={} gold_cases={} chunker_configs={} k={}",
        len(docs), len(gold), len(configs), k
    )
    results = []
    for config in configs:
        chunker = build_chunker(config)
        metric = make_length_metric(config.length_unit)
        chunk_items = []
        rows = []
        started = time.perf_counter()
        for doc in docs:
            start = time.perf_counter()
            chunks = chunker.split(doc.text)
            duration = time.perf_counter() - start
            metrics = evaluate_chunks(doc.text, chunks, max_chunk_size=config.chunk_size,
                                      length_metric=metric)
            if metrics.coverage_ratio < 1:
                raise ValueError(f"source coverage below 100% for {doc.document_id}")
            chunk_items.extend((doc.document_id, chunk) for chunk in chunks)
            rows.append({"document_id": doc.document_id, "sha256": doc.sha256,
                         "language": doc.language, "domain": doc.domain,
                         "elapsed_seconds": duration, "metrics": metrics.to_dict()})
            logger.bind(run_id=run_id, document_id=doc.document_id,
                        algorithm=config.algorithm).info(
                "chunked chars={} chunks={} latency_ms={:.2f}",
                len(doc.text), len(chunks), duration * 1000
            )
        construction_seconds = time.perf_counter() - started
        index_started = time.perf_counter()
        index = DenseIndex(retrieval_provider, chunk_items)
        indexing_seconds = time.perf_counter() - index_started
        queries = []
        for case in gold:
            t0 = time.perf_counter()
            hits = index.search(case.query, k=k)
            latency = time.perf_counter() - t0
            scores = evaluate_retrieval(case, hits, k=k)
            query_result = {"case_id": case.case_id, "query": case.query,
                            "metrics": scores.to_dict(), "elapsed_seconds": latency,
                            "top_k": [{"document_id": hit.document_id,
                                       "start": hit.chunk.start, "end": hit.chunk.end,
                                       "score": hit.score} for hit in hits]}
            queries.append(query_result)
            logger.bind(run_id=run_id, case_id=case.case_id,
                        algorithm=config.algorithm).info(
                "query hit={} mrr={:.4f} evidence_recall={:.4f} latency_ms={:.2f}",
                scores.hit_at_k, scores.reciprocal_rank_at_k,
                scores.evidence_char_recall_at_k, latency * 1000
            )
        results.append({
            "config": config.to_dict(),
            "documents": rows,
            "queries": queries,
            "summary": {
                "mean_hit_at_k": mean(q["metrics"]["hit_at_k"] for q in queries),
                "mean_mrr_at_k": mean(q["metrics"]["reciprocal_rank_at_k"] for q in queries),
                "mean_evidence_char_recall_at_k": mean(
                    q["metrics"]["evidence_char_recall_at_k"] for q in queries
                ),
                "chunk_count": len(chunk_items),
                "mean_duplication_ratio": mean(
                    r["metrics"]["duplication_ratio"] for r in rows
                ),
                "construction_seconds": construction_seconds,
                "indexing_seconds": indexing_seconds,
                "mean_query_seconds": mean(q["elapsed_seconds"] for q in queries),
            },
        })
    result = {
        "schema_version": 3, "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(), "seed": seed, "top_k": k, "split": args.split,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input_hashes": {"manifest_sha256": manifest_hash,
                         "gold_sha256": gold_hash, "config_sha256": config_hash},
        "documents": [asdict(doc) | {"text": None} for doc in docs],
        "retriever": retrieval_identity, "results": results,
        "interpretation_note": (
            "A lexical-hash retriever is only a deterministic plumbing baseline. "
            "Do not claim semantic retrieval improvements unless using a validated real model."
        ),
    }
    # Do not leak full raw corpus into reports; retain only metadata and checksums.
    for document in result["documents"]:
        document.pop("text")
    _atomic_json(args.output_dir / "results.json", result)
    logger.bind(run_id=run_id).info("benchmark_completed output={} methods={}",
                                   str(args.output_dir), len(results))
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(args.output_dir / "run.jsonl", serialize=True, enqueue=False,
                         rotation="10 MB", retention=3)
    try:
        result = run_experiment(args)
    finally:
        logger.remove(sink_id)
    print(f"{len(result['results'])} strategies -> {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
