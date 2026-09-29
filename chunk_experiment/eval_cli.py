from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .dataset import load_evaluation_dataset
from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .retrieval import (
    DenseRetriever,
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from .retrieval_eval import RetrievalGate, evaluate_retriever
from .strategy import StrategyConfig, build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate chunking strategies against a versioned retrieval golden set"
    )
    parser.add_argument("dataset", type=Path, help="Evaluation manifest JSON")
    parser.add_argument(
        "--chunker",
        choices=("recursive", "markdown", "semantic"),
        default="recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--breakpoint-percentile", type=float, default=20.0)
    parser.add_argument("--retriever", choices=("hash", "http"), default="hash")
    parser.add_argument("--embedding-url")
    parser.add_argument("--contextual-headings", action="store_true")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--gate-k", type=int)
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-span-recall", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-ndcg", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retrieval_eval_v3.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_evaluation_dataset(args.dataset)

    if args.retriever == "hash":
        provider = HashEmbeddingProvider()
        provider_label = "deterministic-lexical-hash"
    else:
        if not args.embedding_url:
            raise SystemExit("--embedding-url is required for --retriever http")
        provider = HttpEmbeddingProvider(args.embedding_url)
        provider_label = "http"

    config = StrategyConfig(
        name=args.chunker,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
        breakpoint_percentile=args.breakpoint_percentile,
    )
    chunker = build_strategy(
        config,
        embedding_provider=provider,
    )
    chunks = chunk_documents(
        dataset.documents,
        chunker,
    )
    renderer = (
        contextual_heading_text
        if args.contextual_headings
        else plain_chunk_text
    )
    retriever = DenseRetriever(
        provider,
        chunks,
        renderer=renderer,
    )
    aggregates, per_query = evaluate_retriever(
        retriever,
        dataset.queries,
        top_k=args.top_k,
    )

    payload = {
        "schema_version": 1,
        "dataset": {
            "id": dataset.dataset_id,
            "fingerprint": dataset.fingerprint,
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
        },
        "strategy": {
            "name": config.name,
            "chunk_size": config.chunk_size,
            "overlap": config.overlap,
            "min_chunk_size": config.min_chunk_size,
            "breakpoint_percentile": config.breakpoint_percentile,
        },
        "retrieval": {
            "provider": provider_label,
            "contextual_headings": args.contextual_headings,
            "chunks": len(chunks),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "metrics": {
            str(k): row.to_dict()
            for k, row in aggregates.items()
        },
        "per_query": [row.to_dict() for row in per_query],
        "notes": (
            [
                "hash retrieval is a deterministic lexical smoke baseline; "
                "use a documented real embedding backend for semantic conclusions"
            ]
            if args.retriever == "hash"
            else []
        ),
    }
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"evaluated {len(dataset.queries)} queries over "
        f"{len(chunks)} chunks -> {args.output}"
    )

    gate = RetrievalGate(
        min_hit_rate=args.min_hit_rate,
        min_span_recall=args.min_span_recall,
        min_mrr=args.min_mrr,
        min_ndcg=args.min_ndcg,
    )
    has_gate = any(
        value is not None
        for value in (
            args.min_hit_rate,
            args.min_span_recall,
            args.min_mrr,
            args.min_ndcg,
        )
    )
    if has_gate:
        gate_k = (
            args.gate_k
            if args.gate_k is not None
            else max(aggregates)
        )
        if gate_k not in aggregates:
            raise SystemExit(
                f"--gate-k {gate_k} must be included in --top-k"
            )
        failures = gate.failures(aggregates[gate_k])
        if failures:
            for failure in failures:
                print(
                    f"retrieval gate failed at k={gate_k}: "
                    f"{failure}"
                )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
