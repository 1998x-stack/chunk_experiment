from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .benchmark import evaluate_retriever_profiled
from .dataset import load_evaluation_dataset
from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .parent_child import (
    ParentChildRetriever,
    build_parent_child_index,
)
from .retrieval import (
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from .retrieval_eval import RetrievalGate
from .retriever_factory import RetrieverConfig, build_retriever
from .strategy import StrategyConfig, build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chunking and retrieval strategies against "
            "a versioned retrieval golden set"
        )
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
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
    )
    parser.add_argument(
        "--retriever",
        choices=("hash", "http"),
        default="hash",
        help="Embedding backend for dense/hybrid retrieval and semantic chunking",
    )
    parser.add_argument("--embedding-url")
    parser.add_argument("--contextual-headings", action="store_true")
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        help="Enable child retrieval with larger returned parent contexts",
    )
    parser.add_argument("--parent-overlap", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--cost-k", type=int)
    parser.add_argument("--gate-k", type=int)
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-span-recall", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-ndcg", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retrieval_eval_v4.json"),
    )
    return parser


def _needs_embeddings(args: argparse.Namespace) -> bool:
    return (
        args.chunker == "semantic"
        or args.retrieval_mode in {"dense", "hybrid"}
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_evaluation_dataset(args.dataset)

    provider = None
    provider_label = None
    if _needs_embeddings(args):
        if args.retriever == "hash":
            provider = HashEmbeddingProvider()
            provider_label = "deterministic-lexical-hash"
        else:
            if not args.embedding_url:
                raise SystemExit(
                    "--embedding-url is required for --retriever http"
                )
            provider = HttpEmbeddingProvider(args.embedding_url)
            provider_label = "http"

    config = StrategyConfig(
        name=args.chunker,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
        breakpoint_percentile=args.breakpoint_percentile,
    )
    child_chunker = build_strategy(
        config,
        embedding_provider=provider,
    )
    renderer = (
        contextual_heading_text
        if args.contextual_headings
        else plain_chunk_text
    )
    retriever_config = RetrieverConfig(
        mode=args.retrieval_mode,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )

    parent_child = args.parent_chunk_size is not None
    if parent_child:
        if args.parent_chunk_size <= args.chunk_size:
            raise SystemExit(
                "--parent-chunk-size must be greater than --chunk-size"
            )
        parent_config = StrategyConfig(
            name=args.chunker,
            chunk_size=args.parent_chunk_size,
            overlap=args.parent_overlap,
            min_chunk_size=min(
                args.min_chunk_size,
                args.parent_chunk_size,
            ),
            breakpoint_percentile=args.breakpoint_percentile,
        )
        parent_chunker = build_strategy(
            parent_config,
            embedding_provider=provider,
        )
        index = build_parent_child_index(
            dataset.documents,
            parent_chunker=parent_chunker,
            child_chunker=child_chunker,
        )
        child_retriever = build_retriever(
            retriever_config,
            index.children,
            embedding_provider=provider,
            renderer=renderer,
        )
        retriever = ParentChildRetriever(
            child_retriever,
            index,
        )
        retrieval_chunks = len(index.children)
        returned_units = len(index.parents)
    else:
        chunks = chunk_documents(
            dataset.documents,
            child_chunker,
        )
        retriever = build_retriever(
            retriever_config,
            chunks,
            embedding_provider=provider,
            renderer=renderer,
        )
        retrieval_chunks = len(chunks)
        returned_units = len(chunks)

    aggregates, per_query, cost_summary, per_query_cost = (
        evaluate_retriever_profiled(
            retriever,
            dataset.queries,
            top_k=args.top_k,
            cost_k=args.cost_k,
        )
    )

    payload = {
        "schema_version": 2,
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
            "parent_chunk_size": args.parent_chunk_size,
            "parent_overlap": (
                args.parent_overlap
                if parent_child
                else None
            ),
        },
        "retrieval": {
            "mode": retriever_config.mode,
            "embedding_provider": provider_label,
            "contextual_headings": args.contextual_headings,
            "dense_weight": retriever_config.dense_weight,
            "sparse_weight": retriever_config.sparse_weight,
            "rrf_k": retriever_config.rrf_k,
            "candidate_k": retriever_config.candidate_k,
            "parent_child": parent_child,
            "retrieval_chunks": retrieval_chunks,
            "returned_units": returned_units,
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
        "cost": cost_summary.to_dict(),
        "per_query": [row.to_dict() for row in per_query],
        "per_query_cost": [
            row.to_dict()
            for row in per_query_cost
        ],
        "notes": (
            [
                "hash embeddings are deterministic lexical smoke vectors; "
                "use a documented real embedding backend for semantic conclusions"
            ]
            if provider_label == "deterministic-lexical-hash"
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
        f"{retrieval_chunks} retrieval chunks -> {args.output}"
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
