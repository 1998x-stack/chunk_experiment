from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from chunk_experiment.benchmark import evaluate_retriever_profiled
from chunk_experiment.dataset import load_evaluation_dataset
from chunk_experiment.embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from chunk_experiment.parent_child import (
    ParentChildRetriever,
    build_parent_child_index,
)
from chunk_experiment.retrieval import (
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from chunk_experiment.retriever_factory import (
    RetrieverConfig,
    build_retriever,
)
from chunk_experiment.strategy import StrategyConfig, build_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare chunking, retrieval and parent-child RAG strategies"
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("recursive", "markdown"),
        default=["recursive", "markdown"],
    )
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=("dense", "bm25", "hybrid"),
        default=["dense", "bm25", "hybrid"],
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[128, 256],
    )
    parser.add_argument(
        "--parent-multipliers",
        type=int,
        nargs="+",
        default=[1, 4],
        help="1 means flat retrieval; values >1 enable parent-child retrieval",
    )
    parser.add_argument("--overlap-ratio", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--cost-k", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "http"),
        default="hash",
    )
    parser.add_argument("--embedding-url")
    parser.add_argument(
        "--markdown-context",
        choices=("plain", "headings"),
        default="headings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retrieval_matrix_v4.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.overlap_ratio < 1.0:
        raise SystemExit("--overlap-ratio must be in [0, 1)")
    if any(multiplier <= 0 for multiplier in args.parent_multipliers):
        raise SystemExit("--parent-multipliers must be positive")

    dataset = load_evaluation_dataset(args.dataset)
    if args.embedding_backend == "hash":
        provider = HashEmbeddingProvider()
        provider_label = "deterministic-lexical-hash"
    else:
        if not args.embedding_url:
            raise SystemExit(
                "--embedding-url is required for --embedding-backend http"
            )
        provider = HttpEmbeddingProvider(args.embedding_url)
        provider_label = "http"

    runs: list[dict] = []
    for strategy_name in args.strategies:
        for chunk_size in args.chunk_sizes:
            overlap = max(
                0,
                min(
                    chunk_size - 1,
                    int(chunk_size * args.overlap_ratio),
                ),
            )
            child_config = StrategyConfig(
                name=strategy_name,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min(40, chunk_size),
            )
            child_chunker = build_strategy(
                child_config,
                embedding_provider=provider,
            )
            renderer = (
                contextual_heading_text
                if strategy_name == "markdown"
                and args.markdown_context == "headings"
                else plain_chunk_text
            )

            for parent_multiplier in args.parent_multipliers:
                if parent_multiplier == 1:
                    flat_chunks = chunk_documents(
                        dataset.documents,
                        child_chunker,
                    )
                    retrieval_chunks = flat_chunks
                    index = None
                else:
                    parent_size = chunk_size * parent_multiplier
                    parent_config = StrategyConfig(
                        name=strategy_name,
                        chunk_size=parent_size,
                        overlap=0,
                        min_chunk_size=min(40, parent_size),
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
                    retrieval_chunks = list(index.children)

                for retrieval_mode in args.retrieval_modes:
                    retrieval_config = RetrieverConfig(
                        mode=retrieval_mode,
                    )
                    started = time.perf_counter()
                    base_retriever = build_retriever(
                        retrieval_config,
                        retrieval_chunks,
                        embedding_provider=provider,
                        renderer=renderer,
                    )
                    retriever = (
                        ParentChildRetriever(
                            base_retriever,
                            index,
                        )
                        if index is not None
                        else base_retriever
                    )
                    index_seconds = time.perf_counter() - started
                    (
                        aggregates,
                        _,
                        costs,
                        _,
                    ) = evaluate_retriever_profiled(
                        retriever,
                        dataset.queries,
                        top_k=args.top_k,
                        cost_k=args.cost_k,
                    )
                    runs.append(
                        {
                            "strategy": strategy_name,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "retrieval_mode": retrieval_mode,
                            "parent_multiplier": parent_multiplier,
                            "retrieval_chunks": len(retrieval_chunks),
                            "returned_units": (
                                len(index.parents)
                                if index is not None
                                else len(retrieval_chunks)
                            ),
                            "index_seconds": index_seconds,
                            "metrics": {
                                str(k): metrics.to_dict()
                                for k, metrics in aggregates.items()
                            },
                            "cost": costs.to_dict(),
                        }
                    )

    payload = {
        "schema_version": 2,
        "dataset": {
            "id": dataset.dataset_id,
            "fingerprint": dataset.fingerprint,
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
        },
        "embedding_provider": provider_label,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runs": runs,
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
    print(f"wrote {len(runs)} v4 retrieval runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
