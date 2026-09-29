from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from chunk_experiment.dataset import load_evaluation_dataset
from chunk_experiment.embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from chunk_experiment.retrieval import (
    DenseRetriever,
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from chunk_experiment.retrieval_eval import evaluate_retriever
from chunk_experiment.strategy import StrategyConfig, build_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on a retrieval golden set"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("recursive", "markdown", "semantic"),
        default=["recursive", "markdown"],
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512],
    )
    parser.add_argument("--overlap-ratio", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--retriever", choices=("hash", "http"), default="hash")
    parser.add_argument("--embedding-url")
    parser.add_argument(
        "--markdown-context",
        choices=("plain", "headings", "both"),
        default="both",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retrieval_matrix_v3.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.overlap_ratio < 1.0:
        raise SystemExit("--overlap-ratio must be in [0, 1)")
    dataset = load_evaluation_dataset(args.dataset)

    if args.retriever == "hash":
        provider = HashEmbeddingProvider()
        provider_label = "deterministic-lexical-hash"
    else:
        if not args.embedding_url:
            raise SystemExit("--embedding-url is required for --retriever http")
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
            config = StrategyConfig(
                name=strategy_name,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min(40, chunk_size),
            )
            chunker = build_strategy(
                config,
                embedding_provider=provider,
            )
            indexed = chunk_documents(
                dataset.documents,
                chunker,
            )

            renderers = [("plain", plain_chunk_text)]
            if (
                strategy_name == "markdown"
                and args.markdown_context in {"headings", "both"}
            ):
                renderers = [("headings", contextual_heading_text)]
                if args.markdown_context == "both":
                    renderers.insert(0, ("plain", plain_chunk_text))

            for renderer_name, renderer in renderers:
                started = time.perf_counter()
                retriever = DenseRetriever(
                    provider,
                    indexed,
                    renderer=renderer,
                )
                aggregates, _ = evaluate_retriever(
                    retriever,
                    dataset.queries,
                    top_k=args.top_k,
                )
                elapsed = time.perf_counter() - started
                runs.append(
                    {
                        "strategy": strategy_name,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "renderer": renderer_name,
                        "chunks": len(indexed),
                        "elapsed_seconds": elapsed,
                        "metrics": {
                            str(k): metrics.to_dict()
                            for k, metrics in aggregates.items()
                        },
                    }
                )

    payload = {
        "schema_version": 1,
        "dataset": {
            "id": dataset.dataset_id,
            "fingerprint": dataset.fingerprint,
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
        },
        "retriever": provider_label,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runs": runs,
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
    print(f"wrote {len(runs)} retrieval runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
