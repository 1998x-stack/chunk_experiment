from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from chunk_experiment.benchmark import evaluate_retriever_profiled
from chunk_experiment.dataset import load_evaluation_dataset
from chunk_experiment.embeddings import (
    HashEmbeddingProvider,
    HttpEmbeddingProvider,
)
from chunk_experiment.rerank import TwoStageRetriever
from chunk_experiment.rerank_factory import (
    RerankerConfig,
    build_reranker,
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
from chunk_experiment.two_stage_eval import evaluate_two_stage_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate retrieval and reranking stages "
            "on a retrieval golden set"
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--strategy",
        choices=("recursive", "markdown"),
        default="markdown",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=("dense", "bm25", "hybrid"),
        default=["dense", "bm25", "hybrid"],
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        choices=("none", "lexical"),
        default=["none", "lexical"],
    )
    parser.add_argument(
        "--candidate-ks",
        type=int,
        nargs="+",
        default=[5, 10, 20],
    )
    parser.add_argument(
        "--final-ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "http"),
        default="hash",
    )
    parser.add_argument("--embedding-url")
    parser.add_argument(
        "--contextual-headings",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rerank_matrix_v5.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.candidate_ks) <= 0 or min(args.final_ks) <= 0:
        raise SystemExit("candidate/final K values must be positive")

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

    strategy = StrategyConfig(
        name=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=min(40, args.chunk_size),
    )
    chunker = build_strategy(
        strategy,
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

    runs: list[dict] = []
    for retrieval_mode in args.retrieval_modes:
        base_retriever = build_retriever(
            RetrieverConfig(mode=retrieval_mode),
            chunks,
            embedding_provider=provider,
            renderer=renderer,
        )

        aggregates, _, costs, _ = evaluate_retriever_profiled(
            base_retriever,
            dataset.queries,
            top_k=args.final_ks,
            cost_k=max(args.final_ks),
        )
        runs.append(
            {
                "retrieval_mode": retrieval_mode,
                "reranker": "none",
                "candidate_k": None,
                "final_ks": args.final_ks,
                "metrics": {
                    str(k): metrics.to_dict()
                    for k, metrics in aggregates.items()
                },
                "cost": costs.to_dict(),
                "stage_metrics": None,
            }
        )

        if "lexical" not in args.rerankers:
            continue

        for candidate_k in args.candidate_ks:
            valid_final_ks = [
                k
                for k in args.final_ks
                if k <= candidate_k
            ]
            if not valid_final_ks:
                continue

            reranker = build_reranker(
                RerankerConfig(
                    mode="lexical",
                    candidate_k=candidate_k,
                ),
                renderer=renderer,
            )
            assert reranker is not None
            two_stage = TwoStageRetriever(
                base_retriever,
                reranker,
                candidate_k=candidate_k,
            )
            final_max = max(valid_final_ks)
            aggregates, _, costs, _ = evaluate_retriever_profiled(
                two_stage,
                dataset.queries,
                top_k=valid_final_ks,
                cost_k=final_max,
            )
            stage_metrics, _ = evaluate_two_stage_retriever(
                two_stage,
                dataset.queries,
                final_k=final_max,
            )
            runs.append(
                {
                    "retrieval_mode": retrieval_mode,
                    "reranker": "lexical",
                    "candidate_k": candidate_k,
                    "final_ks": valid_final_ks,
                    "metrics": {
                        str(k): metrics.to_dict()
                        for k, metrics in aggregates.items()
                    },
                    "cost": costs.to_dict(),
                    "stage_metrics": stage_metrics.to_dict(),
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
        "strategy": {
            "name": strategy.name,
            "chunk_size": strategy.chunk_size,
            "overlap": strategy.overlap,
        },
        "embedding_provider": provider_label,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runs": runs,
        "notes": [
            "lexical reranking and hash embeddings are deterministic "
            "pipeline baselines, not semantic-quality models"
        ],
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
    print(f"wrote {len(runs)} v5 reranking runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
