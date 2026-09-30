from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from chunk_experiment.benchmark import evaluate_retriever_profiled
from chunk_experiment.dataset import load_evaluation_dataset
from chunk_experiment.embeddings import (
    HashEmbeddingProvider,
    HttpEmbeddingProvider,
)
from chunk_experiment.http_rerank import HttpRerankScoreProvider
from chunk_experiment.production import (
    CachedEmbeddingProvider,
    CachedRerankScoreProvider,
    ModelIdentity,
    UnitPricing,
)
from chunk_experiment.production_benchmark import (
    pareto_frontier,
    run_benchmark_phase,
)
from chunk_experiment.rerank import (
    LexicalOverlapScoreProvider,
    ScoreProviderReranker,
    TwoStageRetriever,
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
            "Run quality, cold/warm latency, cache, usage and Pareto "
            "benchmarks for RAG retrieval configurations"
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--chunker",
        choices=("recursive", "markdown", "semantic"),
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
        choices=("none", "lexical", "http"),
        default=["none"],
    )
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--contextual-headings", action="store_true")

    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "http"),
        default="hash",
    )
    parser.add_argument("--embedding-url")
    parser.add_argument("--embedding-model", default="m3e")
    parser.add_argument("--embedding-version", default="m3e")
    parser.add_argument("--embedding-provider-label", default="local")
    parser.add_argument("--embedding-revision", default="")
    parser.add_argument("--embedding-endpoint-label", default="")
    parser.add_argument(
        "--embedding-bearer-env",
        help="Environment variable containing an embedding API bearer token",
    )

    parser.add_argument("--rerank-url")
    parser.add_argument("--rerank-model", default="rerank")
    parser.add_argument("--rerank-provider-label", default="local")
    parser.add_argument("--rerank-revision", default="")
    parser.add_argument("--rerank-endpoint-label", default="")
    parser.add_argument(
        "--rerank-bearer-env",
        help="Environment variable containing a rerank API bearer token",
    )

    parser.add_argument("--cold-passes", type=int, default=1)
    parser.add_argument("--warm-passes", type=int, default=3)
    parser.add_argument(
        "--embedding-usd-per-1k-items",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rerank-usd-per-1k-pairs",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("production_benchmark_v6.json"),
    )
    return parser.parse_args()


def _bearer_headers(env_name: str | None) -> dict[str, str]:
    if env_name is None:
        return {}
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(
            f"environment variable {env_name!r} is required but empty"
        )
    return {"Authorization": f"Bearer {value}"}


def _embedding_provider(
    args: argparse.Namespace,
) -> tuple[CachedEmbeddingProvider, ModelIdentity]:
    if args.embedding_backend == "hash":
        raw = HashEmbeddingProvider()
        identity = ModelIdentity(
            kind="embedding",
            provider="deterministic-hash",
            model="lexical-hash",
            revision="v1",
        )
    else:
        if not args.embedding_url:
            raise SystemExit(
                "--embedding-url is required for --embedding-backend http"
            )
        raw = HttpEmbeddingProvider(
            args.embedding_url,
            model=args.embedding_model,
            version=args.embedding_version,
            headers=_bearer_headers(args.embedding_bearer_env),
        )
        identity = ModelIdentity(
            kind="embedding",
            provider=args.embedding_provider_label,
            model=args.embedding_model,
            revision=args.embedding_revision or args.embedding_version,
            endpoint_label=args.embedding_endpoint_label,
        )
    return CachedEmbeddingProvider(
        raw,
        identity=identity,
    ), identity


def _rerank_provider(
    args: argparse.Namespace,
    mode: str,
) -> tuple[CachedRerankScoreProvider | None, ModelIdentity | None]:
    if mode == "none":
        return None, None
    if mode == "lexical":
        raw = LexicalOverlapScoreProvider()
        identity = ModelIdentity(
            kind="rerank",
            provider="deterministic-lexical",
            model="overlap",
            revision="v1",
        )
    else:
        if not args.rerank_url:
            raise SystemExit("--rerank-url is required for --rerankers http")
        raw = HttpRerankScoreProvider(
            args.rerank_url,
            model=args.rerank_model,
            headers=_bearer_headers(args.rerank_bearer_env),
        )
        identity = ModelIdentity(
            kind="rerank",
            provider=args.rerank_provider_label,
            model=args.rerank_model,
            revision=args.rerank_revision,
            endpoint_label=args.rerank_endpoint_label,
        )
    return CachedRerankScoreProvider(
        raw,
        identity=identity,
    ), identity


def _needs_embeddings(
    chunker: str,
    retrieval_mode: str,
) -> bool:
    return (
        chunker == "semantic"
        or retrieval_mode in {"dense", "hybrid"}
    )


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.rerank_candidate_k < args.top_k:
        raise SystemExit("--rerank-candidate-k must be >= --top-k")
    if args.cold_passes <= 0 or args.warm_passes <= 0:
        raise SystemExit("--cold-passes and --warm-passes must be > 0")

    dataset = load_evaluation_dataset(args.dataset)
    pricing = UnitPricing(
        embedding_usd_per_1k_items=args.embedding_usd_per_1k_items,
        rerank_usd_per_1k_pairs=args.rerank_usd_per_1k_pairs,
    )
    renderer = (
        contextual_heading_text
        if args.contextual_headings
        else plain_chunk_text
    )
    query_texts = [case.query for case in dataset.queries]
    runs: list[dict] = []

    for retrieval_mode in args.retrieval_modes:
        needs_embeddings = _needs_embeddings(
            args.chunker,
            retrieval_mode,
        )
        embedding_provider = None
        embedding_identity = None
        if needs_embeddings:
            embedding_provider, embedding_identity = _embedding_provider(args)

        started_build = time.perf_counter()
        strategy = StrategyConfig(
            name=args.chunker,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_size=min(40, args.chunk_size),
        )
        chunker = build_strategy(
            strategy,
            embedding_provider=embedding_provider,
        )
        chunks = chunk_documents(
            dataset.documents,
            chunker,
        )
        base_retriever = build_retriever(
            RetrieverConfig(mode=retrieval_mode),
            chunks,
            embedding_provider=embedding_provider,
            renderer=renderer,
        )
        index_build_seconds = time.perf_counter() - started_build
        index_embedding_usage = (
            embedding_provider.usage()
            if embedding_provider is not None
            else None
        )

        for rerank_mode in args.rerankers:
            if embedding_provider is not None:
                embedding_provider.clear_cache(reset_usage=True)
            rerank_provider, rerank_identity = _rerank_provider(
                args,
                rerank_mode,
            )
            retriever = base_retriever
            two_stage = None
            if rerank_provider is not None:
                reranker = ScoreProviderReranker(
                    rerank_provider,
                    renderer=renderer,
                )
                two_stage = TwoStageRetriever(
                    base_retriever,
                    reranker,
                    candidate_k=args.rerank_candidate_k,
                )
                retriever = two_stage

            if embedding_provider is not None:
                embedding_provider.reset_usage()
            if rerank_provider is not None:
                rerank_provider.reset_usage()

            quality, _, context_cost, _ = evaluate_retriever_profiled(
                retriever,
                dataset.queries,
                top_k=(args.top_k,),
                cost_k=args.top_k,
            )
            stage_metrics = None
            if two_stage is not None:
                stage_metrics, _ = evaluate_two_stage_retriever(
                    two_stage,
                    dataset.queries,
                    final_k=args.top_k,
                )
            quality_embedding_usage = (
                embedding_provider.usage()
                if embedding_provider is not None
                else None
            )
            quality_rerank_usage = (
                rerank_provider.usage()
                if rerank_provider is not None
                else None
            )

            if embedding_provider is not None:
                embedding_provider.clear_cache(reset_usage=True)
            if rerank_provider is not None:
                rerank_provider.clear_cache(reset_usage=True)

            cold = run_benchmark_phase(
                retriever,
                query_texts,
                top_k=args.top_k,
                phase="cold",
                passes=args.cold_passes,
            )
            cold_embedding_usage = (
                embedding_provider.usage()
                if embedding_provider is not None
                else None
            )
            cold_rerank_usage = (
                rerank_provider.usage()
                if rerank_provider is not None
                else None
            )
            cold_estimated_cost = pricing.estimate_usd(
                embedding_usage=cold_embedding_usage,
                rerank_usage=cold_rerank_usage,
            )

            if embedding_provider is not None:
                embedding_provider.reset_usage()
            if rerank_provider is not None:
                rerank_provider.reset_usage()

            warm = run_benchmark_phase(
                retriever,
                query_texts,
                top_k=args.top_k,
                phase="warm",
                passes=args.warm_passes,
            )
            warm_embedding_usage = (
                embedding_provider.usage()
                if embedding_provider is not None
                else None
            )
            warm_rerank_usage = (
                rerank_provider.usage()
                if rerank_provider is not None
                else None
            )
            warm_estimated_cost = pricing.estimate_usd(
                embedding_usage=warm_embedding_usage,
                rerank_usage=warm_rerank_usage,
            )

            aggregate = quality[args.top_k]
            runs.append(
                {
                    "retrieval_mode": retrieval_mode,
                    "reranker": rerank_mode,
                    "chunker": args.chunker,
                    "chunk_size": args.chunk_size,
                    "chunks": len(chunks),
                    "top_k": args.top_k,
                    "index_build_seconds": index_build_seconds,
                    "quality": aggregate.to_dict(),
                    "context_cost": context_cost.to_dict(),
                    "two_stage": (
                        stage_metrics.to_dict()
                        if stage_metrics is not None
                        else None
                    ),
                    "models": {
                        "embedding": (
                            embedding_identity.to_dict()
                            if embedding_identity is not None
                            else None
                        ),
                        "rerank": (
                            rerank_identity.to_dict()
                            if rerank_identity is not None
                            else None
                        ),
                    },
                    "usage": {
                        "index_embedding": (
                            index_embedding_usage.to_dict()
                            if index_embedding_usage is not None
                            else None
                        ),
                        "quality_embedding": (
                            quality_embedding_usage.to_dict()
                            if quality_embedding_usage is not None
                            else None
                        ),
                        "quality_rerank": (
                            quality_rerank_usage.to_dict()
                            if quality_rerank_usage is not None
                            else None
                        ),
                        "cold_embedding": (
                            cold_embedding_usage.to_dict()
                            if cold_embedding_usage is not None
                            else None
                        ),
                        "cold_rerank": (
                            cold_rerank_usage.to_dict()
                            if cold_rerank_usage is not None
                            else None
                        ),
                        "warm_embedding": (
                            warm_embedding_usage.to_dict()
                            if warm_embedding_usage is not None
                            else None
                        ),
                        "warm_rerank": (
                            warm_rerank_usage.to_dict()
                            if warm_rerank_usage is not None
                            else None
                        ),
                    },
                    "benchmark": {
                        "cold": cold.to_dict(),
                        "warm": warm.to_dict(),
                    },
                    "estimated_cost_usd": {
                        "cold": cold_estimated_cost,
                        "warm": warm_estimated_cost,
                    },
                }
            )

    pareto_rows: list[dict[str, float | str | int]] = []
    for index, run in enumerate(runs):
        pareto_rows.append(
            {
                "run_index": index,
                "retrieval_mode": run["retrieval_mode"],
                "reranker": run["reranker"],
                "ndcg": run["quality"]["ndcg"],
                "span_recall": run["quality"]["span_recall"],
                "warm_mean_latency_ms": run["benchmark"]["warm"]["mean_latency_ms"],
                "warm_estimated_cost_usd": run["estimated_cost_usd"]["warm"],
            }
        )
    frontier = pareto_frontier(
        pareto_rows,
        maximize=("ndcg", "span_recall"),
        minimize=(
            "warm_mean_latency_ms",
            "warm_estimated_cost_usd",
        ),
    )

    payload = {
        "schema_version": 1,
        "dataset": {
            "id": dataset.dataset_id,
            "fingerprint": dataset.fingerprint,
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "pricing": {
            "embedding_usd_per_1k_items": args.embedding_usd_per_1k_items,
            "rerank_usd_per_1k_pairs": args.rerank_usd_per_1k_pairs,
            "note": (
                "user-supplied item pricing; this is an estimate, "
                "not vendor billing data"
            ),
        },
        "runs": runs,
        "pareto_frontier": frontier,
        "notes": [
            "raw endpoint URLs and authorization values are intentionally excluded",
            "cold/warm results are environment-dependent; compare controlled runs",
            "hash embeddings and lexical reranking remain deterministic plumbing baselines",
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
    print(
        f"wrote {len(runs)} production benchmark runs "
        f"with {len(frontier)} Pareto points -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
