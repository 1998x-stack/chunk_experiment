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
from .rerank import TwoStageRetriever
from .rerank_factory import RerankerConfig, build_reranker
from .retrieval import (
    chunk_documents,
    contextual_heading_text,
    plain_chunk_text,
)
from .retrieval_eval import RetrievalGate
from .retriever_factory import RetrieverConfig, build_retriever
from .strategy import StrategyConfig, build_strategy
from .two_stage_eval import (
    TwoStageGate,
    evaluate_two_stage_retriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chunking, retrieval and reranking strategies "
            "against a versioned retrieval golden set"
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
        help=(
            "Embedding backend for dense/hybrid retrieval "
            "and semantic chunking"
        ),
    )
    parser.add_argument("--embedding-url")
    parser.add_argument("--contextual-headings", action="store_true")
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Candidate depth used inside hybrid RRF",
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        help="Enable child retrieval with larger returned parent contexts",
    )
    parser.add_argument("--parent-overlap", type=int, default=0)
    parser.add_argument(
        "--reranker",
        choices=("none", "lexical"),
        default="none",
    )
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=20,
        help="First-stage candidate depth exposed to the reranker",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--cost-k", type=int)
    parser.add_argument("--gate-k", type=int)
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-span-recall", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-ndcg", type=float)
    parser.add_argument("--min-candidate-span-recall", type=float)
    parser.add_argument("--min-rerank-final-ndcg", type=float)
    parser.add_argument("--min-rerank-ndcg-lift", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("retrieval_eval_v5.json"),
    )
    return parser


def _needs_embeddings(args: argparse.Namespace) -> bool:
    return (
        args.chunker == "semantic"
        or args.retrieval_mode in {"dense", "hybrid"}
    )


def _stage_gate_requested(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.min_candidate_span_recall,
            args.min_rerank_final_ndcg,
            args.min_rerank_ndcg_lift,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_evaluation_dataset(args.dataset)

    if not args.top_k or min(args.top_k) <= 0:
        raise SystemExit("--top-k must contain positive integers")

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
        base_retriever = ParentChildRetriever(
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
        base_retriever = build_retriever(
            retriever_config,
            chunks,
            embedding_provider=provider,
            renderer=renderer,
        )
        retrieval_chunks = len(chunks)
        returned_units = len(chunks)

    reranker_config = RerankerConfig(
        mode=args.reranker,
        candidate_k=args.rerank_candidate_k,
    )
    reranker = build_reranker(
        reranker_config,
        renderer=renderer,
    )
    two_stage_retriever = None
    retriever = base_retriever
    if reranker is not None:
        required_final_k = max(
            max(args.top_k),
            args.cost_k or 0,
        )
        if reranker_config.candidate_k < required_final_k:
            raise SystemExit(
                "--rerank-candidate-k must be >= max(--top-k, --cost-k)"
            )
        two_stage_retriever = TwoStageRetriever(
            base_retriever,
            reranker,
            candidate_k=reranker_config.candidate_k,
        )
        retriever = two_stage_retriever
    elif _stage_gate_requested(args):
        raise SystemExit(
            "reranking stage gates require --reranker lexical"
        )

    aggregates, per_query, cost_summary, per_query_cost = (
        evaluate_retriever_profiled(
            retriever,
            dataset.queries,
            top_k=args.top_k,
            cost_k=args.cost_k,
        )
    )

    two_stage_metrics = None
    two_stage_rows = None
    if two_stage_retriever is not None:
        two_stage_metrics, two_stage_rows = (
            evaluate_two_stage_retriever(
                two_stage_retriever,
                dataset.queries,
                final_k=max(args.top_k),
            )
        )

    payload = {
        "schema_version": 3,
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
        "reranking": {
            "mode": reranker_config.mode,
            "candidate_k": (
                reranker_config.candidate_k
                if reranker is not None
                else None
            ),
            "stage_metrics": (
                two_stage_metrics.to_dict()
                if two_stage_metrics is not None
                else None
            ),
            "per_query_stage": (
                [
                    row.to_dict()
                    for row in two_stage_rows
                ]
                if two_stage_rows is not None
                else None
            ),
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
                "hash embeddings and lexical reranking are deterministic "
                "pipeline baselines; use documented real models for "
                "semantic-quality conclusions"
            ]
            if (
                provider_label == "deterministic-lexical-hash"
                or reranker_config.mode == "lexical"
            )
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

    if _stage_gate_requested(args):
        assert two_stage_metrics is not None
        stage_gate = TwoStageGate(
            min_candidate_span_recall=args.min_candidate_span_recall,
            min_final_ndcg=args.min_rerank_final_ndcg,
            min_mean_ndcg_lift=args.min_rerank_ndcg_lift,
        )
        stage_failures = stage_gate.failures(
            two_stage_metrics
        )
        if stage_failures:
            for failure in stage_failures:
                print(f"reranking stage gate failed: {failure}")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
