from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from chunk_experiment import (
    HashEmbeddingProvider,
    RecursiveChunker,
    SemanticChunker,
    approximate_token_length,
    character_length,
    evaluate_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible directory benchmark")
    parser.add_argument("dataset", type=Path, help="Directory containing UTF-8 .txt files")
    parser.add_argument("--output", type=Path, default=Path("v2_benchmark_results.json"))
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument(
        "--include-semantic-hash",
        action="store_true",
        help="Include deterministic lexical-hash baseline (not a semantic-quality model)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.dataset.rglob("*.txt"))[: args.max_files]
    if not files:
        raise SystemExit(f"no .txt files found under {args.dataset}")

    results: list[dict] = []
    hash_provider = HashEmbeddingProvider()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for chunk_size in args.chunk_sizes:
            overlap = min(chunk_size - 1, max(0, int(chunk_size * args.overlap_ratio)))
            candidates = [
                (
                    "recursive",
                    RecursiveChunker(chunk_size, overlap),
                    character_length,
                    {"chunk_size": chunk_size, "chunk_overlap": overlap},
                )
            ]
            if args.include_semantic_hash:
                candidates.append(
                    (
                        "semantic-hash",
                        SemanticChunker(hash_provider, chunk_size=chunk_size, min_chunk_size=0),
                        approximate_token_length,
                        {
                            "chunk_size": chunk_size,
                            "embedding_backend": "deterministic-lexical-hash",
                        },
                    )
                )

            for name, chunker, metric, params in candidates:
                started = time.perf_counter()
                chunks = chunker.split(text)
                elapsed = time.perf_counter() - started
                metrics = evaluate_chunks(
                    text, chunks, max_chunk_size=chunk_size, length_metric=metric
                )
                results.append(
                    {
                        "document": str(path.relative_to(args.dataset)),
                        "document_sha256": doc_sha,
                        "algorithm": name,
                        "parameters": params,
                        "elapsed_seconds": elapsed,
                        "metrics": metrics.to_dict(),
                    }
                )

    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "dataset": str(args.dataset),
        "documents": len(files),
        "results": results,
        "notes": [
            "semantic-hash is a deterministic lexical baseline for plumbing/CI, "
            "not a semantic-quality benchmark",
            "use a documented real embedding provider before drawing semantic-quality conclusions",
        ],
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(results)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
