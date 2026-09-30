from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import HashEmbeddingProvider, HttpEmbeddingProvider
from .evaluation import evaluate_chunks
from .length import approximate_token_length, character_length
from .recursive import RecursiveChunker
from .semantic import SemanticChunker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible text chunking experiments")
    parser.add_argument("input", type=Path, help="UTF-8 text file")
    parser.add_argument(
        "--algorithm",
        choices=("recursive", "semantic-hash", "semantic-http"),
        default="recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--min-chunk-size", type=int, default=40)
    parser.add_argument("--embedding-url")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.input.read_text(encoding="utf-8")

    if args.algorithm == "recursive":
        metric = character_length
        chunker = RecursiveChunker(args.chunk_size, args.overlap, length_metric=metric)
    else:
        metric = approximate_token_length
        if args.algorithm == "semantic-hash":
            provider = HashEmbeddingProvider()
        else:
            if not args.embedding_url:
                raise SystemExit("--embedding-url is required for semantic-http")
            provider = HttpEmbeddingProvider(args.embedding_url)
        chunker = SemanticChunker(
            provider,
            chunk_size=args.chunk_size,
            min_chunk_size=args.min_chunk_size,
            length_metric=metric,
        )

    chunks = chunker.split(text)
    metrics = evaluate_chunks(text, chunks, max_chunk_size=args.chunk_size, length_metric=metric)
    payload = {
        "algorithm": args.algorithm,
        "metrics": metrics.to_dict(),
        "chunks": [
            {"start": c.start, "end": c.end, "text": c.text, "metadata": dict(c.metadata)}
            for c in chunks
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
