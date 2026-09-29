from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .benchmark import run_benchmark
from .chunkers import BoundaryAwareChunker
from .models import ChunkingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunk-exp",
        description="Reproducible text chunking workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("split", "benchmark"):
        sub = subparsers.add_parser(command)
        sub.add_argument("input", type=Path)
        sub.add_argument("--chunk-size", type=int, default=500)
        sub.add_argument("--overlap", type=int, default=50)
        sub.add_argument("--min-chunk-size", type=int, default=80)

    split_parser = subparsers.choices["split"]
    split_parser.add_argument("--output", type=Path)

    benchmark_parser = subparsers.choices["benchmark"]
    benchmark_parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.input.read_text(encoding="utf-8")
    config = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
    )
    chunker = BoundaryAwareChunker(config)

    if args.command == "split":
        chunks = chunker.split(text)
        rows = [
            json.dumps(
                {
                    "id": chunk.ordinal,
                    "start": chunk.start,
                    "end": chunk.end,
                    "size": chunk.size,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
                ensure_ascii=False,
            )
            for chunk in chunks
        ]
        payload = "\n".join(rows) + ("\n" if rows else "")
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0

    result, _ = run_benchmark(text, chunker, repeats=args.repeats)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
