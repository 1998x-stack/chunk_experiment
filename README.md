# chunk_experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A bilingual RAG chunking research workbench: build chunkers, compare their intrinsic
properties, and evaluate retrieval against source evidence spans instead of
algorithm-specific chunk ids.

> 当前仓库同时保留历史实验脚本与新的 v2 核心包。v2 采用增量迁移，避免为了重构而一次性
> 破坏旧结果与 notebooks。

## What changed in v2

The repository is evolving from a script collection into a reproducible experiment
system:

- source-faithful `Chunk(text, start, end, metadata)` model;
- deterministic Chinese/English boundary-aware baseline;
- explicit overlap and hard chunk-size invariants;
- chunk quality metrics: coverage, duplication, size distribution, boundary alignment;
- span-based retrieval evaluation: Hit@K, MRR, relevant-span Recall@K;
- deterministic benchmark harness;
- `chunk-exp` CLI;
- installable `pyproject.toml` package;
- pytest + Ruff + Python 3.10/3.11/3.12 CI;
- architecture and evaluation design docs.

The historical modules under `src/*.py`, `util/*.py`, notebooks, and existing JSON
results remain available as legacy research assets.

## Quick start

### Core workbench

```bash
python -m pip install -e ".[dev]"
pytest
```

Split a UTF-8 document:

```bash
chunk-exp split path/to/document.txt \
  --chunk-size 500 \
  --overlap 50 \
  --output chunks.jsonl
```

Benchmark the baseline:

```bash
chunk-exp benchmark path/to/document.txt \
  --chunk-size 500 \
  --overlap 50 \
  --repeats 5
```

The core v2 path intentionally has no runtime third-party dependency.

### Legacy experiments

Install the historical dependency set through the compatibility extra:

```bash
python -m pip install -e ".[legacy]"
python ablation_experiments.py
python results_analysis.py
```

The root `requirements.txt` is retained for older workflows.

## Evaluation model

Chunking is not evaluated by one opaque score.

### Intrinsic chunk quality

- chunk count and size distribution;
- exact source coverage;
- duplication introduced by overlap;
- natural-boundary alignment;
- determinism;
- latency.

### Retrieval quality

Gold evidence is stored as original-document spans:

```json
{
  "case_id": "example-001",
  "relevant_spans": [[1830, 2014], [4420, 4518]]
}
```

This matters because chunk ids change when the chunking strategy changes. Source spans
remain stable, so recursive, semantic, structure-aware, contextual, and late-chunking
strategies can be compared against the same evidence labels.

See [docs/EVALUATION.md](docs/EVALUATION.md).

## Architecture

```text
document
  -> parser / normalizer
  -> chunker
  -> Chunk(text, source offsets, metadata)
  -> optional contextual / entity enrichment
  -> dense + sparse indexes
  -> retrieval
  -> optional reranking
  -> generation
  -> stage-specific evaluation
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

```text
src/chunk_experiment/       # v2 installable core
  models.py
  chunkers.py
  metrics.py
  benchmark.py
  cli.py

src/*.py                    # historical chunking implementations
util/*.py                   # historical utilities
tests/                      # v2 invariant tests
docs/                       # architecture + evaluation design
ablation_experiments.py     # historical experiment runner
results_analysis.py         # historical result analysis
*_SUMMARY.md                # historical write-ups
```

## Historical results

Existing reports are retained for provenance:

- `EXPERIMENT_SUMMARY.md`
- `ENHANCEMENT_SUMMARY.md`
- `algorithm_comparison_results.json`
- `param_ablation_results.json`
- `chunking_analysis.png`

These numbers should be treated as **legacy results**, not silently mixed with v2
experiments. The semantic similarity and multilingual length-counting paths have
received correctness fixes in v2, so affected historical results should be regenerated
under a new experiment id before making direct comparisons.

## Data policy

The repository contains large, already-tracked corpora and papers. Although `data/`
is ignored for new untracked files, that does not remove historical objects from Git.

The planned migration is:

1. keep existing tracked data during compatibility work;
2. add small license-clear fixtures for CI;
3. move large corpora to a versioned external dataset/artifact store;
4. keep checksums, manifests, provenance, and download scripts in Git;
5. treat any Git-history rewrite as a separate coordinated operation.

## Design roadmap

1. wrap the historical recursive algorithm behind the v2 `Chunker` contract;
2. migrate semantic chunking to explicit embedding/tokenizer interfaces;
3. make Markdown/HTML/LaTeX/code chunkers source-offset aware;
4. add dense + BM25 + hybrid retrieval experiments;
5. add reranker evaluation;
6. add contextual chunk enrichment;
7. add late-chunking embedding experiments;
8. add entity/KG metadata and graph-neighborhood retrieval;
9. publish versioned benchmark manifests and per-query result artifacts.

## Research references

- Late Chunking: https://arxiv.org/abs/2409.04701
- Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval
- RAGChecker: https://arxiv.org/abs/2408.08067
- CoFE-RAG: https://arxiv.org/abs/2410.12248

## License

MIT — see [LICENSE](LICENSE).
