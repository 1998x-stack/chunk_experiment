# Architecture v2

## Goal

Turn `chunk_experiment` from a collection of research scripts into a reproducible
RAG chunking workbench while keeping the original experiments available as legacy
baselines.

The main design rule is **source fidelity**: every chunk carries exact `[start, end)`
offsets into the original document. This makes highlighting, retrieval evaluation,
error analysis, deduplication, and future knowledge-graph linking independent of a
specific embedding model or vector database.

## Current repository findings

The pre-v2 repository mixes four concerns in the same top-level tree:

1. chunking implementations;
2. downloadable/raw corpora and papers;
3. experiment runners and generated result artifacts;
4. notebooks, debug scripts, and package placeholders.

Several package/config files are empty, imports mutate `sys.path`, tests include
print-driven smoke checks, and the existing semantic path uses inconsistent notions
of "token" across Chinese and English text.

The v2 package is intentionally additive so the historical experiments remain
inspectable while migration can happen incrementally.

## Target pipeline

```text
Document source
  -> parser / normalizer
  -> Chunk(text, start, end, metadata)
  -> optional enrichment
       - document/section context
       - entities / knowledge-graph ids
       - summaries
  -> index adapters
       - dense embeddings
       - BM25 / sparse
       - hybrid
  -> retriever
  -> optional reranker
  -> RAG generation
  -> evaluation
```

### 1. Core model

`Chunk` is the stable interchange type. It contains exact source text, source
offsets, ordinal, and metadata.

Algorithm-specific fields belong in metadata rather than changing the core schema.
This allows recursive, semantic, Markdown/HTML-aware, late-chunking, or future
graph-aware algorithms to share the same evaluation and visualization code.

### 2. Chunker contract

A chunker implements:

```python
class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...
```

The first v2 implementation is `BoundaryAwareChunker`:

- deterministic;
- hard maximum character budget;
- configurable overlap;
- prioritized Chinese/English paragraph and sentence boundaries;
- source-preserving offsets;
- no embedding/network dependency.

It is a baseline, not a claim that character units are universally optimal.

### 3. Length units

The legacy code mixes "character" and approximate whitespace "token" counts.
The v2 core therefore starts with an explicit character budget.

Future token-aware adapters should expose the tokenizer/model name in experiment
metadata. A number should never be called "tokens" unless the tokenizer is defined.

### 4. Evaluation contract

Chunking quality and retrieval quality are separate.

Intrinsic chunking metrics:

- size distribution;
- source coverage;
- duplication introduced by overlap;
- natural-boundary alignment;
- deterministic output;
- latency.

Retrieval evaluation uses **gold source spans instead of gold chunk ids**. Chunk ids
change when the chunker changes; source spans do not. This avoids coupling the
benchmark labels to one algorithm's boundaries.

### 5. Experiment manifest

A future experiment runner should persist, at minimum:

```yaml
experiment:
  id: ...
  seed: 42
dataset:
  name: ...
  version: ...
chunker:
  name: ...
  config: ...
embedding:
  provider: ...
  model: ...
retrieval:
  strategy: dense|bm25|hybrid
  top_k: ...
reranker:
  enabled: false
environment:
  python: ...
  git_sha: ...
```

Generated metrics should live under a timestamped/result-id directory rather than
overwriting repository-root JSON files.

## Data and artifact strategy

The repository currently contains large tracked corpora and PDFs even though
`.gitignore` ignores `data/`. Ignoring a path does not remove already tracked
files or repository history.

Recommended migration:

1. keep the current tracked files during the compatibility phase;
2. create a small, license-clear fixture dataset for CI;
3. publish large corpora via a dataset release/object store/DVC-style manifest;
4. store only checksums, source metadata, and download scripts in Git;
5. keep generated plots/results under an artifact directory or CI artifacts;
6. perform history rewriting only as a separate, explicitly coordinated operation.

## Package boundaries

```text
src/chunk_experiment/
  models.py      # stable Chunk / config models
  chunkers.py    # chunker protocol + baseline
  metrics.py     # intrinsic + span-based retrieval metrics
  benchmark.py   # deterministic timing/evaluation harness
  cli.py         # reproducible command-line surface
```

Legacy modules under `src/*.py` and `util/*.py` remain available during migration.

## Next adapters

Priority order:

1. adapter around the current recursive implementation;
2. corrected semantic chunker with explicit embedding/tokenizer interfaces;
3. Markdown/HTML/LaTeX structure-aware chunkers returning source offsets;
4. hybrid retrieval benchmark (dense + BM25);
5. reranking evaluation;
6. contextual chunk enrichment;
7. late-chunking embedding adapter;
8. entity/KG-aware chunk metadata and neighborhood retrieval.

## Compatibility policy

- v2 code is additive.
- historical result JSON/summary files are treated as legacy evidence.
- when a metric definition changes, regenerate under a new experiment id rather
  than silently replacing old numbers.
- correctness fixes that alter semantic results must be called out in release notes.
