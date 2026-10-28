# AGENTS.md — Instructions for AI coding agents

This file governs changes made by Codex, OpenCode, Claude Code and other
automated coding assistants in this repository. Read it **before** editing.
Human instructions and repository CI remain authoritative.

## 0. Mission and scope

Build an **auditable Factory Mode research workbench** for RAG chunking.
The goal is to learn which component improves retrieval under controlled,
reproducible experiments—not merely to add algorithms or produce attractive
plots.

**Non-negotiable engineering choices:**

- Use Python 3.10+ and the existing `pyproject.toml`.
- Prefer type-annotated, small, testable components and explicit dependencies.
- Use the `chunk_experiment/` package for maintained functionality.
- Use `ChunkerConfig` + `build_chunker` from `chunk_experiment/factory.py`.
- Use **Loguru** for structured runtime logging, not ad hoc `print()` calls in
  library or experiment internals. CLI output may print the final result path.
- Make small, reviewable commits. Do not rewrite the entire repository.

## 1. First read and repository boundaries

Read, in order:

1. `README.md` — supported commands and state of the project.
2. `docs/FACTORY_PIPELINE.md` — research protocol and acceptance gates.
3. `docs/EVALUATION.md` — what each metric does and does not establish.
4. `docs/REPOSITORY_LAYOUT.md` — active vs historical paths and migration notes.
5. `chunk_experiment/models.py`, `factory.py`, `evaluation.py`,
   `retrieval.py`, `datasets.py` — invariant and API contracts.
6. `tests/`, `configs/factory_smoke.json` and
   `benchmarks/fixtures/` — reproducible baseline.

Directory ownership:

| Path | Status | Change policy |
| --- | --- | --- |
| `chunk_experiment/` | **Maintained** | Production-grade library changes with tests |
| `experiments/run_factory.py` | **Maintained** | Experiment orchestration only; keep algorithms in the library |
| `experiments/visualize_factory.py` | **Maintained** | Read result artifacts; never silently change metrics |
| `configs/` | **Maintained** | Reproducible, explicit experiment configurations |
| `benchmarks/fixtures/` | **Maintained fixtures** | Small, independently labelled examples |
| `tests/` | **Maintained** | Deterministic tests and failure-case coverage |
| `docs/`, `README.md` | **Maintained docs** | Keep interfaces, commands and caveats aligned with code |
| `src/`, `util/` | Historical research | Preserve old interfaces unless migration is separately requested |
| Root historical scripts | Legacy compatibility | Do not import them into maintained library code |
| `notebooks/legacy/`, `archive/` | Archival | Avoid bulk edits; do not overwrite original evidence |
| `data/`, `chunk_papers/` | Large historical assets | Do not move, normalize, re-encode, commit generated copies or bulk rewrite |

Use `docs/REPOSITORY_LAYOUT.md` for canonical paths. Some historical
notebooks have machine-specific absolute paths and may not run unmodified.

## 2. Mandatory source-alignment contract

`Chunk(text, start, end, metadata)` uses **zero-based, half-open Unicode
character offsets into the exact original Python string**.

For each emitted chunk:

```python
assert 0 <= chunk.start < chunk.end <= len(source)
assert chunk.text == source[chunk.start:chunk.end]
```

For every nonempty document under evaluation:

- Report source coverage; fail a factory run if coverage is below 1.0.
- Track overlapping source positions explicitly, never by fuzzy string matching.
- Do not drop whitespace, separators or punctuation while claiming full source
  alignment.
- Do not silently normalize HTML, PDF, Markdown or Unicode and then reuse
  original offsets; preserve a tested source-map contract if preprocessing is
  unavoidable.
- Ensure deterministic ordering, bounded chunk sizes, and no state leakage
  between documents.
- Reject invalid configurations, out-of-range spans and NaN/Inf embeddings.

When adding a new chunker, first specify how it preserves the source offsets
and maximum length budget. Do not merge an implementation that cannot satisfy
these invariants.

## 3. Factory Mode — how to extend the library

1. Implement a strategy in a **new focused module** under
   `chunk_experiment/`, returning `list[Chunk]`.
2. Accept a `LengthMetric` and any embedding provider by dependency injection.
   No hidden downloads, API calls, global caches or mutable module state.
3. Add a typed configuration field/algorithm key to `ChunkerConfig` and a
   construction branch in `build_chunker`. Validate ranges and unsupported
   combinations before processing data.
4. Keep algorithm-specific metadata deterministic and JSON serializable.
5. Add tests for empty/whitespace-only inputs, Chinese/English/mixed text,
   long unbroken spans, boundary placement, budgets, overlap and repeatability.
6. Extend the matrix configuration, benchmark integration and README/table.
7. Compare against `fixed` and `recursive` on **identical data, length units,
   retrieval embedding backend and Top-K**.

If a method alters **index embeddings** (e.g. late chunking), **retrieval
context** (e.g. small-to-big) or **query/rerank behavior**, represent that as
an explicit pipeline stage. Do not present such changes as a pure splitter
ablation. Keep confounded variables fixed or disclose them in the experiment.

## 4. Data, gold labels and evaluation discipline

Data contract:

- Corpus: UTF-8 JSONL manifest with `document_id`, `path`, `split`,
  `language`, `domain` and ideally the real file `sha256`.
- Gold: UTF-8 JSONL containing `case_id`, `query` and
  `relevant_spans=[{document_id,start,end}]`.
- Every gold span refers to an unmodified source document. Never derive a
  gold label from any specific chunker's output.
- Validate duplicate IDs, cross-split exact duplicates, bad paths, checksums,
  missing referenced documents and invalid spans. For real data also check
  **near-duplicate leakage**, licenses and annotation provenance.
- Select/tune parameters on development data; reserve test data for final
  evaluation. Do not optimize directly on the final test labels.

Evaluation tiers:

1. **Intrinsic:** coverage, overlap duplication, size compliance and
   boundary/length diagnostics.
2. **Retrieval:** Hit@K, MRR@K and evidence-character Recall@K with immutable
   source-span gold. Store **every per-query ranking**, not just the average.
3. **End-to-end RAG (future):** answer correctness, citation support,
   hallucinations/unsupported claims and retrieval-vs-generation error
   decomposition.

The bundled `semantic-hash` and hash retriever are **lexical, deterministic
CI baselines**, not semantic models. Never report them as proof that a
semantic strategy is better. For credible research, pin the real
embedding model/version/tokenizer, index settings and data snapshot. Report
query-level paired differences and uncertainty (e.g. paired bootstrap) before
claiming a statistically supported win.

## 5. Experiment reproducibility and artifact policy

A reproducible run should record:

- Git SHA, dataset manifest SHA, gold SHA and configuration SHA;
- algorithm key and all effective hyperparameters;
- length unit and exact tokenizer/model identity where used;
- embedder/retriever model, version, seed, split, Top-K;
- runtime environment, chunk construction latency, indexing latency,
  retrieval latency, number of indexed chunks;
- per-document diagnostics and per-query rankings / gold metrics.

Write generated results to `runs/<run-name>/`, which is ignored by Git.
Do not overwrite historical JSON or figures under `archive/`. Avoid
committing large generated indexes, raw vectors or model weights by default.
Keep `run.jsonl` structured and machine-readable.

When benchmarking runtime, isolate model startup/cold cache from steady-state
work and label latency appropriately. Do not compare results from different
hardware or backend caches without disclosure.

## 6. Logging and error handling

Use Loguru with consistent context keys. Typical pattern:

```python
from loguru import logger

run_logger = logger.bind(run_id=run_id, algorithm=config.algorithm)
run_logger.info("chunked document_id={} chunks={}", document_id, len(chunks))
```

The factory runner sets up a serialized JSONL sink. Keep logging at the
experiment boundary; library objects should not silently add global sinks.
Use clear exceptions on invalid user input or corrupted data. Never swallow
exceptions to manufacture a success score.

**Do not log:** raw private corpus content, full queries containing sensitive
data, HTTP authorization headers, tokens/cookies, private model endpoints or
arbitrary entire embedding response bodies. In published artifacts use
anonymized IDs when datasets are sensitive.

## 7. Required validation commands

Run from the repository root (the same checks used by GitHub Actions):

```bash
python -m pip install -e ".[dev]"
ruff check chunk_experiment tests experiments/run_factory.py \
  experiments/visualize_factory.py scripts/github_fetch_checkout.py
python -m pytest
python experiments/run_factory.py \
  --manifest benchmarks/fixtures/manifest.jsonl \
  --dataset-root benchmarks/fixtures/documents \
  --gold benchmarks/fixtures/gold.jsonl \
  --config configs/factory_smoke.json \
  --output-dir runs/ci-smoke
```

Run visualization when changing plots (requires `pip install -e ".[viz]"`):

```bash
python experiments/visualize_factory.py runs/ci-smoke/results.json
```

GitHub Actions tests Python 3.10/3.11/3.12. Do not assert CI passed
unless the **latest branch HEAD** has a completed successful run.

For any layout change, verify the README and AGENTS links, current paths,
archived-path compatibility, and the CLI/runner commands; do not introduce
broken copy-paste examples.

## 8. Git, review and delivery

- Work on a task-specific feature branch unless explicitly instructed
  otherwise. Never force-push to `main`.
- Check current `main`, active PRs and the latest commit before editing;
  previous branches may already have been merged.
- Commit a coherent concern at a time; prefer
  `docs:`, `chore:`, `test:`, `fix:` and `feat:` prefixes.
- Do not reorganize historic `src/`, `util/`, notebooks or large datasets
  merely to satisfy an aesthetic convention. Document compatibility and
  migrate only where a reliable test/adapter exists.
- Never delete ground-truth data, original research reports, historical
  experiment results or licensed research papers without explicit scope.
- When a task requires new datasets or real embedding access that are not
  provided, complete the offline fixture path, document the missing evidence,
  and **do not fabricate results**.
- Record the exact files changed, validation outcome and any remaining
  scientific limitations in the PR.

## 9. Preferred next research work

Prioritize issues that materially affect experimental validity:

1. Curate licensed, source-mapped multilingual evaluation corpora with
   independently reviewed gold spans and leakage checks.
2. Compare real embedding and exact-tokenizer baselines under fixed
   retrieval settings, including paired per-query confidence intervals.
3. Add structural Markdown/HTML/AST boundaries without losing source
   alignment, especially for tables, code, formulas and CJK punctuation.
4. Separate splitter, embedding, context enrichment, retriever and
   reranker ablations; add explicit cost and index-footprint reporting.
5. Build error-analysis visualizations (lost evidence, split entities,
   too-small chunks, redundant context, boundary failures).

Quality means **reproducible evidence and explainable trade-offs**, not the
number of implemented chunkers.
