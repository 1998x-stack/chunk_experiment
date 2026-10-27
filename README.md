# chunk_experiment

> 可复现的 RAG 文本切分实验框架：递归切分、embedding 驱动语义切分、统一指标、CLI 与目录级 benchmark。  
> Reproducible chunking experiments for RAG with explicit length metrics, embedding backends and source-aligned evaluation.

## Why this refactor

The repository started as a collection of research scripts and notebooks. That was useful for exploration, but several implementation details made cross-run conclusions hard to trust:

- legacy semantic experiments used synthetic/mock embeddings;
- enhanced defaults could generate random embeddings;
- "token count" was based on whitespace, which is not meaningful for unsegmented Chinese;
- cosine similarity was named as such but the legacy path used a raw dot product;
- adaptive thresholds could be written back to instance state and leak across documents;
- chunk text did not have a single, auditable offset/metadata contract.

The v2 core keeps the historical artifacts, but makes those experimental dependencies explicit.

## v2 architecture

```text
document
  ├─ LengthMetric
  ├─ RecursiveChunker
  └─ SemanticChunker ── EmbeddingProvider
             │
             └──────────────> Chunk(text, start, end, metadata)
                                      │
                                      └─ evaluate_chunks(...)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/REVIEW.md](docs/REVIEW.md).

### Core invariants

- **Source aligned**: `chunk.text == source[chunk.start:chunk.end]`.
- **Explicit units**: character length and token-like length are separate metrics.
- **Deterministic offline baseline**: hash embeddings are reproducible and clearly labelled lexical, not semantic.
- **Real cosine similarity**: vectors are normalized/validated.
- **No hidden state mutation** across documents.
- **Algorithm-neutral metrics**: coverage, duplication, size compliance and dispersion.

## Install

For the new core:

```bash
python -m pip install -e ".[dev]"
```

For the historical scripts as well:

```bash
python -m pip install -e ".[legacy,dev]"
```

HTTP embedding support is optional:

```bash
python -m pip install -e ".[http]"
```

## CLI

Recursive splitting:

```bash
chunk-experiment data/general/en/example.txt \
  --algorithm recursive \
  --chunk-size 500 \
  --overlap 100 \
  --output recursive.json
```

Deterministic lexical baseline for smoke tests / CI:

```bash
chunk-experiment data/general/en/example.txt \
  --algorithm semantic-hash \
  --chunk-size 200 \
  --output semantic_hash.json
```

Real HTTP embedding service:

```bash
chunk-experiment input.txt \
  --algorithm semantic-http \
  --embedding-url "$EMBEDDING_URL" \
  --chunk-size 200
```

## Reproducible directory benchmark

```bash
python experiments/run_v2.py data/general/en \
  --chunk-sizes 200 500 1000 \
  --include-semantic-hash \
  --output v2_benchmark_results.json
```

The result records the document SHA-256, parameters, runtime environment and unified metrics for every run.

## v3 Factory Mode + retrieval evaluation

New experiments should use the checked-in [Factory Pipeline Guide](docs/FACTORY_PIPELINE.md).
The source-aligned experiment layer now includes **fixed-window, recursive,
Markdown-heading, lexical-hash semantic and HTTP semantic chunking**.

Create a JSONL document manifest with `document_id`, relative `path`, `split`,
`language`, `domain` and an optional SHA-256; pair it with JSONL queries
and source-character evidence spans. Then run:

```bash
pip install -e ".[dev,viz]"
python experiments/run_factory.py \
  --manifest benchmarks/my_corpus.jsonl \
  --dataset-root benchmarks/documents \
  --gold benchmarks/my_gold.jsonl \
  --config configs/factory_smoke.json \
  --output-dir runs/my-first-benchmark
python experiments/visualize_factory.py runs/my-first-benchmark/results.json
```

The runner validates input checksums and gold offsets, enforces 100% source
coverage, records Loguru serialized JSONL event traces and preserves raw
per-query rankings. It reports Hit@K, MRR@K and non-double-counting evidence
character recall. Its default lexical-hash retrieval is a **CI/smoke baseline
only**; use a fixed real embedding service for semantic-quality conclusions.
The example config is a template and does not include curated gold labels.

## Reconstruct a local working tree by serial GitHub fetch

This environment cannot download the repository's roughly 946 MB of tracked
files. A sequential per-blob downloader with SHA verification and Loguru
progress logs is included for execution on your own computer:

```bash
export GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
python scripts/github_fetch_checkout.py \
  --repo 1998x-stack/chunk_experiment \
  --ref feat/factory-retrieval-eval-v3 \
  --target ./chunk-experiment-local \
  --exclude-prefix data \
  --exclude-prefix chunk_papers
```

Remove the exclusions for all tracked files. Re-running the command verifies
existing blobs and fetches only missing ones. The downloader creates a
**working-tree snapshot**, not a clone of Git history. For full Git history,
use `git clone --filter=blob:none` with normal Git.

## Testing

```bash
pytest
ruff check chunk_experiment tests
```

CI runs the v2 suite on Python 3.10, 3.11 and 3.12.

## Historical results

The following files are preserved as historical research artifacts:

- `EXPERIMENT_SUMMARY.md`
- `ENHANCEMENT_SUMMARY.md`
- `algorithm_comparison_results.json`
- `param_ablation_results.json`
- notebooks and the original `src/` / `util/` implementations

Those historical semantic comparisons were produced with mocked/synthetic embeddings, so they are useful for execution-flow and parameter-sensitivity inspection, **not as evidence that one semantic strategy has higher semantic coherence than another**.

New runs from `ablation_experiments.py` use a deterministic lexical-hash backend and write `*_v2.json` files so the historical outputs are not overwritten.

For semantic-quality conclusions, use a documented real embedding model and retrieval-grounded downstream metrics (for example Recall@K, MRR or nDCG when labels are available).

## License

MIT — see [LICENSE](LICENSE).
