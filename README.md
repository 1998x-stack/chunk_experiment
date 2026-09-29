# chunk_experiment

> 可复现、可审计、面向真实 RAG 检索效果的文本切分实验框架。  
> Reproducible chunking and retrieval experiments with source-aligned evidence labels.

## Current architecture

The repository has evolved in four additive stages:

- **v2 — trustworthy chunking core**: exact source offsets, explicit length metrics,
  deterministic baselines, real cosine similarity and reproducible experiments.
- **v3 — retrieval-grounded evaluation**: Markdown-aware chunking, versioned golden sets,
  source-span labels, HitRate/Precision/SpanRecall/MRR/nDCG and CI quality gates.
- **v4 — RAG retrieval architecture experiments**: BM25, dense + sparse RRF fusion,
  parent/child retrieval and latency/context-cost metrics.

Historical notebooks, scripts and result files remain available for research provenance.

## Design invariants

- Every returned chunk is an exact source slice: `chunk.text == source[start:end]`.
- Gold relevance labels point to stable document spans, never transient chunk IDs.
- Character/token-like length units are explicit and injectable.
- Retrieval strategies share one `Retriever` protocol.
- Hybrid retrieval fuses **ranks**, not incomparable dense/BM25 raw scores.
- Parent/child retrieval searches small evidence units but returns larger source-aligned context.
- Context cost is measured alongside retrieval quality.
- Hash embeddings are a deterministic CI/plumbing baseline, not evidence of semantic quality.

See:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md)
- [docs/ARCHITECTURE_V4.md](docs/ARCHITECTURE_V4.md)
- [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md)
- [docs/HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md)
- [docs/REVIEW.md](docs/REVIEW.md)

## Install

Core development environment:

```bash
python -m pip install -e ".[dev]"
```

HTTP embedding support:

```bash
python -m pip install -e ".[http]"
```

Historical scripts and dependencies:

```bash
python -m pip install -e ".[legacy,dev]"
```

## Chunking CLI

Recursive:

```bash
chunk-experiment input.txt \
  --algorithm recursive \
  --chunk-size 500 \
  --overlap 100
```

Markdown structure-aware:

```bash
chunk-experiment guide.md \
  --algorithm markdown \
  --chunk-size 256
```

Semantic chunking with a real HTTP embedding service:

```bash
chunk-experiment input.txt \
  --algorithm semantic-http \
  --embedding-url "$EMBEDDING_URL" \
  --chunk-size 200
```

## Retrieval evaluation

Dense retrieval:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-mode dense \
  --contextual-headings \
  --top-k 1 3
```

Dependency-free BM25:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-mode bm25 \
  --top-k 1 3
```

Dense + BM25 weighted RRF:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-mode hybrid \
  --contextual-headings \
  --top-k 1 3 \
  --gate-k 3 \
  --min-hit-rate 1.0 \
  --min-span-recall 1.0
```

Parent/child retrieval uses small chunks for matching and larger parents for returned context:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 12 \
  --parent-chunk-size 40 \
  --retrieval-mode hybrid \
  --contextual-headings \
  --top-k 1 3 \
  --cost-k 3
```

The output contains retrieval metrics plus mean/p95 query latency, returned context characters,
unique context characters and context duplication ratio.

## Experiment runners

Chunk-size benchmark:

```bash
python experiments/run_v2.py data/general/en \
  --chunk-sizes 200 500 1000 \
  --output v2_benchmark_results.json
```

Retrieval-grounded v3 matrix:

```bash
python experiments/run_retrieval_v3.py examples/retrieval_eval/golden.json
```

v4 RAG architecture matrix:

```bash
python experiments/run_retrieval_v4.py examples/retrieval_eval/golden.json \
  --strategies recursive markdown \
  --retrieval-modes dense bm25 hybrid \
  --chunk-sizes 12 20 \
  --parent-multipliers 1 3
```

## Testing

```bash
pytest
ruff check chunk_experiment tests experiments
```

CI runs Python 3.10, 3.11 and 3.12 and includes an end-to-end retrieval golden-set gate.

## Historical results

The original `src/`, `util/`, notebooks and historical JSON reports are intentionally retained.
Older semantic comparisons used mocked/synthetic embeddings and should be treated as
execution-flow or parameter-sensitivity artifacts, not as evidence that one semantic strategy
has superior semantic coherence.

## License

MIT — see [LICENSE](LICENSE).
