# chunk_experiment

> 可复现、可审计、面向真实 RAG 检索效果的文本切分与检索实验框架。  
> Reproducible chunking, retrieval and reranking experiments with production benchmark support.

## Current architecture

The repository has evolved in six additive stages:

- **v2 — trustworthy chunking core**: exact source offsets, explicit length metrics,
  deterministic baselines, real cosine similarity and reproducible experiments.
- **v3 — retrieval-grounded evaluation**: Markdown-aware chunking, versioned golden sets,
  source-span labels, HitRate/Precision/SpanRecall/MRR/nDCG and CI quality gates.
- **v4 — RAG retrieval architecture experiments**: BM25, dense + sparse RRF fusion,
  parent/child retrieval and latency/context-cost metrics.
- **v5 — two-stage retrieval and reranking**: candidate-ceiling diagnostics, explicit
  reranking, ranking-lift metrics and separate candidate/rerank latency.
- **v6 — real-model production benchmark layer**: generic HTTP reranking, model
  fingerprints, embedding/rerank caches, usage accounting, cold/warm latency, QPS,
  user-supplied cost estimates and Pareto-frontier reporting.

## Design invariants

- Every returned chunk is an exact source slice: `chunk.text == source[start:end]`.
- Gold relevance labels point to stable document spans, never transient chunk IDs.
- Character/token-like length units are explicit and injectable.
- Retrieval strategies share one `Retriever` protocol.
- Hybrid retrieval fuses ranks rather than incomparable dense/BM25 raw scores.
- Parent/child retrieval searches small evidence units but returns larger source-aligned context.
- Candidate retrieval and reranking failures are measured separately.
- Model identity is fingerprinted without storing credentials or raw endpoint URLs.
- Cache usage and backend usage are reported separately.
- Cost estimates use explicit user-supplied pricing and are never presented as vendor billing.
- Hash embeddings and lexical reranking remain deterministic CI baselines, not semantic models.

See:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ARCHITECTURE_V3.md](docs/ARCHITECTURE_V3.md)
- [docs/ARCHITECTURE_V4.md](docs/ARCHITECTURE_V4.md)
- [docs/ARCHITECTURE_V5.md](docs/ARCHITECTURE_V5.md)
- [docs/ARCHITECTURE_V6.md](docs/ARCHITECTURE_V6.md)
- [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md)
- [docs/HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md)
- [docs/RERANKING.md](docs/RERANKING.md)
- [docs/PRODUCTION_BENCHMARK.md](docs/PRODUCTION_BENCHMARK.md)
- [docs/REVIEW.md](docs/REVIEW.md)

## Install

Core development environment:

```bash
python -m pip install -e ".[dev]"
```

HTTP embedding / reranking support:

```bash
python -m pip install -e ".[http]"
```

## Retrieval evaluation

Example deterministic two-stage evaluation:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-mode bm25 \
  --reranker lexical \
  --rerank-candidate-k 3 \
  --top-k 1 3
```

## v6 production benchmark

Offline deterministic smoke run:

```bash
python experiments/run_production_v6.py examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-modes dense bm25 hybrid \
  --rerankers none lexical \
  --rerank-candidate-k 3 \
  --top-k 3 \
  --cold-passes 1 \
  --warm-passes 2
```

Real HTTP embedding + reranking:

```bash
export EMBEDDING_TOKEN="..."
export RERANK_TOKEN="..."

python experiments/run_production_v6.py examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 256 \
  --retrieval-modes dense hybrid \
  --embedding-backend http \
  --embedding-url "$EMBEDDING_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-provider-label internal-embedding \
  --embedding-endpoint-label prod \
  --embedding-bearer-env EMBEDDING_TOKEN \
  --rerankers http \
  --rerank-url "$RERANK_URL" \
  --rerank-model "$RERANK_MODEL" \
  --rerank-provider-label internal-rerank \
  --rerank-endpoint-label prod \
  --rerank-bearer-env RERANK_TOKEN \
  --rerank-candidate-k 20 \
  --top-k 5 \
  --embedding-usd-per-1k-items 0.0 \
  --rerank-usd-per-1k-pairs 0.0 \
  --output production_benchmark_v6.json
```

The output intentionally excludes raw endpoint URLs and authorization values. It contains model
fingerprints, index-build usage, quality metrics, candidate/rerank stage metrics, cold/warm p95
and p99 latency, QPS, cache hit rates, backend item/character counts, optional cost estimates and
a non-dominated Pareto frontier.

## Experiment runners

- `experiments/run_v2.py`: chunk-size experiments.
- `experiments/run_retrieval_v3.py`: retrieval-grounded chunking matrix.
- `experiments/run_retrieval_v4.py`: hybrid and parent/child matrix.
- `experiments/run_rerank_v5.py`: candidate-depth and reranking matrix.
- `experiments/run_production_v6.py`: real-model quality/latency/cache/cost benchmark.

## Testing

```bash
pytest
ruff check chunk_experiment tests experiments
```

CI runs Python 3.10, 3.11 and 3.12, the retrieval golden-set gate, and a deterministic v6
production-benchmark smoke test.

## Historical results

The original `src/`, `util/`, notebooks and historical JSON reports are intentionally retained.
Older semantic comparisons used mocked/synthetic embeddings and should be treated as
execution-flow or parameter-sensitivity artifacts, not as evidence that one semantic strategy
has superior semantic coherence.

## License

MIT — see [LICENSE](LICENSE).
