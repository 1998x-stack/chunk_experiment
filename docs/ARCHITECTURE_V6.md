# Architecture v6 — real-model and production benchmark layer

v6 adds a production-oriented measurement boundary around the v2–v5 experiment stack.

```text
dataset + source-span gold
        │
   chunk strategy
        │
  retrieval index ───────────── index build time / embedding usage
        │
 candidate retriever
        │
     candidates
        │
   rerank provider
        │
    final top-K
        │
 ┌──────┼───────────────────────────────────────────────┐
 │      │               │              │               │
quality stage metrics  context cost   latency/QPS   provider usage
 │                                                     │
 └────────────── model identity + pricing ─────────────┘
                         │
                   Pareto frontier
```

## Model identity

`ModelIdentity` records only reproducibility-safe fields: kind, provider label, model,
revision and endpoint label. A SHA-256 fingerprint is derived from those fields.

Raw endpoint URLs, bearer tokens and arbitrary authorization headers are deliberately excluded
from benchmark output.

## Caching

`CachedEmbeddingProvider` caches exact text embeddings by model fingerprint and text hash.
`CachedRerankScoreProvider` caches query/document scores by model fingerprint and pair hash.

Usage distinguishes requested items from backend items so cache effects are directly visible.

## Real reranking

`HttpRerankScoreProvider` uses a generic HTTP contract and accepts either a score array or
indexed result objects. It does not depend on a vendor SDK.

## Benchmark phases

The production runner reports index build separately from query serving. Query serving is then
measured in cold and warm phases. Each phase reports elapsed time, QPS, mean latency, p95 and p99.

Warm-cache performance must not be compared with a cold-cache run as if they were equivalent.

## Pareto reporting

The runner returns non-dominated configurations across retrieval quality, span recall, warm
latency and optional estimated cost. It intentionally does not collapse those dimensions into
one weighted score or declare a universal winner.
