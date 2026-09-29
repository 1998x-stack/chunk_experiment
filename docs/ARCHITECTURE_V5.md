# Architecture v5 — two-stage retrieval and reranking

v5 separates retrieval failures into candidate-generation failures and ranking failures.

```text
source documents
      │
 chunk strategy
      │
 retrieval index
      │
 candidate retriever ───────────── candidate@K metrics
      │
 candidates
      │
 rerank score provider
      │
 reranker ──────────────────────── ranking lift metrics
      │
 final top-K
      │
 final retrieval metrics + context cost
```

## Failure attribution

For every query, v5 records three views:

1. **Candidate ceiling at candidate-K** — whether the relevant source evidence entered the
   candidate set at all.
2. **Candidate prefix at final-K** — the ranking quality before reranking at the same final
   depth.
3. **Reranked final-K** — the ranking quality after reranking.

This makes two common failure modes distinguishable:

- low candidate span recall means the reranker never had access to the evidence;
- high candidate recall but weak final ranking points to a ranking/reranking problem.

## Reranker contract

A `RerankScoreProvider` scores a query against a batch of candidate texts. The generic
`ScoreProviderReranker` converts those scores into stable `SearchResult` ranks while
preserving the original source-aligned chunks.

The built-in lexical overlap scorer is intentionally a deterministic CI baseline. Real
cross-encoders or hosted rerank APIs can implement the same score-provider protocol without
changing evaluation contracts.

## Metrics

The two-stage evaluator reports:

- candidate Hit Rate and Span Recall at candidate-K;
- MRR/nDCG of the unreordered candidate prefix at final-K;
- final Hit Rate, Span Recall, MRR and nDCG;
- MRR lift and nDCG lift from reranking;
- candidate retrieval latency and reranking latency separately.

Existing v4 context-cost metrics remain available on the final returned results.
