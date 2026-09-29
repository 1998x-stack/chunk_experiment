# Architecture v4 — hybrid and parent/child RAG evaluation

v4 extends the v3 source-span evaluation contract without changing its labels.

```text
                    EvaluationDataset
                documents + gold spans
                         │
                  ChunkingStrategy
                         │
             ┌───────────┴───────────┐
             │                       │
          flat chunks          ParentChildIndex
                                     │
                               child chunks
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
     DenseRetriever             BM25Retriever              both
          │                          │                          │
          └──────────────────────────┴───────────RRF────────────┘
                                     │
                               Retriever protocol
                                     │
                          optional parent collapse
                                     │
                          source-aligned results
                                     │
       HitRate / Precision / SpanRecall / MRR / nDCG
                                     +
        latency / context chars / unique chars / duplication
```

## Why rank fusion

Dense cosine scores and BM25 scores do not share a calibrated numeric scale. v4 therefore uses
weighted Reciprocal Rank Fusion (RRF) instead of adding raw scores. This keeps the combination
auditable and avoids accidental dominance caused purely by score magnitude.

## Parent/child contract

A parent chunk and every child chunk are exact slices of the same original document. Children are
used as retrieval units. Retrieved child IDs are mapped back to deduplicated parents before
evaluation and context-cost measurement.

This directly exposes the trade-off:

- smaller children can improve evidence matching;
- larger parents can improve downstream answer context;
- larger returned context also raises context cost.

The evaluator reports both retrieval quality and returned-context size so the trade-off is visible.

## Scope

BM25 and dense search are exact in-memory implementations for controlled experiments. They are
not production-scale ANN or inverted-index engines. Production backends can implement the same
`Retriever` protocol later without changing golden labels or metrics.
