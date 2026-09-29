# Architecture v3 — retrieval-grounded chunking

v2 established deterministic, source-aligned chunking primitives. v3 adds the layer required to
answer the next question: **which chunking strategy improves retrieval for the queries we care
about?**

```text
                       ┌──────────────────────────┐
                       │ EvaluationDataset        │
                       │ docs + query gold spans │
                       └────────────┬─────────────┘
                                    │
                       dataset SHA-256 fingerprint
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
  RecursiveChunker          MarkdownChunker          SemanticChunker
          │                  heading metadata                │
          └─────────────────────────┼─────────────────────────┘
                                    │
                             IndexedChunk[]
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                plain text                heading-context text
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                             EmbeddingProvider
                                    │
                              DenseRetriever
                                    │
                          top-K SearchResult[]
                                    │
                source-span relevance matching
                                    │
          HitRate@K / Precision@K / SpanRecall@K / MRR / nDCG
                                    │
                           optional CI thresholds
```

## New invariants

1. **Gold labels are source evidence, not chunk IDs.** Chunk boundaries can change without
   invalidating benchmark labels.
2. **All compared strategies can share one `LengthMetric`.** This avoids comparing character
   and token budgets by accident.
3. **Context enrichment is representation-only.** Heading breadcrumbs may be used for
   embedding while returned chunks remain exact source slices.
4. **Duplicate overlap cannot manufacture relevance.** nDCG gain is deduplicated by annotated
   source span.
5. **Every evaluation dataset has a content fingerprint.** Results can be traced to the exact
   corpus/query snapshot.
6. **Quality can gate changes.** Retrieval thresholds can fail CI independently of generation.

## Scope boundaries

The v3 retriever uses exact dense matrix scoring on purpose. It is an evaluation component, not
a production ANN engine. ANN indexes, rerankers, hybrid BM25 fusion, late chunking token pooling
and LLM-based contextualization can be added behind explicit interfaces in later versions
without changing the golden-set or metric contracts.
