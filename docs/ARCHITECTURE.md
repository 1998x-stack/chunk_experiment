# Architecture v2

The repository now separates **chunking mechanics** from **experiment dependencies**.

```text
input document
   │
   ├── LengthMetric ───────────────┐
   │                               │
   ├── RecursiveChunker            ├──> Chunk(start, end, text, metadata)
   │                               │                │
   └── SemanticChunker             │                └──> evaluation
          │                        │
          └── EmbeddingProvider ───┘
               ├── HashEmbeddingProvider (deterministic lexical baseline / CI)
               └── HttpEmbeddingProvider (real model service)
```

## Design invariants

1. **Source alignment** — every chunk is an exact slice of the original document: `chunk.text == source[start:end]`.
2. **Explicit length unit** — algorithms receive a `LengthMetric`; characters and approximate tokens are never silently mixed.
3. **No random semantic baseline** — CI uses deterministic lexical hashing; real semantic experiments inject a real embedding service/model.
4. **Cosine really means cosine** — similarity normalizes vectors and validates shapes.
5. **No hidden document-level mutation** — adaptive thresholds are computed per call and are not written back into configuration.
6. **Evaluation is algorithm-neutral** — coverage, duplication, size compliance, and length dispersion are computed from `Chunk` objects.

## Legacy compatibility

The original `src/`, `util/`, notebooks and result JSON files are retained as historical research artifacts. Their public classes are not removed in this change. New experiments should use the `chunk_experiment` package and the CLI so the length metric, embedding backend and evaluation contract are explicit.

## Benchmark guidance

The historical ablation runner used mocked embeddings with repeated synthetic vectors. Those results remain useful for execution-flow and chunk-size sensitivity checks, but should not be interpreted as a benchmark of semantic coherence. For semantic claims, rerun with a documented embedding model, immutable dataset snapshot, fixed parameters, and retrieval-grounded downstream metrics such as Recall@K / MRR / nDCG where labels exist.
