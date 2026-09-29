# Retrieval evaluation in v3

v3 evaluates chunking at the point where it matters for RAG: **can retrieval recover the
source evidence for a query?** Chunk-local properties such as average size and semantic
coherence remain useful diagnostics, but they are not treated as a proxy for retrieval quality.

## Golden-set schema

A golden set is a version-controlled JSON manifest:

```json
{
  "schema_version": 1,
  "dataset_id": "my-eval",
  "documents": [
    {"id": "guide", "path": "guide.md", "metadata": {"format": "markdown"}}
  ],
  "queries": [
    {
      "id": "q1",
      "query": "How do I enable GPU acceleration?",
      "relevant_spans": [
        {"document_id": "guide", "start": 120, "end": 177, "relevance": 2}
      ]
    }
  ]
}
```

Gold labels are **document spans**, not precomputed chunk IDs. Chunk IDs change whenever a
strategy or chunk size changes, while the underlying answer evidence does not. The evaluator
maps retrieved chunks back to stable source spans using exact offsets.

The loader validates duplicate IDs, unknown documents, invalid ranges and out-of-bounds spans.
It also computes a SHA-256 fingerprint over document content and query labels so benchmark
outputs can be tied to an immutable evaluation snapshot.

## Metrics

For every configured `K`, v3 reports:

- **Hit Rate@K** — fraction of queries with at least one relevant retrieved chunk.
- **Precision@K** — fraction of returned top-K chunks overlapping a relevant source span.
- **Span Recall@K** — fraction of unique annotated evidence spans recovered in top K.
- **MRR** — reciprocal rank of the first relevant retrieved chunk.
- **nDCG@K** — ranking quality using optional graded relevance. A gold span contributes gain
  only once, so overlapping/duplicate chunks cannot inflate nDCG by repeatedly matching the
  same evidence.

This span-based formulation makes comparisons fairer across fixed, recursive, semantic and
structure-aware chunkers.

## Structure-aware Markdown retrieval

`MarkdownChunker` respects heading boundaries and records `heading_path` metadata. Oversized
sections fall back to recursive splitting while retaining the same heading breadcrumb.

For retrieval experiments, `contextual_heading_text()` can prepend that breadcrumb only to the
**embedding representation**:

```text
Installation > GPU
<source-aligned chunk text>
```

The source chunk itself is never rewritten, so citation offsets and auditability remain exact.

## Regression gates

`chunk-retrieval-eval` accepts minimum thresholds:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --contextual-headings \
  --top-k 1 3 5 \
  --gate-k 5 \
  --min-hit-rate 0.95 \
  --min-span-recall 0.90 \
  --min-mrr 0.80
```

A failed gate exits with status `2`, making retrieval quality usable as a CI regression check.

## Hash backend scope

The deterministic lexical hash backend is intentionally limited to CI, smoke tests and pipeline
reproducibility. It is not a semantic embedding model. For research conclusions, use a
documented real embedding backend and record its model/version alongside the dataset
fingerprint and chunking parameters.
