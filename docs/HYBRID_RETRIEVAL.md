# Hybrid retrieval and context-cost evaluation

## BM25

`BM25Retriever` is dependency-free and deterministic. Its tokenizer recognizes Latin/numeric
terms and individual CJK characters, which makes it suitable as a stable lexical experiment
baseline without assuming whitespace tokenization for Chinese.

## Dense + sparse fusion

`HybridRetriever` combines any retrievers using weighted Reciprocal Rank Fusion:

```text
score(document) = Σ weight_i / (rrf_k + rank_i)
```

Only rank positions are fused. Dense cosine values and BM25 raw scores are never directly added.

## Parent/child retrieval

`build_parent_child_index()` creates larger parent chunks and source-aligned children. A
`ParentChildRetriever` searches the children and collapses results to unique parents.

The default evaluation path considers all child candidates when collapsing. This favors
experimental correctness over production latency. A bounded candidate set can be supplied when
studying latency/quality trade-offs.

## Cost metrics

`evaluate_retriever_profiled()` records, at a selected K:

- mean query latency;
- p95 query latency;
- returned context characters;
- unique source characters;
- duplication ratio caused by overlapping returned contexts.

Latency is environment-dependent and should only be compared within controlled runs. Context
size and duplication are deterministic for a fixed retrieval result.

## Interpretation

No single scalar is treated as “the best chunking strategy.” A useful experiment should examine
retrieval quality together with context size, duplication and latency, then choose operating
points appropriate to the target RAG system.
