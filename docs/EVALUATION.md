# Evaluation design

## Why the old benchmark is not enough

Chunk count, average size, and wall-clock time are useful diagnostics, but they do
not show whether a chunking strategy improves retrieval or final RAG answers.

A robust benchmark separates three layers:

1. **chunk construction** — are chunks valid, stable, and structurally reasonable?
2. **retrieval** — does the system retrieve the evidence needed for the query?
3. **generation** — does the final answer use that evidence correctly?

## 1. Intrinsic chunk metrics

For every document/algorithm/configuration record:

- number of chunks;
- mean / median / standard deviation of size;
- min / max size;
- source coverage ratio;
- overlap duplication ratio;
- preferred-boundary alignment ratio;
- processing latency;
- deterministic-output check;
- constraint violations.

These metrics are diagnostic. They are not a substitute for retrieval accuracy.

## 2. Retrieval gold labels

Use source spans:

```json
{
  "case_id": "finance-001",
  "query": "What changed in Q2 revenue?",
  "relevant_spans": [[1830, 2014], [4420, 4518]]
}
```

Do not label a query with one algorithm's chunk ids. If chunk boundaries change,
chunk-id gold data becomes invalid or unfair.

The v2 core exposes span-based:

- Hit@K;
- Mean Reciprocal Rank;
- mean relevant-span Recall@K.

Future benchmark layers can add NDCG, precision, answer-support coverage, and
retrieval-context redundancy.

## 3. End-to-end RAG metrics

Measure at least:

- answer correctness / exact factual support;
- faithfulness to retrieved evidence;
- citation/evidence coverage;
- unsupported-claim rate;
- latency;
- embedding/reranking/generation cost.

Keep retriever failures separate from generator failures so an answer error can be
diagnosed rather than assigned to "RAG" as one opaque component.

## 4. Dataset slices

Report aggregate numbers and slices across:

- Chinese / English / mixed language;
- plain prose / Markdown / HTML / LaTeX / code;
- short / medium / long documents;
- fact lookup / multi-hop / aggregation / exact identifier queries;
- dense narrative vs tables/lists/code blocks;
- entity-heavy documents for future knowledge-graph experiments.

A single average can hide a chunker's failure mode.

## 5. Baseline matrix

Every new strategy should be compared against fixed baselines using the same corpus,
queries, embedding model, index, top-k, and reranker settings.

Suggested baseline families:

| Family | Purpose |
|---|---|
| fixed/window | lower-bound sanity check |
| boundary-aware | deterministic lexical baseline |
| recursive | classic hierarchical separators |
| structure-aware | headings / DOM / code syntax |
| semantic | embedding-driven boundaries |
| contextual enrichment | preserve document-level context |
| late chunking | contextualized chunk embeddings |

Do not vary chunking, embedding, retrieval strategy, and reranker at the same time
when attributing a gain to one component.

## 6. Reproducibility

Persist:

- dataset version/checksum;
- query/gold version;
- Git SHA;
- Python/dependency versions;
- chunker configuration;
- tokenizer/model name;
- embedding model;
- retrieval/reranking settings;
- random seed;
- raw per-query results.

Use bootstrap confidence intervals or repeated runs where model/API stochasticity is
present. Performance claims should include the evaluation population and configuration.

## Research references informing the redesign

- Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models:
  https://arxiv.org/abs/2409.04701
- Contextual Retrieval:
  https://www.anthropic.com/engineering/contextual-retrieval
- RAGChecker:
  https://arxiv.org/abs/2408.08067
- CoFE-RAG:
  https://arxiv.org/abs/2410.12248

These references motivate context-preserving embeddings, hybrid retrieval/reranking,
and evaluation that diagnoses stages rather than treating the pipeline as one score.
