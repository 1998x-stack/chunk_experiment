# Reranking evaluation

## Why two stages are measured separately

A final retrieval score cannot explain whether an error came from candidate generation or
reranking. v5 therefore keeps the first-stage candidate list observable.

A reranker can only reorder evidence that was retrieved. If candidate SpanRecall@K is zero,
no downstream reranking method can recover the missing evidence without changing candidate
retrieval or increasing candidate depth.

## Built-in deterministic baseline

`LexicalOverlapScoreProvider` scores candidate text by query-term coverage with a small
precision component. It is useful for:

- CI;
- regression tests;
- validating reranking plumbing;
- controlled demonstrations of ranking lift.

It is not a cross-encoder and must not be used as evidence of semantic reranking quality.

## Plugging in real rerankers

Implement:

```python
class MyScoreProvider:
    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        ...
```

Then pass the provider to `ScoreProviderReranker`. The provider should return exactly one
numeric score per candidate, in input order. Cardinality is validated.

This interface is suitable for local cross-encoders, hosted reranking APIs and other pairwise
or listwise scoring systems.

## Regression gates

The CLI can gate both final quality and stage quality:

```bash
chunk-retrieval-eval examples/retrieval_eval/golden.json \
  --chunker markdown \
  --chunk-size 20 \
  --retrieval-mode hybrid \
  --reranker lexical \
  --rerank-candidate-k 10 \
  --top-k 1 3 \
  --min-candidate-span-recall 1.0 \
  --min-span-recall 1.0
```

A candidate-recall gate failing is intentionally distinguishable from a final retrieval gate.
