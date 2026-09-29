# Production benchmark guide

## HTTP rerank contract

Request:

```json
{
  "model": "rerank-model",
  "query": "query text",
  "documents": ["candidate A", "candidate B"]
}
```

Accepted response forms:

```json
{"scores": [0.1, 0.9]}
```

or:

```json
{
  "results": [
    {"index": 1, "relevance_score": 0.9},
    {"index": 0, "relevance_score": 0.1}
  ]
}
```

Every input document must receive exactly one numeric score.

## Credentials

The production runner accepts bearer-token *environment variable names*. It reads the secret at
runtime and never writes the value to the result JSON. Prefer an internal proxy or secret manager
in real deployments.

## Cache semantics

The v6 cache is intentionally in-memory and scoped to one benchmark process. It is designed for
controlled cold/warm measurements rather than as a production distributed cache.

`requested_items` counts caller-visible work. `backend_items` counts work that reached the
underlying model after deduplication/cache lookup. The difference is the measurable cache effect.

## Cost estimates

Pricing values are optional and user supplied:

- embedding USD per 1,000 backend text items;
- rerank USD per 1,000 backend query/document pairs.

These are normalized experiment estimates, not invoice reconstruction. If a provider bills by
tokens, compute an appropriate normalized item rate externally or leave pricing at zero and use
the reported backend character/item counts.

## Comparing runs

Prefer controlled comparisons with the same:

- dataset fingerprint;
- model fingerprints;
- machine/runtime environment;
- cold/warm pass counts;
- top-K and candidate-K;
- chunking parameters.

Latency results from different hardware, networks or service load should not be interpreted as
directly comparable.
