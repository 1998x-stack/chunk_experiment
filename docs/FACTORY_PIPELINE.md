# Factory Mode chunking experiments — v3

## Goal and baseline

The frozen audit base is `b8b204308b958b3f8028ea198d55f7f9251b9a44` (2026-10-09).
This project previously included extensive raw HTML, Chinese/English novels,
historical research notebooks and experimental outputs. They are **not** equivalent
to a labelled, independent retrieval benchmark. Preserve legacy results, but do
not use mocked vectors as evidence of semantic superiority.

Pipeline:
```text
dataset manifest + immutable files ----> strict UTF-8 reader ----> Document
                   |                                           |
         SHA256 / leakage audit                         Chunker Factory
                   |                          fixed / recursive / markdown-heading
gold queries ----> offset-span labels            semantic-hash / semantic-HTTP
                   |                                           |
                   |                              source-aligned Chunk + metrics
                   |                                           |
                   +--------------> fixed retriever + exact index
                                                |
                                     ranked source offsets
                                                |
                                 Hit@K / MRR@K / evidence-char recall
                                                |
                                  raw JSON + Loguru JSONL + figures
```

## Entry points

`chunk_experiment.factory.build_chunker(ChunkerConfig(...))`: explicit strategy
construction; no silent random embeddings; no hidden shared mutable chunker.

`chunk_experiment.datasets.load_corpus`: manifest JSONL with
`document_id, path, split, language, domain, sha256`. Paths are relative to
`--dataset-root`; duplicate IDs, cross-split exact duplicates, escaping the
data root, and mismatched hashes fail closed. UTF-8 errors also fail closed.

`chunk_experiment.datasets.load_gold`: query JSONL with
`case_id, query, relevant_spans: [{document_id,start,end}, ...]`.
Gold is labelled by character offsets in **unmodified** UTF-8 decoded source
text, not by algorithm-specific chunk IDs. Spans must be nonempty, within the
document, and from the requested split. Multiple disjoint evidence spans may
be used for a query.

`experiments/run_factory.py`: matrix runner and `run.jsonl`.
`experiments/visualize_factory.py`: quality/size scatter and per-query heatmap.

## Quickstart

Run the **checked-in fixture** before introducing a new corpus:

```bash
pip install -e '.[dev,viz]'
python experiments/run_factory.py \
  --manifest benchmarks/fixtures/manifest.jsonl \
  --dataset-root benchmarks/fixtures/documents \
  --gold benchmarks/fixtures/gold.jsonl \
  --config configs/factory_smoke.json \
  --output-dir runs/v3-smoke
python experiments/visualize_factory.py runs/v3-smoke/results.json
python -m pytest
ruff check chunk_experiment tests experiments/run_factory.py \
  experiments/visualize_factory.py scripts/github_fetch_checkout.py
```

The fixture contains labelled source spans but is **too small to support
statistical or semantic quality conclusions**. To run research experiments,
replace the manifest, dataset root and gold paths with your own curated,
licensed, independently labelled corpus. `runs/` is Git-ignored.

## Methodology and acceptance gates

1. **Source integrity (P0).** Every emitted chunk must equal the raw source
   slice `source[start:end]`, use valid nonempty offsets, and achieve
   coverage = 1.0 for each nonempty eligible document. Overlaps must not
   double-count evidence. Any discrepancy fails the run.
2. **Fairness (P0).** Compare methods on the same split, query IDs, evidence
   labels, length unit, top-k, retriever model, exact-index policy and hardware.
   Use identical preprocessing. Compare character-budget with character-budget
   and model-token budgets with the *same versioned tokenizer*. Approximate
   tokens are a smoke metric only.
3. **Gold quality (P0).** Manually sample query/span correspondence before
   drawing conclusions. Include unanswerable queries, long-context queries,
   numeric/table queries, multi-hop and exact identifiers in a separate
   upgraded annotation schema; do not fabricate supervision from chunk output.
4. **Paired analysis (P1).** Inspect *per-query* improvements/regressions.
   Report bootstrap confidence intervals of paired query-level deltas, repeat
   uncertain model or GPU timing measurements, and avoid declaring wins on
   a single small sample.
5. **Efficiency (P1).** Record chunk construction time, embedding/indexing time,
   per-query time and number of indexed chunks. Add measured peak memory,
   index bytes, token budgets and dollar/API cost before production claims.
6. **Provenance (P1).** Keep code SHA, config/dataset/gold SHA256, model identity,
   seed, Python/NumPy versions and raw per-query rankings. Loguru serialized
   JSONL logs identify every document/query without logging raw corpus text,
   private model endpoints or auth headers.

## What metrics mean

- Hit@K: 1 when at least one of top-K chunks intersects *any* relevant span.
- MRR@K: reciprocal rank of first intersecting chunk.
- Evidence-character recall@K: fraction of unique gold source characters
  covered by the union of retrieved chunks. This measures partial-span
  recovery and avoids overlap double-counting.

**These metrics are not answer faithfulness.** Both MRR and hit are permissive
about partial evidence; report the stricter recall metric next to them.
The exact dense index intentionally avoids ANN and reranker confounders.

When `retriever.kind=hash`, the results are *lexical-only engineering
smoke tests*, not semantic retrieval results. For real retrieval claims,
hold a documented actual embedding model fixed across all strategies using:

```json
{"retriever":{"kind":"http","model":"m3e","version":"m3e"}}
```

Provide the endpoint using `CHUNK_RETRIEVER_EMBEDDING_URL`. Chunking
with `semantic-http` separately uses `embedding_url` in its chunker
configuration. Do not commit private endpoint credentials in configurations.

## Upgrade roadmap

**Stage 0 — Invariants, CI, data contracts (current):** deterministic factory,
checksum manifests, gold validation, source/coverage tests and CI.

**Stage 1 — Retrieval benchmark (current):** independently labelled evidence
spans, exact retrieval baseline, query-level metrics, raw reports and plots.
A small smoke test is *not* a scientific leaderboard.

**Stage 2 — Real corpora and loaders:** curate stratified Chinese/English/mixed
collections with permission and license metadata; enrich legal, finance,
technical, HTML, Markdown and code cases; maintain train/dev/test by document
and near-duplicate detection (e.g. MinHash) across splits. Keep source maps for
PDF/HTML cleaning; an offset into transformed text is not automatically an
offset into the original file.

**Stage 3 — Algorithms:** structural Markdown/DOM/AST, sentence windows,
adaptive semantic thresholds, semantic change point and embedding window,
small-to-big retrieval, contextual retrieval and late chunking. Every method
must satisfy the same source-map and length-budget contract. Contextual
enrichment and late chunking should be represented as index-time treatments
rather than falsely disguised as only a splitter.

**Stage 4 — Reproducible experiments:** stratified paired bootstrap,
hyperparameter tuning on dev only, blind test lock, multiple seeds,
model/cache version pinning and end-to-end QA faithfulness. Distinguish
construction, retrieval, reranking and generation failures.

**Stage 5 — Publication:** quality/index-size and quality/latency Pareto
frontiers, per-query heatmaps, segment-level chunk visualizations, error
taxonomy, counterexamples, reproducibility manifest and limitations.

## Known v3 boundaries

- Markdown-heading is intentionally an ATX-heading *baseline*, not a complete
  CommonMark/HTML/LaTeX/AST parser. Do not give it structure-aware credit for
  constructs it cannot parse.
- Offline lexical hashing is not a semantic model and can collide.
- No gold-free self-supervised intrinsic metric proves downstream quality.
- The exact brute-force dense baseline is not suitable as an ANN scalability
  or production latency benchmark.
- Historical `src/html_chunk.py` contains a placeholder recursive splitter;
  migrate it only with real DOM tests and explicit source mappings.
