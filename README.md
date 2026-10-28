# Chunk Experiment

**Factory-driven, reproducible experiments for text chunking and retrieval-augmented generation (RAG).**

[![CI](https://github.com/1998x-stack/chunk_experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/chunk_experiment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 面向中英文与结构化文档的文本切块实验工作台。通过统一的 Factory 配置、源文本对齐、可核验的数据集、检索级评测与 Loguru 日志，将“切块策略是否更好”转化为可重复、可审计的实验问题。

**Status:** v3 research workbench. The end-to-end fixture is a *smoke test*, not an independently validated semantic leaderboard. The legacy research code remains available for reproducibility, but the maintained public API lives under `chunk_experiment/`.

## 1. Why this project

Chunk count and average length alone cannot determine whether a chunking method improves RAG. This repository separates the concerns that are often conflated:

1. **Chunk construction:** exact source offsets, complete coverage, explicit length budgets and stable segmentation.
2. **Retrieval quality:** source-span evidence labels independent of chunk IDs, with identical retriever and Top-K across methods.
3. **Experiment traceability:** dataset/config hashes, raw per-query rankings, timing, environments and structured logs.

The central rule is **one controlled change at a time**. A better segmentation metric is a diagnostic observation—not evidence that end-user answers are more accurate.

## 2. Quick start

Run these commands from the **repository root**. Python 3.10+ is required.

```bash
python -m pip install -e ".[dev,viz]"

# End-to-end evaluation on the included tiny, labelled fixture:
python experiments/run_factory.py \
  --manifest benchmarks/fixtures/manifest.jsonl \
  --dataset-root benchmarks/fixtures/documents \
  --gold benchmarks/fixtures/gold.jsonl \
  --config configs/factory_smoke.json \
  --output-dir runs/smoke

# Figures: retrieval quality vs index size; per-query recall heatmap
python experiments/visualize_factory.py runs/smoke/results.json

# Core tests and code quality
python -m pytest
ruff check chunk_experiment tests experiments/run_factory.py \
  experiments/visualize_factory.py scripts/github_fetch_checkout.py
```

The runner creates:

| Artifact | Purpose |
| --- | --- |
| `runs/smoke/results.json` | Configuration, provenance, per-document chunk diagnostics, raw query rankings and aggregate retrieval metrics |
| `runs/smoke/run.jsonl` | Loguru serialized event log with per-document and per-query context |
| `runs/smoke/figures/*.png` | Quality/index-size plot and evidence-recall heatmap |

`runs/` is ignored by Git. No cloud embedding API is required for this **lexical-hash smoke test**.

### Single-document CLI

```bash
chunk-experiment benchmarks/fixtures/documents/fruit.txt \
  --algorithm recursive --chunk-size 24 --overlap 4
```

This CLI returns source-aligned chunks and intrinsic quality diagnostics as JSON. It is distinct from the full retrieval benchmark.

## 3. System architecture

```text
     checksummed corpus manifest        source-span query labels
                  |                               |
             strict loader                    gold validator
                  |                               |
          Document + metadata                     |
                  |                               |
           ChunkerConfig                          |
                  |                               |
            Chunker Factory                       |
           /     |       \\                       |
       fixed  recursive  heading / semantic      |
           \\     |       /                      |
          source-aligned Chunk[]                  |
                  |                               |
       coverage / size / duplication             |
                  |                               |
           fixed embedding backend               |
                  |                               |
          exact cosine retrieval ----------------+
                  |
       Hit@K / MRR@K / evidence recall
                  |
          JSON + Loguru + plots
```

Core contracts:

- A chunk must satisfy `chunk.text == document.text[chunk.start:chunk.end]`.
- Character length and approximate token length are different units; all compared strategies must use the **same** unit.
- Every eligible document must reach full source coverage; a missing character is an evaluation failure.
- Gold evidence is recorded as source-character spans, **not** chunk IDs.
- Retrieval embedding backend, corpus split, query set and Top-K must remain fixed across an ablation.

## 4. Available strategies

| Strategy / Factory key | Implementation | Interpretation |
| --- | --- | --- |
| Fixed window / `fixed` | `baselines.py` | Deterministic budget/overlap control |
| Recursive / `recursive` | `recursive.py` | Hierarchical separator-aware baseline |
| Markdown heading / `markdown-heading` | `baselines.py` | ATX-heading aware baseline; **not** a complete CommonMark parser |
| Hash-based split / `semantic-hash` | `semantic.py` | Deterministic **lexical** hashing for CI and pipeline checks |
| HTTP semantic / `semantic-http` | `semantic.py`, `embeddings.py` | Real remote embedding support; results depend on the identified provider and model |
| Injected provider / `semantic-provider` | `factory.py` | Programmatic dependency injection for custom embedding implementations |

Factory example:

```python
from chunk_experiment.factory import ChunkerConfig, build_chunker
from chunk_experiment.evaluation import evaluate_chunks

source = "第一段内容。第二段内容。"
config = ChunkerConfig(
    algorithm="recursive",
    chunk_size=12,
    chunk_overlap=2,
    length_unit="characters",
)
chunks = build_chunker(config).split(source)
metrics = evaluate_chunks(source, chunks, max_chunk_size=config.chunk_size)
assert metrics.coverage_ratio == 1.0
```

**Important:** a hash-based baseline is not a semantic embedding model. The repository does not yet claim that one strategy is better on an independently annotated, held-out production corpus.

## 5. Datasets, gold labels and evaluation

The factory runner accepts a UTF-8 JSONL document manifest and query labels. Full schemas and acceptance rules are in [Factory Pipeline](docs/FACTORY_PIPELINE.md).

**Corpus manifest example** (one object per line; `path` is relative to `--dataset-root`):

```json
{"document_id":"manual-01","path":"manual.txt","split":"test","language":"zh","domain":"technical","sha256":"<actual SHA-256 of file bytes>"}
```

**Gold query example** (offsets are half-open `[start,end)` into the *decoded original text*):

```json
{"case_id":"manual-qa-01","query":"文档介绍了什么？","relevant_spans":[{"document_id":"manual-01","start":0,"end":12}]}
```

The supplied manifest/gold fixtures in `benchmarks/fixtures/` are executable examples. For scientific studies, build separate train/dev/test splits with vetted labels, provenance and data-leakage audits.

| Metric | Meaning | Caveat |
| --- | --- | --- |
| Source coverage | Fraction of original characters covered by emitted chunks | Integrity gate, not retrieval quality |
| Duplication ratio | Duplicated source span per source character | Depends on overlap configuration |
| Size compliance | Fraction within the configured length budget | Requires a documented length metric |
| Hit@K | Any retrieved chunk intersects a gold evidence span | Partial evidence may count |
| MRR@K | Reciprocal rank of the first intersecting chunk | Not answer correctness |
| Evidence-character Recall@K | Fraction of gold characters covered by the union of retrieved spans | Robust to overlap double-counting, still not answer faithfulness |

Keep indexing and query latency, total chunk count and per-query results alongside quality scores. Compare on a **held-out dataset** before interpreting improvements.

## 6. Experiment configuration and logging

The reproducible matrix is defined in [`configs/factory_smoke.json`](configs/factory_smoke.json). The runner records the configuration, Git commit (when available), dataset/gold/config SHA-256, environment, seed and query-level ranking.

Loguru emits structured JSONL events, including chunking progress, query metrics and run identity. Do not log original private documents, tokens, cookies, credentials or service URLs.

For a genuine semantic retrieval run, configure a documented HTTP retrieval model and set `CHUNK_RETRIEVER_EMBEDDING_URL` in the process environment. A separate chunker-side HTTP embedding configuration is required for `semantic-http`. Freeze the model identity and preprocessing across comparisons.

Optional extras:

```bash
pip install -e ".[http]"       # HTTP embedding client
pip install -e ".[viz]"        # Matplotlib reports
pip install -e ".[legacy]"     # Historical notebooks / research dependencies
```

## 7. Repository map

```text
chunk_experiment/        Maintained chunking, embedding, data and evaluation APIs
experiments/             Factory runner, visualizer and v2 benchmark runner
configs/                 Version-controlled experiment configurations
benchmarks/fixtures/     Small, labelled CI/smoke dataset
tests/                   Maintained automated regression and integration tests
docs/                    Architecture, methodology, review and repository guide
scripts/                 Verified GitHub per-blob working-tree fetcher
notebooks/legacy/        Archived research notebooks (not production APIs)
archive/reports/         Historical experiment narratives
archive/results/         Historical unverified comparison JSON
archive/figures/         Historical figures
src/, util/              Preserved legacy implementations
data/, chunk_papers/     Large historical source corpus and research PDFs
```

Some early root-level `*_test.py`, `ablation_experiments.py` and `results_analysis.py` are preserved as **legacy entry points**. Do not treat them as the maintained package API. See [Repository Layout](docs/REPOSITORY_LAYOUT.md) and [AGENTS.md](AGENTS.md) before making structural changes.

## 8. Clone or fetch locally

For a complete checkout with Git history:

```bash
git clone https://github.com/1998x-stack/chunk_experiment.git
cd chunk_experiment
```

The repository contains a substantial historical data corpus. For code-first, lower-download checkout, use Git partial clone + sparse checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/1998x-stack/chunk_experiment.git
cd chunk_experiment
git sparse-checkout set chunk_experiment experiments tests configs \
  benchmarks docs scripts src util
```

Alternatively, [`scripts/github_fetch_checkout.py`](scripts/github_fetch_checkout.py) downloads files **sequentially** from a pinned GitHub ref, verifies each Git blob SHA and resumes safely. It produces a **working-tree snapshot, not a Git clone**:

```bash
python scripts/github_fetch_checkout.py \
  --repo 1998x-stack/chunk_experiment \
  --ref main --target ../chunk-experiment-snapshot \
  --exclude-prefix data --exclude-prefix chunk_papers
```

The public GitHub API has rate limits; `GITHUB_TOKEN` is optional but may be used for authenticated access. Never place the token in a committed file or shell transcript.

## 9. Documentation, legacy research and contribution policy

- [Architecture](docs/ARCHITECTURE.md) — source-aligned core contracts.
- [Factory Pipeline](docs/FACTORY_PIPELINE.md) — experiment protocol, datasets, metrics and research roadmap.
- [Evaluation](docs/EVALUATION.md) — intrinsic, retrieval and end-to-end RAG metric design.
- [Code review](docs/REVIEW.md) — earlier correctness findings and limitations.
- [Repository Layout](docs/REPOSITORY_LAYOUT.md) — new paths and backwards compatibility.
- [AGENTS.md](AGENTS.md) — instructions for Codex and other code-writing agents.
- [Historical reports](archive/README.md) — original analyses, figures and notebook caveats.

Contributions should be scoped to one measurable hypothesis, with a factory config, regression tests and raw per-query outputs for changes affecting retrieval. Keep legacy comparisons labelled as historical; old mocked-embedding results cannot support semantic superiority claims.

## License

MIT — see [LICENSE](LICENSE).
