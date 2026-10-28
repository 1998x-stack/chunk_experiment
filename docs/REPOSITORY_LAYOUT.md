# Repository structure and migration guide

This document distinguishes **maintained production-facing interfaces**,
**experimental entry points**, **fixtures**, and **historical materials**.

## Active and legacy layout

```text
.
├── AGENTS.md                     # Mandatory instructions for coding agents
├── README.md                     # Project overview, reproducible quickstart
├── pyproject.toml                # Canonical Python package and optional extras
├── chunk_experiment/            # Maintained library
│   ├── factory.py               # ChunkerConfig + build_chunker
│   ├── models.py                # Source-aligned Chunk
│   ├── baselines.py             # Fixed and Markdown-heading baselines
│   ├── recursive.py             # Boundary-aware chunking
│   ├── semantic.py              # Embedding-dependent chunking
│   ├── embeddings.py            # Hash and HTTP embedding providers
│   ├── length.py                # Explicit length units
│   ├── datasets.py              # Checksummed manifests and gold labels
│   ├── evaluation.py            # Intrinsic chunk metrics
│   └── retrieval.py             # Exact cosine and source-span retrieval metrics
├── experiments/
│   ├── run_factory.py           # Maintained experiment matrix runner
│   ├── visualize_factory.py     # Figures from completed results.json
│   └── run_v2.py                # Earlier benchmark entry point (retained)
├── configs/
│   └── factory_smoke.json       # Versioned baseline matrix
├── benchmarks/
│   └── fixtures/                # Tiny labelled documents, manifest and gold
├── tests/                       # Maintained regression + integration tests
├── scripts/
│   └── github_fetch_checkout.py # Serial SHA-verified GitHub snapshot downloader
├── docs/                        # Research protocol, evaluation and architecture
├── archive/
│   ├── README.md                # How to interpret historical research
│   ├── reports/                 # Historical research notes
│   ├── results/                 # Legacy experiment output JSON files
│   └── figures/                 # Legacy visualization output
├── notebooks/
│   ├── README.md                # Historical notebook usage caveats
│   └── legacy/                  # Three archived notebooks
├── src/                         # Original implementations (legacy)
├── util/                        # Original helpers (legacy)
├── data/                        # Large historical text/document corpus
├── chunk_papers/                # Research-paper PDF archive
└── *.py                         # Retained legacy scripts / compatibility tools
```

Generated experiment output belongs in ignored `runs/`; it must not be
confused with archived (tracked) historical results.

## Paths changed in the documentation/layout cleanup

| Previous root path | Current path | Migration note |
| --- | --- | --- |
| `ENHANCEMENT_SUMMARY.md` | `archive/reports/ENHANCEMENT_SUMMARY.md` | Frozen historical narrative |
| `EXPERIMENT_SUMMARY.md` | `archive/reports/EXPERIMENT_SUMMARY.md` | Frozen historical narrative |
| `algorithm_comparison_results.json` | `archive/results/algorithm_comparison_results.json` | Historical output |
| `param_ablation_results.json` | `archive/results/param_ablation_results.json` | Historical output |
| `chunking_analysis.png` | `archive/figures/chunking_analysis.png` | Archived figure |
| `chunk_highlight_test.ipynb` | `notebooks/legacy/chunk_highlight_test.ipynb` | Historical notebook |
| `recursive&semantic_test.ipynb` | `notebooks/legacy/recursive_semantic_test.ipynb` | Simplified filename |
| `structure_test.ipynb` | `notebooks/legacy/structure_test.ipynb` | Historical notebook |

These are **file moves, not algorithm rewrites**. The original Git blobs remain
identical. All documented links should use new paths.

The legacy root `results_analysis.py` now locates historical JSON in
`archive/results/` when a relative filename is missing in the current
working directory. New factory result reports use the separate
`runs/<name>/results.json` schema and the maintained visualization script
`experiments/visualize_factory.py`. Do not mix these two output schemas.

## Compatibility rules

- Keep `src/` and `util/` as legacy imports until their callers are covered
  by tests and a deliberate migration plan. They are **not** re-exported as the
  maintained package.
- Keep legacy Python scripts in the repository root for now. Moving those
  scripts would change their relative imports and working-directory behavior.
- Launch v3 commands from the repository root; this is what CI validates.
- Historical notebooks may contain local absolute paths or assume an older
  working directory. Open from the repository root or adjust their
  environment manually; they are not covered by CI.
- Large tracked inputs in `data/` and `chunk_papers/` are not rewritten as
  part of file organization. A Git ignore rule does not retroactively
  untrack existing files.
- `config/` is an empty legacy configuration skeleton. Use `configs/` for
  real versioned v3 experiment matrices; avoid adding new code to `config/`.
- `setup.py` and `requirements.txt` are compatibility / historical entry
  points; prefer `pyproject.toml` optional extras for installation.
- Document any future removal of an old path with compatibility notes and
  tests before merging.

## Quality checks after structural changes

```bash
ruff check chunk_experiment tests experiments/run_factory.py \
  experiments/visualize_factory.py scripts/github_fetch_checkout.py
python -m pytest
python experiments/run_factory.py \
  --manifest benchmarks/fixtures/manifest.jsonl \
  --dataset-root benchmarks/fixtures/documents \
  --gold benchmarks/fixtures/gold.jsonl \
  --config configs/factory_smoke.json \
  --output-dir runs/smoke
```

Also check Markdown link resolution in `README.md`, `AGENTS.md`,
`docs/`, `archive/` and `notebooks/` when moving documentation.
