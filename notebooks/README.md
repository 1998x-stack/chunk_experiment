# Historical research notebooks

The notebooks in [`legacy/`](legacy/) are preserved to keep the early
RAG-chunking experiments inspectable:

- `chunk_highlight_test.ipynb`
- `recursive_semantic_test.ipynb` (formerly `recursive&semantic_test.ipynb`)
- `structure_test.ipynb`

These notebooks are **not** the supported v3 experiment runner. Their source
cells may assume a repository-root current working directory, historical
data locations, additional optional libraries or machine-specific paths.
Moving the notebook files did not rewrite their code cells.

For reproducible comparisons, use the command-line factory runner and
versioned JSON configurations instead. The Notebook archive may be opened
manually for historical inspection; notebook execution is **not** part of CI.

See [README](../README.md),
[Repository Layout](../docs/REPOSITORY_LAYOUT.md) and
[Historical Results](../archive/README.md).
