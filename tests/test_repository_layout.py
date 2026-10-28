"""Guard documented repository paths and archived legacy artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

DOCS = (
    "README.md",
    "AGENTS.md",
    "docs/REPOSITORY_LAYOUT.md",
    "docs/ARCHITECTURE.md",
    "docs/EVALUATION.md",
    "docs/FACTORY_PIPELINE.md",
    "docs/REVIEW.md",
    "archive/README.md",
    "notebooks/README.md",
)
ARCHIVED = (
    "archive/reports/ENHANCEMENT_SUMMARY.md",
    "archive/reports/EXPERIMENT_SUMMARY.md",
    "archive/results/algorithm_comparison_results.json",
    "archive/results/param_ablation_results.json",
    "archive/figures/chunking_analysis.png",
    "notebooks/legacy/chunk_highlight_test.ipynb",
    "notebooks/legacy/recursive_semantic_test.ipynb",
    "notebooks/legacy/structure_test.ipynb",
)
REMOVED_FROM_ROOT = (
    "ENHANCEMENT_SUMMARY.md",
    "EXPERIMENT_SUMMARY.md",
    "algorithm_comparison_results.json",
    "param_ablation_results.json",
    "chunking_analysis.png",
    "chunk_highlight_test.ipynb",
    "recursive&semantic_test.ipynb",
    "structure_test.ipynb",
)


def test_historical_archives_preserved_and_root_uncluttered():
    for relative in ARCHIVED:
        assert (ROOT / relative).is_file(), f"missing archived artifact: {relative}"
    for relative in REMOVED_FROM_ROOT:
        assert not (ROOT / relative).exists(), f"obsolete artifact remains: {relative}"


def test_documentation_internal_markdown_links_resolve():
    # This covers authored Markdown links, not code-fenced shell examples.
    markdown_link = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in DOCS:
        document = ROOT / path
        assert document.is_file()
        content = document.read_text(encoding="utf-8")
        for match in markdown_link.finditer(content):
            target = match.group(1).strip().split(" ", 1)[0]
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = unquote(urlsplit(target).path)
            if not relative:
                continue
            resolved = (document.parent / relative).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), (
                f"external relative reference in {path}: {target}"
            )
            assert resolved.exists(), f"broken link in {path}: {target}"
