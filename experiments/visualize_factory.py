"""Publication-oriented figures for factory benchmark results.

Requires: pip install -e '.[viz]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize chunk factory results")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload["results"]
    if not rows:
        raise ValueError("no experiment results to visualize")
    output = args.output_dir or args.results.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)

    # A Pareto view: source-aligned retrieval quality vs indexing cost.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for result in rows:
        name = result["config"]["algorithm"]
        size = result["config"]["chunk_size"]
        score = result["summary"]["mean_evidence_char_recall_at_k"]
        chunks = result["summary"]["chunk_count"]
        ax.scatter(chunks, score, s=70)
        ax.annotate(f"{name} / {size}", (chunks, score), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="Indexed chunks (proxy for index size)",
           ylabel=f"Mean evidence-character recall @{payload['top_k']}",
           title="Retrieval quality vs indexing footprint", ylim=(-0.03, 1.05))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_index_size.png", dpi=180)
    plt.close(fig)

    # Per-query heatmap exposes regressions hidden by the aggregate mean.
    case_ids = [q["case_id"] for q in rows[0]["queries"]]
    labels = [f"{r['config']['algorithm']} / {r['config']['chunk_size']}" for r in rows]
    import numpy as np

    heatmap = np.array([
        [next(q["metrics"]["evidence_char_recall_at_k"]
              for q in r["queries"] if q["case_id"] == case_id)
         for case_id in case_ids]
        for r in rows
    ])
    fig, ax = plt.subplots(figsize=(max(8, len(case_ids) * 0.55),
                                    max(3, len(rows) * 0.65)))
    image = ax.imshow(heatmap, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(case_ids)), labels=case_ids, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set(title="Evidence recall by query and chunker", xlabel="Gold query",
           ylabel="Chunker / budget")
    fig.colorbar(image, ax=ax, label="Evidence-char recall")
    fig.tight_layout()
    fig.savefig(output / "query_recall_heatmap.png", dpi=180)
    plt.close(fig)

    print(f"Wrote 2 figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
