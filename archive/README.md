# Historical research archive

This directory preserves exploratory artifacts as **evidence of past research**,
not validated results from the maintained v3 Factory Mode pipeline.

| Directory | Content |
| --- | --- |
| [`reports/`](reports/EXPERIMENT_SUMMARY.md) | Original experiment and enhancement narratives |
| [`results/`](results/param_ablation_results.json) | Original parameter-ablation and algorithm-comparison JSON files |
| [`figures/`](figures/chunking_analysis.png) | Previous analysis figure |

Original contents were moved without changing their Git blob contents.

**Interpretation warning:** historical semantic comparisons used synthetic or
mocked embedding workflows. They support inspection of older code paths and
parameter sweeps, **not claims of real semantic retrieval superiority**.

Do not overwrite these files with new experiment outputs. Put new, generated
results under `runs/<run-name>/` (Git-ignored). The maintained factory runner
writes `results.json` and serialized Loguru `run.jsonl`.

The root `results_analysis.py` remains a legacy entry point and can locate
JSON in `archive/results/`; its historical output schema differs from
`experiments/visualize_factory.py`.

See [Repository Layout](../docs/REPOSITORY_LAYOUT.md) and
[Factory Pipeline](../docs/FACTORY_PIPELINE.md).
