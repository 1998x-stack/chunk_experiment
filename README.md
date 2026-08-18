# chunk_experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A text-**chunking algorithm experiment**: compare semantic chunking (cumulative & window
> modes) against traditional recursive chunking, with ablation over chunk size / overlap /
> similarity thresholds. —— 文本分块算法实验:对比语义分块(累积/窗口)与经典递归分块,并进行参数消融实验。

## 📊 Key Results (from `EXPERIMENT_SUMMARY.md`)

| Metric | Recursive | Semantic (Window) | Semantic (Cumulative) |
|---|---|---|---|
| Time(per doc) | 0.0001s | 0.0094–0.0477s | — |
| # Chunks | 22 | 286 | 5 |
| Avg chunk | 555.0 chars | 32.5 chars | 1860.8 chars |

- **Speed**: recursive ≫ semantic; **Sensitivity**: semantic more structure-aware; **Predictability**: recursive most stable.

## 📚 Reports (authoritative write-ups)

- **`EXPERIMENT_SUMMARY.md`** — ablation methodology, parameters & findings(本实验的权威总结)。
- **`ENHANCEMENT_SUMMARY.md`** — enhancement rebuild summary。
- `algorithm_comparison_results.json` / `param_ablation_results.json` — raw metrics。
- `chunking_analysis.png` — analysis chart。

## 🔧 Reproduce / Run

```bash
pip install -r requirements.txt   # 注意: 锁定 transformers==4.33.0 等
python ablation_experiments.py
python results_analysis.py
```

> 说明:样例脚本(`download_*.sh` 下载语料、其余 `test_*.py` / notebooks)运行依赖
> `requirements.txt`(transformers / SentencePiece / jieba 等)与语料;结果以仓库内 summary 与
> JSON 为准。

## ✅ Quality Bar

- 实验结果以仓库内 `*_SUMMARY.md` + `*.json` 为可复现来源(本 README 不虚构指标)。
- `results_analysis.py` 等脚本驱动分析。

## 📄 License

MIT — see `LICENSE`。