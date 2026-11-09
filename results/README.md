# Results artifacts

The files here are transcribed from the camera-ready paper. They are the
source of truth for README figures and are not overwritten by model runs.
`python scripts/make_figures.py` regenerates the CSVs/JSON from the same values
and rebuilds `assets/`.

| File | Contents |
|---|---|
| `zero_shot.csv` | Zero-shot baseline models, evaluated rows, macro F1, accuracy. |
| `few_shot.csv` | Four-shot baselines, including partial-row runs. |
| `fine_tuned.csv` | Submitted Gemma LoRA-64 adapter vs. SmolVLM LoRA-32. |
| `per_class_f1.csv` | Per-class F1 for every reported validation run. |
| `validation_metrics.json` | Submitted adapter validation metrics and error notes. |
| `test_metrics.json` | Official test macro F1, class counts, per-class metrics. |

Notes:

- Partial-row baseline scores (397, 445, 610 rows) are not comparable to
  full-split scores.
- `same_figure` and `same_paper` test F1 values are derived from the reported
  precision and recall: `F1 = 2PR / (P + R)`.
- `unrelated_papers` test precision/recall were not reported separately, so its
  F1 is left blank instead of being estimated.
- The paper's test analysis used the organizers' solution file after the
  leaderboard submission. Test labels were not used in training, model
  selection, or prompt design.
