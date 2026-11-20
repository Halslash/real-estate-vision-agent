# Final Member A report — CNN visual proxy

**Status: static implementation complete; all Kaggle/runtime results PENDING.**

This report is the single place to record measured results. Every value marked
**PENDING** must be filled from an actual Kaggle GPU run (see
`docs/KAGGLE_GUIDE.md`) before submission. Do **not** copy numbers between
experiments, do not estimate, and do not use smoke-test (`--max_images 200`)
numbers as performance.

> **Interpretation warning:** all targets are market-derived visual proxies
> (tertiles of relative price per square foot), not human-annotated physical
> condition or aesthetic labels, and not prices. See `docs/MODEL_CARD.md`.

---

## 1. Objective and scope

Member A builds and evaluates a CNN that scores property exterior photos with a
normalized visual proxy, exposed to Member B through `VisionAgent`. Scope is
strictly Member A: no structural Random Forest, no Linear Regression, no final
price prediction.

---

## 2. Method summary

- **Backbone:** MobileNetV2 ImageNet V1 (default); EfficientNet-B0 optional.
- **Head:** `Dropout(0.2) → Linear(in_features, 3)`; classes `low/average/high`.
- **Label v2 (default):** `log(price/sqft)` clipped to train-only 1st/99th
  percentiles, minus shrunken city log median (`k=10`), tertiled on train only.
- **Split:** deterministic 70/15/15 by `image_id`, seed 42, no overlap.
- **Training:** two phases (frozen head → last 4 blocks), AdamW, cosine LR,
  label smoothing 0.05, validation-macro-F1 early stopping in Phase 2.
- **Augmentation:** improved (aligned geometry, no rotation/crop) vs legacy.
- **Selection:** validation macro F1 only; test split evaluated once at the end.

---

## 3. Known baseline (user-provided training log)

These are the only measured numbers available so far and come from the existing
baseline checkpoint's **validation** run — not from this implementation's test
evaluation.

| Item | Value |
|---|---|
| Samples | 15,474 |
| Epochs trained | 10 |
| Best epoch | 9 |
| Validation accuracy | 39.60% |
| Validation macro F1 | 0.3961 |

Baseline **test** metrics (via `docs/KAGGLE_GUIDE.md` section D, baseline folder):
**PENDING**.

---

## 4. Experiment results

Fill one row per executed run from `docs/EXPERIMENT_PLAN.md`. Use validation
metrics for every selection decision; add the test row only for the final winner.

| ID | Config / changed flags | Output directory | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| E0 | Baseline evaluate/diagnose | `output/` | — | — | PENDING | PENDING | Test here is descriptive only. |
| E1 | v2 labels | `output/experiments/E1_labels` | PENDING | PENDING | — | — | |
| E2 | improved augmentation | `output/experiments/E2_augmentation` | PENDING | PENDING | — | — | |
| E3 | 20 epochs + cosine (primary) | `output/improved_mobilenet` | PENDING | PENDING | PENDING | PENDING | Selected winner: ______ |
| E4 | label smoothing 0.1 | `output/experiments/E4_smoothing` | PENDING | PENDING | — | — | |
| E5 | frozen probe | chosen run's `probe_results.json` | PENDING | PENDING | — | — | Probe val macro F1: ______ |
| E6 | EfficientNet-B0 | `output/efficientnet_b0` | PENDING | PENDING | — | — | |
| E7 | shrinkage k=3 / k=30 | `output/experiments/E7_*` | PENDING | PENDING | — | — | |
| E8 | 299px resolution | `output/experiments/E8_299` | PENDING | PENDING | — | — | |

Multi-seed runs are intentionally omitted: `--seed` also fixes the split, so
changing it would destroy the fixed-split comparability of E0–E8.

Reference baselines are computed by `src/evaluate.py` for the exact test split
(majority class and seeded uniform random ≈33.3% accuracy); record them from
`metrics.json` when E0 and the winner are evaluated: **PENDING**.

---

## 5. Diagnostics to attach

- Learning curves (`results/learning_curves.png`) — train/val loss and accuracy/F1.
- Predicted vs true class distribution (`results/prediction_distribution.png`) — check for prior collapse.
- Confusion matrices (`results/*confusion_matrix.png`) — identify class confusion.
- Frozen probe result (`results/probe_results.json`) — separate feature/label ceiling from training-recipe issues.
- All figures/metrics are written automatically by `src/evaluate.py`; paste or reference them here after the runs.

---

## 6. Findings (fill after runs)

1. **Did v2 labels help?** Compare E1 validation macro F1 with the baseline. PENDING.
2. **Did augmentation alignment help / what was the train–val gap?** PENDING.
3. **Was early stopping active, and at which epoch?** PENDING.
4. **Probe verdict:** feature/label ceiling or training-recipe limitation? PENDING.
5. **Architecture/resolution/shrinkage decision and justification.** PENDING.

---

## 7. Integration hand-off to Member B

- Selected checkpoint: **PENDING** (path + hash once exported).
- `condition_score = 0.5 * P(average) + 1.0 * P(high)`, in `[0, 1]`.
- API: `VisionAgent(checkpoint_path=...)`, `.score(path)`, `.score_batch(paths)`.
- Fields: `condition_score`, `label`, `probabilities`, `confidence`, `model`, `score_type`.
- Missing/invalid images raise clear errors; Member B decides imputation; no fabricated scores.
- See `docs/MEMBER_B_INTEGRATION.md` for the full contract.

---

## 8. Limitations and honest claims

- Weak, noisy market proxy labels; most price-per-sqft variance is invisible in an exterior photo.
- No human condition ground truth; a high score does **not** mean better physical condition.
- Possible near-duplicate leakage from a non-grouped split would make held-out numbers optimistic.
- Confidence is not calibration, especially with label smoothing.
- Southern California specific; not portable to another market without retraining.
- No calibrated per-image uncertainty is provided.

Do not claim the model is "trained and improved", give performance numbers, or
claim physical-condition/price accuracy in the presentation until the PENDING
cells above are filled from real Kaggle results.
