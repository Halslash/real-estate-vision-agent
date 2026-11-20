# Model card — Member A visual proxy CNN

## Model details

- **Owner:** Member A, Projet 10 — Évaluation Immobilière Multi-Modale (PropTech).
- **Model type:** ImageNet-pretrained transfer-learning classifier with a 3-class
  ordered head; continuous proxy score derived from softmax probabilities.
- **Backbones:** MobileNetV2 (default) and EfficientNet-B0 (comparison only),
  224×224 RGB default input.
- **Head:** `Dropout(0.2) → Linear(1280 or 1280, 3)`, ImageNet normalization.
- **Output:** `condition_score = 0.5·P(average) + 1.0·P(high)` in `[0,1]`, plus
  argmax `label`, class `probabilities`, `confidence`, `model`, `score_type`.
- **Public API:** `src.vision_agent.VisionAgent` — `VisionAgent(device="cuda")`,
  `.load(path)`, `.score(path)`, `.score_batch(paths, batch_size=64)`.
- **Checkpoint metadata:** architecture, class map, preprocessing, label stats,
  full training config, best epoch, validation metrics, history, PyTorch version.

## Intended use

Produce a **normalized, market-derived visual proxy** feature to append to
Member B's structural feature set. Accepts exterior property photographs.

**Out of scope / prohibited:** physical-condition grading, aesthetic scoring,
price estimation, using the score as a calibrated probability, or treating it as
ground truth. Member B owns final price prediction.

## Training data and label construction

- Real-estate listings with `image_id`, `price`, `sqft`, optional `citi`.
- 70/15/15 stratified split, seed 42; duplicate image ids removed.
- `v2_log_shrunk` proxy: clip `log(price/sqft)` to train p1/p99, subtract the
  city log-median shrunk with `K=10`, then split the training values into
  low/average/high tertiles using observed distinct values.
- All clipping, medians, shrinkage and thresholds are fitted on **train only**
  and stored in the checkpoint. `v1_legacy` is preserved for old checkpoints.

## Evaluation

Historical incumbent (untouched test split, `v2_log_shrunk`):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4384 | 0.4323 | 0.4362 | 0.4309 | 0.4327 |

Per-class test F1: low 0.4684, average 0.3293, high 0.4948.
Validation macro F1 0.4521 (accuracy 0.4589) at best epoch 15.
Computed majority/random baselines are saved with every evaluation report.

A frozen ImageNet logistic probe scored 0.4561 validation macro F1 — level with
the CNN, suggesting the feature/label signal is the binding constraint. Probe
results are diagnostics, not CNN performance.

New experiments are listed in `docs/EXPERIMENT_PLAN.md` and recorded in
`docs/EXPERIMENT_RESULTS.md`; none have been run in this work.

## Ethical considerations and limitations

- **No human labels.** Targets are inferred from market prices and confounded
  by location, neighbourhood demand, market timing and unobserved property
  characteristics. A higher score does **not** prove better physical condition.
- **Weak middle class.** The average tertile is a heterogeneous residual bin
  (test F1 0.3293). Do not interpret it as a "medium condition" verdict.
- **Possible leakage.** Duplicates and same-street listings have not yet been
  audited; run `src/data_audit.py` on Kaggle. Held-out metrics may be optimistic.
- **In-sample feature risk for Member B.** Use the OOF scoring path
  (`src/make_folds.py` + `src/oof_predict.py`) for training rows.
- **No calibration or uncertainty.** The soft score is a heuristic mixture, not
  a probability of any real-world event.
- **Bias.** Trained on one regional market; generalisation to other regions is
  untested and should not be assumed.

## Reproduction

See `docs/KAGGLE_GUIDE.md`. Training occurs only on Kaggle GPU; no local
training, no GitHub push, no modification of historical checkpoints.
