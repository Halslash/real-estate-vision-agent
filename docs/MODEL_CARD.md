# Model card — Member A visual proxy CNN

## Model details

- **Owner:** Member A, Projet 10 — Évaluation Immobilière Multi-Modale (PropTech).
- **Model type:** ImageNet-pretrained transfer-learning classifier with a 3-class
  ordered head; continuous proxy score derived from softmax probabilities.
- **Backbones:** MobileNetV2 (default, used by the final model) and EfficientNet-B0
  (comparison only), 224×224 RGB crop from a 256 resize.
- **Head:** `Dropout(0.2) → Linear(1280 or 1280, 3)`, ImageNet normalization.
- **Output:** `condition_score = 0.5·P(average) + 1.0·P(high)` in `[0,1]`, plus
  argmax `label`, full class `probabilities`, `confidence`, `model`, `score_type`.
- **Public API:** `src.vision_agent.VisionAgent` — `VisionAgent(device="cuda")`,
  `.load(path)`, `.score(path)`, `.score_batch(paths, batch_size=64)`.
- **Checkpoint metadata:** architecture, class map, preprocessing, label stats,
  full training config, best epoch, validation metrics, history, PyTorch version.
- **Final checkpoint:** `/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth`
  (MobileNetV2, `e0_reference` configuration, property-disjoint split).

## Intended use

Produce a **normalized, market-derived visual proxy** feature to append to
Member B's structural feature set. Accepts exterior property photographs.

**Out of scope / prohibited:** physical-condition grading, aesthetic scoring,
direct price estimation, using the score as a calibrated probability, or treating
it as ground truth. Member B owns final price prediction.

## Training data and label construction

- Real-estate listings with `image_id`, `price`, `sqft`, optional `citi`.
- Final run: property-disjoint split (normalized full street address + exact
  image SHA-256), seed 42 → train 10,829 / val 2,337 / test 2,308. No usable
  address or exact-hash group crosses splits; near-duplicate dHash grouping was
  off. Historical runs used a 70/15/15 random stratified split (seed 42) with
  duplicate image ids removed. **The two protocols are not directly comparable.**
- `v2_log_shrunk` proxy: clip `log(price/sqft)` to train p1/p99, subtract the
  city log-median shrunk with `K=10`, then split the training values into
  low/average/high tertiles using observed distinct values.
- All clipping, medians, shrinkage and thresholds are fitted on **train only**
  and stored in the checkpoint. `v1_legacy` is preserved for old checkpoints.

## Evaluation

### Final model — `e0_reference_group_split` (property-disjoint split)

Best epoch **27** (30 epochs run, no early stop), validation Macro F1
**0.4314855**. One-time evaluation of the untouched test split (**2,308**
samples):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4281 | 0.4277 | 0.4296 | 0.4240 | 0.4233 |

Per-class test F1: low **0.4582**, average **0.3539**, high **0.4600**.
Average-class recall is **0.3070** (weak). Majority/random baselines, per-class
precision/recall/support, raw and normalized confusion matrices, and class
distributions are in `results/metrics.json`, `results/classification_report.json`
and `results/evaluation_report.md`.

The test split was evaluated **once** and sealed in `results/evaluation_gate.json`.
It must never be reused for model selection or tuning.

### Historical random-split incumbent — preserved, not comparable

Previous incumbent (`v2_log_shrunk`, random 70/15/15 split, seed 42):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4384 | 0.4323 | 0.4362 | 0.4309 | 0.4327 |

Per-class test F1: low 0.4684, average 0.3293, high 0.4948. Validation macro F1
0.4521 (accuracy 0.4589) at best epoch 15.

These numbers predate the property-disjoint split and are **not directly
comparable** with the final model: the historical test set shares addresses and
byte-identical images with training, while the final test set does not. The
lower final F1 reflects the stricter protocol, not a proven regression.

A frozen ImageNet logistic probe scored 0.4561 validation macro F1 — level with
the historical CNN, suggesting the feature/label signal is the binding
constraint. Probe results are diagnostics, not CNN performance. No new probe was
run for the final model.

## Ethical considerations and limitations

- **No human labels.** Targets are inferred from market prices and confounded
  by location, neighbourhood demand, market timing and unobserved property
  characteristics. A higher score does **not** prove better physical condition.
- **Weak middle class.** The average tertile is a heterogeneous residual bin
  (final test F1 0.3539, recall 0.3070). Do not interpret it as a "medium
  condition" verdict.
- **Leakage addressed for the final run.** The final split is property-disjoint
  (address + exact SHA-256); zero usable addresses or exact-hash groups cross
  splits. Near-duplicate perceptual matching was disabled, so similar photos at
  different addresses remain untested.
- **Not a direct price estimator.** The CNN maps one image to a market-derived
  proxy class; Member B owns price prediction.
- **No calibration or uncertainty.** `confidence` is the maximum class
  probability, **not** a calibrated confidence; the soft score is a heuristic
  mixture, not a probability of any real-world event.
- **Sealed test set.** Evaluated once. Never reuse it for model selection or
  tuning.
- **In-sample feature risk for Member B.** Use out-of-fold scores from fold
  checkpoints that never trained on the scored images; see
  `docs/MEMBER_A_TECHNICAL_HANDOFF.md`.
- **Bias.** Trained on one regional market; generalisation to other regions is
  untested and should not be assumed.

## Reproduction

See `docs/ACADEMIC_REPORT.md` for the examiner-facing summary,
`docs/MEMBER_A_KAGGLE_READINESS.md` for the exact pinned preflight, training and
one-time evaluation cells, and `docs/MEMBER_A_TECHNICAL_HANDOFF.md` for the
interface contract. Training occurs only on Kaggle GPU; no local training, no
GitHub push, no modification of historical checkpoints or sealed metrics.
