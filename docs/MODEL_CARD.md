# Model card — Member A visual proxy CNN

## Model details

- **Owner:** Member A, Projet 10 — Évaluation Immobilière Multi-Modale (PropTech).
- **Model type:** Transfer-learning image classifier with a 3-class ordered head
  and a continuous proxy score derived from softmax probabilities.
- **Backbones:** MobileNetV2 (default) and EfficientNet-B0 (optional comparison),
  ImageNet-pretrained weights, 224×224 RGB default input.
- **Architecture head:** `Dropout(0.2) → Linear(1280, 3)` with ImageNet
  normalization. The same head convention is used for both backbones.
- **Output:** `condition_score = 0.5 * P(average) + 1.0 * P(high)` in `[0, 1]`,
  plus argmax label, class probabilities and confidence.
- **Public API:** `src.vision_agent.VisionAgent` (see
  `docs/MEMBER_B_INTEGRATION.md`).
- **Versioned configuration:** checkpoints carry `arch`,
  `label_config_version` (`v1_legacy` or `v2_log_shrunk`),
  `preprocessing_version`, the full preprocessing block, class mapping, label
  statistics, training config, best epoch, validation metrics and PyTorch version.

## Intended use

- Produce a **market-derived visual proxy score** from an exterior photograph of
  a Southern California property, for use as one soft feature in Member B's
  structural price-estimation pipeline.
- Compare label/preprocessing/training recipes under a fixed split on Kaggle GPU.
- Diagnose whether image data carries signal about relative price per square foot
  in this specific market.

## Prohibited interpretations (explicit)

- ❌ Not a verified physical-condition classifier.
- ❌ Not an objective aesthetic-quality classifier.
- ❌ Not a calibrated probability that a house is "good" or "well maintained".
- ❌ Not a property-price estimator; it does not output a price and Member B owns
  final price estimation.
- ❌ A high score does **not** prove the house is in better physical condition.
- ❌ Not portable to another housing market without retraining; the score is
  relative to this dataset's price-per-square-foot distribution.
- ❌ Not suitable for individual property valuation decisions, lending, insurance
  or any high-stakes determination.

The dataset contains **no human-annotated condition labels**. Price per square
foot is confounded by location, neighbourhood, school district, lot,
renovations, demand and market timing — factors largely invisible in an exterior
photo. The learned signal is expected to be weak.

## Data

- Source: Kaggle `nikhilkushwaha2529/real-estate-with-image`, SoCal subset.
- CSV: `.../archive-2/socal2.csv`; images: `.../archive-2/socal_pics/<image_id>.jpg`.
- ~15,474 rows; columns `image_id, street, citi, n_citi, bed, bath, sqft, price`.
- Rows with invalid/non-positive price or sqft, duplicate IDs or missing files
  are skipped; image pixels are decoded lazily with bounded corrupt-image retry.
- Split: deterministic 70/15/15 train/validation/test by `image_id`, seed 42;
  temporary price-per-sqft quantiles are used only to balance the split.

## Label generation

Two versioned methodologies exist; the checkpoint records which one it uses.

**`v1_legacy` (original baseline):** clip raw `price/sqft` to train-only 1st/99th
percentiles; divide by the raw city median (global median fallback); tertile the
training ratios. City medians have no minimum group size or shrinkage.

**`v2_log_shrunk` (improved default):** `log(price/sqft)` clipped to train-only
1st/99th percentiles; city median `m̂_c = (n_c·m_c + k·m_global)/(n_c + k)` with
default `k = 10` (missing/unseen cities use the global median); proxy value is
`clipped_log_pps − m̂_c`; tertile thresholds are taken from observed distinct
training values. All constants (clip bounds, global median, raw and shrunken
city medians, city counts, shrinkage k, thresholds, label version) are fitted on
the **training split only** and stored in the checkpoint.

The three classes are ordered `low < average < high`. Thresholds are never
refitted on validation or test, and a checkpoint is never evaluated with another
version's thresholds.

## Training

- Two phases: frozen backbone + head (AdamW `--lr`, default 1e-4), then last
  `--finetune_blocks` (default 4) unfrozen at lower LR (`--lr_finetune`, 1e-5).
- Default `--epochs 20` (≈10 + 10); per-phase cosine LR decay (`--scheduler
  cosine`, or `none`); label smoothing default 0.05; weighted cross-entropy only
  when class imbalance exceeds `--imbalance_ratio 1.5`.
- Model selection by validation macro F1, with early stopping available in
  Phase 2 (`--patience 5`). Seeds fixed for Python, NumPy, PyTorch, CUDA; cuDNN
  deterministic mode enabled.
- Training augmentation v2: resize + center crop (geometry aligned with
  evaluation), horizontal flip, restrained colour jitter; **no rotation** and no
  aggressive random crop, which could remove rooflines/facades or tilt houses.

## Performance

**Measured baseline (user-provided training log, validation split):** best epoch
9, validation accuracy 39.60%, validation macro F1 0.3961, 15,474 samples.

**Held-out test metrics: PENDING** — no test evaluation has been run by Member A's
static implementation work. Run `docs/KAGGLE_GUIDE.md` section D and record the
output in `docs/FINAL_MEMBER_A_REPORT.md`. Do not fill these numbers by hand.

Approximate references for a balanced 3-class task: uniform random ≈33.3%
accuracy, majority-class ≈33.3% accuracy but ≈0.167 macro F1. Exact baselines for
the specific test split are computed by `src/evaluate.py` from the true labels.

## Limitations and known biases

- Weak, noisy proxy labels; most price-per-square-foot variance is not visible
  in an exterior photograph.
- Small-city median noise in v1; v2 shrinkage mitigates but cannot remove it.
- Non-grouped split may place near-duplicate images from the same street in more
  than one split; this would make held-out numbers optimistic, not pessimistic.
- 224px resolution may miss subtle material/condition cues.
- Label smoothing softens probability outputs; confidence is not calibration.
- Dataset is Southern California specific and may encode neighbourhood price
  conventions rather than visual quality.
- No uncertainty estimate for an individual image's score is provided.

## Reproducibility

- Exact commands in `docs/KAGGLE_GUIDE.md`; experiment IDs and decision rules in
  `docs/EXPERIMENT_PLAN.md`.
- Checkpoint plus sibling JSON files (`split_indices.json`,
  `preprocessing_stats.json`, `city_medians.json`, `model_config.json`) fully
  determine evaluation and inference; no local training files are needed.
- Checkpoint stores the PyTorch version when available. Use a compatible
  torch/torchvision pair when reconstructing the model.

## Ethical considerations

Do not use this model to make or influence high-stakes decisions about
individual properties or their owners. Report it only as a weak, market-relative
visual signal with the interpretation warnings above attached.
