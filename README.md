# Member A — CNN Visual Proxy Agent

**Projet 10: Évaluation Immobilière Multi-Modale (PropTech)**

- **Member A (this repo):** visual analysis of house exterior photos → normalized visual proxy score + class label.
- **Member B (separate):** structural Random Forest + final Linear Regression → final price estimate.

> **Scientific honesty note (read first):** this CNN produces a **market-derived visual proxy**, not an objectively verified physical-condition or aesthetic grade, and **not a price estimate**. The dataset has no human condition labels; targets are tertiles of price per square foot relative to comparable properties. A higher score does **not** prove better physical condition. Member B owns price estimation.

---

## 1. What this repository contains

A transfer-learning image classifier (MobileNetV2 by default, EfficientNet-B0 optional) that looks at a property's **exterior photograph** and returns a `condition_score` in `[0, 1]`. It is intentionally only Member A's deliverable; no Random Forest, Linear Regression, or final price prediction is implemented here.

The score is one soft feature Member B can append to their structural matrix. All interpretation limits are documented in `docs/MODEL_CARD.md`.

---

## 2. Quick links

| Document | Purpose |
|---|---|
| `docs/KAGGLE_GUIDE.md` | Copy-paste Kaggle commands A–H (train, evaluate, probe, export, download). |
| `docs/MEMBER_B_INTEGRATION.md` | The `VisionAgent` API contract and integration examples. |
| `docs/EXPERIMENT_PLAN.md` | 11 controlled experiments and the mandatory selection rules. |
| `docs/EXPERIMENT_RESULTS.md` | Comparison table (architecture, labels, loss, F1, duration, checkpoint). |
| `docs/MODEL_CARD.md` | Intended use, prohibited interpretations, limitations. |
| `docs/FINAL_MEMBER_A_REPORT.md` | 13-section final report: method, controls, results, demos, limitations. |
| `docs/FINAL_ACADEMIC_READINESS_AUDIT.md` | Verified vs tested vs implemented-not-run vs remaining limitations. |
| `docs/IMPLEMENTATION_COMPLETION_REPORT.md` | Static implementation status and verification checklist. |
| `docs/CNN_PERFORMANCE_AUDIT.md` | Evidence-backed audit: what was wrong and what changed. |

New tooling: `src/run_experiment.py` + `configs/experiments.json` (isolated
experiments), `src/losses.py` (focal/ordinal/effective-number),
`src/data_audit.py` (leakage diagnostics), `src/make_folds.py` +
`src/oof_contract.py` + `src/oof_predict.py` (honest OOF scores for Member B),
`src/demonstrate.py` (real-image demo), and `tests/`.
Run locally with `python -m unittest discover -s tests -v`; train only on Kaggle.

---

## 3. Architecture and pipeline

```
Image (JPEG)
  -> RGB conversion + Resize(256) + CenterCrop(224) + ImageNet normalization
  -> MobileNetV2 (ImageNet pretrained)   [or EfficientNet-B0]
  -> Dropout(0.2) -> Linear(in_features -> 3)
  -> Softmax -> P(low), P(average), P(high)
  -> condition_score = 0.5 * P(average) + 1.0 * P(high)
```

**Two-phase training** (`src/train.py`):

1. **Feature extraction** — backbone frozen, only the new head trained with AdamW (`--lr`, default `1e-4`). Always completes.
2. **Fine-tuning** — last `--finetune_blocks` (default 4) unfrozen at a lower LR (`--lr_finetune`, `1e-5`). Validation macro F1 drives early stopping (`--patience 5`) **in this phase only**.

`--epochs` is the **total** across both phases (default 20 → ~10 + 10). A per-phase cosine LR schedule (`--scheduler cosine`, or `none`) and label smoothing (`--label_smoothing`, default `0.05`) are configurable. Weighted cross-entropy is used only when class imbalance exceeds `--imbalance_ratio` (default 1.5). Pretrained ImageNet weights are mandatory: a download failure raises a clear error, never silent random initialization.

Seeds (Python/NumPy/PyTorch/CUDA) are fixed and cuDNN is deterministic. CPU fallback is supported via `--device cpu` (or automatic with `auto`).

---

## 4. Dataset and label construction

On Kaggle this project expects:

- CSV: `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv`
- Images: `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/`

~15,474 rows with columns `image_id, street, citi, n_citi, bed, bath, sqft, price`; images named `<image_id>.jpg`. Columns are detected case-insensitively; pass `--id_col image_id --price_col price --sqft_col sqft --city_col citi` to be explicit (ambiguous matches raise instead of guessing).

Lazy validation: the CSV scan only checks file existence; pixels are decoded lazily in `HouseImageDataset.__getitem__` with a bounded corrupt-image retry that keeps the label of the image actually loaded. Grayscale/RGBA/palette images are converted to RGB.

**Two versioned proxy-label methodologies** (the checkpoint records which one it uses):

- `v1_legacy` — the original baseline: clip raw `price/sqft` to train-only 1st/99th percentiles, divide by the raw city median (global fallback), tertile the training ratios.
- `v2_log_shrunk` (default) — `log(price/sqft)` clipped to train-only percentiles; city median shrunk toward the global median with `k = 10` pseudo-counts (`m̂_c = (n_c·m_c + k·m_global)/(n_c + k)`); proxy value is `clipped_log_pps − m̂_c`; tertiles from observed distinct training values.

In both cases **all statistics are fitted on the training split only** and stored in the checkpoint; validation/test never influence them, and a checkpoint is never evaluated with another version's thresholds. The 70/15/15 stratified split (seed 42) uses temporary price-per-sqft quantiles only to balance itself — those bins are not labels.

---

## 5. Augmentation policy

- **Training (`improved`, default):** Resize(256) + CenterCrop(224) — geometry aligned with evaluation; horizontal flip; restrained colour jitter; ImageNet normalization. **No rotation** and no aggressive random crop, which could rotate houses or remove rooflines/facades.
- **Legacy (`--augmentation legacy`):** the original RandomResizedCrop + flip + 10° rotation + stronger jitter, for controlled comparison.
- **Validation/test/inference:** deterministic Resize(256) + CenterCrop(224) + ImageNet normalization, built from the checkpoint's saved preprocessing config.

---

## 6. Evaluation methodology

`src/evaluate.py` reloads the checkpoint, reads its sibling `split_indices.json`, reuses the saved label statistics, column names, preprocessing and architecture, rebuilds the exact model, and evaluates **only the untouched test split** — no split is regenerated and the test set is never used for selection. It reports:

- Accuracy; macro precision/recall/F1; weighted F1; per-class metrics.
- True vs predicted class distributions; raw, row-normalized (recall) and column-normalized confusion matrices.
- Honest reference baselines computed on this split (majority class, seeded uniform random).

Artifacts written under `<output_dir>/`:

```
models/
  best_model.pth             # self-contained checkpoint: weights + arch + preprocessing + stats + history
  model_config.json          # argparse config + derived training settings
  preprocessing_stats.json   # clip bounds, medians, thresholds, shrinkage, label version
  class_thresholds.json      # thresholds and label names
  city_medians.json          # raw/shrunk city medians and counts
  split_indices.json         # exact train/val/test image IDs
  preprocessing_config.json  # image size / mean / std / augmentation / arch
  training_summary.json      # best epoch, best F1, early-stopping decision
results/
  training_history.csv       # per-epoch losses, accuracy, macro F1, LR
  learning_curves.png
  metrics.json
  classification_report.json
  evaluation_report.md
  confusion_matrix.png
  normalized_confusion_matrix.png
  column_normalized_confusion_matrix.png
  prediction_distribution.png
```

**Optional frozen probe** (`src/probe.py`): logistic regression on frozen features, fitted on the saved train split and scored on validation only; separate `probe_results.json`, no weights or test data touched. See `docs/KAGGLE_GUIDE.md` section E.

---

## 7. VisionAgent API (for Member B)

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(
    checkpoint_path="models/best_model.pth"   # legacy .load(path) still works
)
result = agent.score("example_house.jpg")
print(result)
```

```python
{
    "condition_score": 0.73,
    "label": "high",
    "probabilities": {"low": 0.10, "average": 0.24, "high": 0.66},
    "confidence": 0.66,
    "model": "mobilenet_v2",
    "score_type": "market_derived_visual_proxy",
}
```

Batch scoring: `results = agent.score_batch(paths, batch_size=64)` (input order preserved). Inference needs **only** the checkpoint — no CSV, no dataset, no pretrained download. CPU/CUDA is auto-selected. Invalid paths or corrupt images raise clear exceptions; no fabricated fallback scores are ever returned. Full contract and integration example: `docs/MEMBER_B_INTEGRATION.md`.

---

## 8. Kaggle execution (summary)

All training/evaluation runs on **Kaggle GPU**, not locally. See `docs/KAGGLE_GUIDE.md` for complete copy-paste commands. In short:

```python
# B: inspect CSV
!python src/inspect_csv.py

# C: train improved MobileNetV2 (20 epochs; separate output dir)
!python src/train.py \
    --csv_path <CSV> --images_dir <IMAGES> \
    --output_dir /kaggle/working/output/improved_mobilenet \
    --epochs 20 --batch_size 64 --seed 42 --num_workers 2 \
    --architecture mobilenet_v2 --image_size 224 --resize_size 256 \
    --label_config_version v2_log_shrunk --shrinkage_k 10 \
    --augmentation improved --label_smoothing 0.05 --scheduler cosine \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi

# D: evaluate the selected checkpoint once on the test split
!python src/evaluate.py --output_dir /kaggle/working/output/improved_mobilenet
```

**Never overwrite the original baseline checkpoint** at `/kaggle/working/output/models/best_model.pth`; every new run uses its own `--output_dir`.

---

## 9. Command-line arguments

Core (both `train.py` and `evaluate.py`): `--images_dir`, `--csv_path`, `--output_dir`, `--epochs`, `--max_images`, `--batch_size`, `--lr`, `--id_col`, `--price_col`, `--sqft_col`, `--city_col`.

Training essentials: `--lr_finetune` (`1e-5`), `--num_workers` (2), `--patience` (5), `--finetune_blocks` (4), `--weight_decay` (`1e-4`), `--imbalance_ratio` (1.5), `--seed` (42), `--architecture` (`mobilenet_v2` | `efficientnet_b0`), `--image_size` (224), `--resize_size` (256), `--label_smoothing` (0.05), `--scheduler` (`cosine` | `none`), `--label_config_version` (`v2_log_shrunk` | `v1_legacy`), `--shrinkage_k` (10), `--augmentation` (`improved` | `legacy`), `--device` (`auto` | `cpu` | `cuda`).

Evaluation/probe extras: `--checkpoint`, `--num_workers`, `--seed`, `--device`. Probe also has `--feature_source` (`pretrained` | `checkpoint`). Run `python src/train.py --help` for the authoritative list.

---

## 10. Repository layout

```
vision_agent/                       # repository root
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # columns, proxy labels, splits, transforms, Dataset
│   ├── model.py                    # architectures + self-contained checkpoint I/O
│   ├── plots.py                    # shared headless matplotlib helpers
│   ├── train.py                    # two-phase training, scheduler, artifacts
│   ├── evaluate.py                 # test evaluation and reports
│   ├── probe.py                    # optional frozen-feature linear probe
│   ├── vision_agent.py             # Member B's inference interface
│   ├── inspect_csv.py              # standalone CSV inspection
│   ├── losses.py                   # CE / focal / ordinal / class weighting
│   ├── data_audit.py               # read-only leakage diagnostics
│   ├── make_folds.py               # 5 OOF folds from canonical train split
│   ├── oof_contract.py             # pure OOF isolation + CSV column invariants
│   ├── oof_predict.py              # honest OOF scores for Member B
│   ├── run_experiment.py           # config-driven isolated experiments
│   └── demonstrate.py              # real-image qualitative demonstration
├── docs/                           # guides, model card, reports, audit
├── requirements.txt
├── README.md
└── .gitignore                      # generated weights/datasets stay untracked
```

Generated weights and datasets are intentionally ignored by Git (see `.gitignore`).
