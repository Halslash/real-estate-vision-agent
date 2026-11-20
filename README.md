# Member A — CNN Vision Agent

**Projet 10: Évaluation Immobilière Multi-Modale (PropTech)**
Member A: CNN-based visual analysis of house exterior photos.
Member B: structural features (Random Forest) and final price estimation (Linear Regression), integrating Member A's visual output.

> **Scientific honesty note:** this CNN estimates a **market-derived visual-quality proxy**, not an objectively verified physical or aesthetic condition. It does **not** predict the final property price. Member B owns price estimation.

---

## 1. Project description

This repository contains only Member A's deliverable: a transfer-learning image classifier that looks at the **exterior photograph** of a Southern California property and returns a normalized `condition_score` in `[0, 1]`.

Because the dataset has **no human-annotated condition labels**, labels are constructed from market data (price per square foot relative to comparable properties). The CNN learns to recognize visual patterns correlated with that market-derived signal. The score is a continuous proxy feature that Member B can append to their structural feature matrix.

---

## 2. Responsibility split

| Member | Responsibility | Output |
|--------|----------------|--------|
| **A (this repo)** | Image preprocessing, proxy labels, MobileNetV2 training, visual scoring, evaluation, `VisionAgent` API | `condition_score` + `label` per image |
| **B (separate)** | Structural Random Forest, final Linear Regression, combining structural + visual features | Final estimated price |

The boundary is the `VisionAgent.score()` method. Member B never needs to touch training code.

---

## 3. Architecture

```
Image (JPEG)
   -> RGB conversion + resize/center crop (224x224) + ImageNet normalization
   -> MobileNetV2 (ImageNet pretrained weights)
   -> Dropout(0.2) -> Linear(1280 -> 3)
   -> Softmax -> P(low), P(average), P(high)
   -> condition_score = 0.5 * P(average) + 1.0 * P(high)
```

**Training is two-phase:**

1. **Feature extraction:** the entire MobileNetV2 backbone is frozen; only the new classification head is trained with AdamW (`--lr`, default `1e-4`) and cross-entropy loss. This phase always completes.
2. **Fine-tuning:** the last `--finetune_blocks` (default 4) backbone blocks are unfrozen and trained with a lower learning rate (`--lr_finetune`, default `1e-5`). Early stopping monitors validation macro F1 **in this phase only**, so Phase 1 can never be cut short and Phase 2 is never skipped on a normal run.

If the class counts are imbalanced (max/min class weight above `--imbalance_ratio`, default `1.5`), a weighted cross-entropy loss is used.

`--epochs` is the **total** number of epochs across both phases, split roughly in half (e.g. 10 -> 5 + 5). With `--epochs 1`, only Phase 1 runs, which is sufficient for a smoke test.

Unlike a regression head, the 3-class formulation gives calibrated-style probabilities that are easy for Member B to consume and inspect.

---

## 4. Dataset

On Kaggle, this project expects:

- Images:
  `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/`
- CSV:
  `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv`

Images are named `0.jpg`, `1.jpg`, ... (~15,000 files). Each CSV row is matched to `<image_id>.jpg`.

Column names are detected case-insensitively using aliases (e.g. `image_id`/`id`/`image`, `price`/`listing_price`, `sqft`/`square_feet`/`living_area`, `city`/`citi`/`n_citi`). City is optional. Override detection with:

```
--id_col --price_col --sqft_col --city_col
```

If a required column is missing, the script prints all available columns and raises a clear error. If several columns plausibly match a role (for example `price` and `price_per_sqft`, or `living_area` and `lot_area`), detection also raises an ambiguity error instead of guessing — pass the exact column with an override.

**Lazy image validation:** the initial CSV scan only checks that `<image_id>.jpg` exists via `os.path.isfile()`; it does **not** open or decode images. Rows with missing/non-positive price or square footage, duplicate image IDs, or missing files are skipped during this fast scan (~15,000 images). Actual pixel decoding happens lazily in `HouseImageDataset.__getitem__()` while training/evaluation batches are built. If a file is corrupt or becomes unreadable, a bounded retry skips it and uses the next readable image while keeping the label of the image actually loaded. Grayscale, RGBA and palette images are converted to RGB at decode time.

**Always run `inspect_csv.py` first** if detection fails or reports ambiguity, then pass the real column names explicitly.

---

## 5. Proxy-label construction and limitations

For each property:

1. `price_per_sqft = price / sqft`.
2. Clip `price_per_sqft` to the **1st and 99th percentiles fitted on the training split only**.
3. For each city, compute the **median** clipped price per square foot on the training split. Unseen/missing cities use the global training median.
4. `ratio = clipped_price_per_sqft / city_median`.
5. Split training ratios into tertiles -> `0 = low`, `1 = average`, `2 = high`. The same thresholds are applied unchanged to validation and test.

The 70/15/15 stratified split uses temporary quantiles of raw price-per-sqft **only to balance the split**; those temporary bins are not the labels and are never used to fit city statistics or thresholds. The split is seeded (42), has no overlapping image IDs, and raises a clear error if any stratum is too small.

**Limitations — important for the presentation:**

- The labels are a **market proxy**. A high price per square foot may reflect location, lot size, renovations, or school districts — not merely a better-looking exterior.
- The model can learn city-specific price conventions rather than pure visual quality.
- Tertile classes are relative to this dataset, not absolute quality grades.
- The score is **not calibrated** as a physical-condition measurement and should be treated as a soft feature, not ground truth.
- No test data is used for fitting statistics, thresholds, early stopping or model selection.

All fitted values (clip bounds, city medians, global median, class thresholds, detected columns) are saved in the checkpoint and in `models/*.json`, so training, evaluation and inference stay consistent without the CSV.

---

## 6. Training methodology

- GPU is used when `torch.cuda.is_available()`, otherwise CPU.
- Batch size 64, 2 DataLoader workers, seed 42.
- Train augmentation: random resized crop, horizontal flip, small rotation, color jitter, ImageNet normalization.
- Validation/test: resize to 256, center crop 224, ImageNet normalization (exact same transform as inference).
- Best checkpoint selected by **validation macro F1**; early stopping (`--patience`, default 5) applies to Phase 2 only.
- Pretrained ImageNet weights are mandatory: a download failure raises a clear error; the code never silently falls back to random initialization.
- Small datasets fail loudly: if a split is empty, or clipping/tertiles cannot produce three distinct classes in training, training raises a clear error instead of silently producing an invalid split. The 200-image smoke test is checked the same way.

Artifacts written to `<output_dir>` (default `/kaggle/working/output`):

```
models/
  best_model.pth             # self-contained checkpoint (weights + config + stats)
  model_config.json          # argparse config and derived training settings
  preprocessing_stats.json   # clip bounds, medians, thresholds
  class_thresholds.json      # class thresholds and label names
  city_medians.json          # per-city + global medians
  split_indices.json         # exact train/val/test image IDs
  preprocessing_config.json  # image size / mean / std
results/
  training_history.csv       # per-epoch losses, accuracy, macro F1, learning rate
```

---

## 7. Evaluation methodology

`src/evaluate.py` reloads the checkpoint from `<output_dir>/models/best_model.pth` (override with `--checkpoint`), reads `split_indices.json` from the checkpoint's own folder, reuses the saved proxy-label statistics and saved column names, rebuilds **the exact same model**, and evaluates only on the untouched test set — no split is regenerated:

- Accuracy
- Macro precision, macro recall, macro F1
- Weighted F1
- Per-class precision / recall / F1 / support
- Confusion matrix and row-normalized confusion matrix

Outputs (in `<output_dir>/results/`):

- `confusion_matrix.png`
- `normalized_confusion_matrix.png`
- `classification_report.json`
- `metrics.json`

The test set is never used for early stopping or model selection.

---

## 8. Output files summary

| File | Purpose |
|------|---------|
| `models/best_model.pth` | Trained weights + architecture info + class mapping + preprocessing config + proxy-label statistics |
| `models/*.json` | Human-readable configuration and statistics |
| `models/split_indices.json` | Reproduces the exact evaluation split |
| `results/training_history.csv` | Loss/accuracy/F1 curves data |
| `results/metrics.json` | Headline test metrics |
| `results/classification_report.json` | Per-class metrics and matrices |
| `results/*confusion_matrix.png` | Visual confusion matrices |

---

## 9. VisionAgent API (for Member B)

```python
from src.vision_agent import VisionAgent

agent = VisionAgent()
agent.load("models/best_model.pth")

result = agent.score("example_house.jpg")

print(result)
```

Example return value (illustrative):

```python
{
    "condition_score": 0.73,
    "label": "high",
    "probabilities": {"low": 0.10, "average": 0.24, "high": 0.66},
    "confidence": 0.66,
}
```

- `condition_score = 0.5 * P(average) + 1.0 * P(high)`, always in `[0, 1]`.
- `label` is the argmax class name: `low`, `average` or `high`.
- `confidence` is the maximum class probability.
- Probabilities sum to 1 (up to floating-point rounding).
- Inference runs in inference mode with gradients disabled, on CUDA when available.
- Invalid paths and corrupted images raise clear exceptions.
- Requires **only** the checkpoint: no CSV, no dataset, no MobileNetV2 knowledge for the caller.

With the default Kaggle output directory, the checkpoint lives at:
`/kaggle/working/output/models/best_model.pth`.
Download it into the repository's `models/` folder (or pass that path directly to `agent.load`) for the snippet above to work verbatim.

**How Member B integrates it:** score every property image once, then append `condition_score` as one extra column to the structural feature table used by the Random Forest / Linear Regression pipeline, e.g.:

```python
import os
from src.vision_agent import VisionAgent

agent = VisionAgent().load("models/best_model.pth")
df["condition_score"] = [
    agent.score(os.path.join(images_dir, f"{image_id}.jpg"))["condition_score"]
    for image_id in df["image_id"]
]
# Member B: train Random Forest on structural + condition_score,
# then feed its output into the final Linear Regression.
```

This repository intentionally does **not** implement Member B's models.

---

## 10. Kaggle execution instructions

Kaggle notebooks provide the libraries in `requirements.txt`; no local installation or local dataset is needed.

### Step 1 — Clone repository

```python
!git clone YOUR_REPOSITORY_URL
%cd vision_agent
```

(If your repository URL clones to a different folder name, `%cd` into that folder instead — the commands below assume the repository root contains `src/`.)

### Step 2 — Inspect CSV

```python
!python src/inspect_csv.py
```

If the printed column names do not match the aliases, run with overrides, e.g.:

```python
!python src/inspect_csv.py --csv_path /kaggle/input/.../socal2.csv
```

### Step 3 — Smoke test

```python
!python src/train.py --epochs 1 --max_images 200 --batch_size 16
```

### Step 4 — Full training

```python
!python src/train.py --epochs 10 --batch_size 64
```

If column auto-detection fails:

```python
!python src/train.py --epochs 10 --batch_size 64 \
    --id_col image_id --price_col price --sqft_col sqft --city_col city
```

### Step 5 — Evaluate

```python
!python src/evaluate.py
```

`--max_images` is normally unnecessary here: evaluation reuses the saved test IDs, it does not rebuild the split. Use `--output_dir` for a non-default output location and `--checkpoint` to point at a moved checkpoint (its folder must contain `split_indices.json`).

**Downloading results from Kaggle:** in the notebook right sidebar open **Output → Files**, navigate to `output/models` and `output/results`, hover a file and use the download icon. Alternatively use **File → Save & Export Version → Notebook Output** to archive the whole output directory, then download the archive. Put `best_model.pth` into this repository's `models/` folder (or keep the full `/kaggle/working/output/models/best_model.pth` path) before using `VisionAgent`.

### Step 6 — Use VisionAgent

```python
from src.vision_agent import VisionAgent

agent = VisionAgent()
agent.load("models/best_model.pth")

result = agent.score("example_house.jpg")

print(result)
```

---

## 11. Command-line arguments

Both `train.py` and `evaluate.py` accept:

| Argument | Default | Notes |
|----------|---------|-------|
| `--images_dir` | Kaggle `socal_pics/` path | Directory of `<id>.jpg` files |
| `--csv_path` | Kaggle `socal2.csv` path | Metadata CSV |
| `--output_dir` | `/kaggle/working/output` | Where models/results are written |
| `--epochs` | `10` | Total epochs (evaluate: informational) |
| `--max_images` | `None` | Cap for smoke tests; evaluate reuses saved IDs |
| `--batch_size` | `64` | |
| `--lr` | `1e-4` | Phase 1 learning rate (evaluate: informational) |
| `--id_col` | `None` | Explicit CSV column override |
| `--price_col` | `None` | Explicit CSV column override |
| `--sqft_col` | `None` | Explicit CSV column override |
| `--city_col` | `None` | Optional city column override |

Training-only extras: `--lr_finetune` (default `1e-5`, Phase 2), `--num_workers` (2), `--patience` (5, Phase 2 only), `--finetune_blocks` (4), `--weight_decay` (`1e-4`), `--imbalance_ratio` (`1.5`), `--seed` (42).
Evaluation-only extras: `--checkpoint`, `--num_workers`, `--seed`.

---

## 12. Files

```
vision_agent/                # repository root (this directory)
├── src/
│   ├── __init__.py
│   ├── model.py             # MobileNetV2 + checkpoint utilities
│   ├── dataset.py           # columns, proxy labels, splits, transforms, Dataset
│   ├── train.py             # two-phase training
│   ├── evaluate.py          # test evaluation and reports
│   ├── vision_agent.py      # Member B's inference interface
│   └── inspect_csv.py       # standalone CSV inspection
├── results/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```
