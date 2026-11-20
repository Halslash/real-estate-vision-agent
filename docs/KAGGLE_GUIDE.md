# Kaggle execution and delivery guide (Member A)

All training and evaluation runs on the **Kaggle GPU**, not on the local
CachyOS laptop. The dataset is mounted in Kaggle; it is not downloaded or
searched for locally. Commands below are copy-paste cells for a Kaggle notebook.

Kaggle defaults used throughout:

- CSV: `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv`
- Images: `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/`
- Real columns: `image_id`, `price`, `sqft`, `citi`
- Batch size 64, seed 42, 2 workers, ImageNet pretrained weights required.
- **Never overwrite the original baseline checkpoint** at
  `/kaggle/working/output/models/best_model.pth`. Every new run uses its own
  `--output_dir`.

---

## A. Clone the repository

```python
!git clone <YOUR_REPOSITORY_URL> /kaggle/working/vision_agent
%cd /kaggle/working/vision_agent
```

If your clone lands elsewhere, `%cd` into the folder that contains `src/`.

---

## B. Inspect the CSV

```python
!python src/inspect_csv.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv
```

Expected ~15,474 rows, columns `image_id, street, citi, n_citi, bed, bath,
sqft, price`, no missing values. If detection reports ambiguity, pass the
actual columns explicitly with `--id_col image_id --price_col price --sqft_col
sqft --city_col citi`.

Optional smoke test (fast, small, does not touch any real output directory):

```python
!python src/train.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output/experiments/smoke \
    --epochs 2 --max_images 200 --batch_size 16 --num_workers 2 \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

---

## C. Train the improved MobileNetV2 (primary run)

```python
!python src/train.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output/improved_mobilenet \
    --epochs 20 --batch_size 64 --seed 42 --num_workers 2 \
    --architecture mobilenet_v2 --image_size 224 --resize_size 256 \
    --label_config_version v2_log_shrunk --shrinkage_k 10 \
    --augmentation improved --label_smoothing 0.05 \
    --scheduler cosine --lr 1e-4 --lr_finetune 1e-5 \
    --finetune_blocks 4 --patience 5 --device auto \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

Expected console output includes: detected columns, split sizes, class counts,
phase trainable-parameter counts, one line per epoch with loss/accuracy/macro F1,
"new best model saved" lines, and the final best epoch. Progress is printed each
epoch; the initial CSV scan only checks file existence, so it stays fast.

Artifacts: `/kaggle/working/output/improved_mobilenet/models/` and
`.../results/`.

---

## D. Evaluate the selected model (test set used once)

```python
!python src/evaluate.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output/improved_mobilenet \
    --checkpoint /kaggle/working/output/improved_mobilenet/models/best_model.pth \
    --batch_size 64 --num_workers 2 --seed 42 --device auto \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

Outputs in `.../results/`: `metrics.json`, `classification_report.json`,
`evaluation_report.md`, `confusion_matrix.png`,
`normalized_confusion_matrix.png`, `column_normalized_confusion_matrix.png`,
`prediction_distribution.png`, `learning_curves.png`.

To evaluate the **original baseline checkpoint** (do not modify its folder):

```python
!python src/evaluate.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output \
    --checkpoint /kaggle/working/output/models/best_model.pth \
    --batch_size 64 --num_workers 2 --seed 42
```

The baseline checkpoint carries no version field; the code automatically uses
the legacy v1 label statistics and MobileNetV2 preprocessing stored inside it.

---

## E. Frozen-feature linear probe (optional diagnostic)

```python
!python src/probe.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output/improved_mobilenet \
    --checkpoint /kaggle/working/output/improved_mobilenet/models/best_model.pth \
    --batch_size 64 --num_workers 2 --seed 42 --device auto \
    --feature_source pretrained \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

Result: `.../results/probe_results.json`. It fits on the saved **train** split
and scores the saved **validation** split only; the test split is never used.
With `--feature_source pretrained` the ImageNet backbone matches the
checkpoint's architecture (MobileNetV2 or EfficientNet-B0); use
`--feature_source checkpoint` to probe the trained backbone instead.

Interpretation:

- Probe macro F1 ≈ CNN validation macro F1 → likely **feature/label ceiling**;
  further head tuning is low value.
- Probe macro F1 substantially above CNN → likely **training-recipe limitation**;
  prioritise schedule/capacity, not label redesign.

---

## F. EfficientNet-B0 comparison (optional, separate output directory)

```python
!python src/train.py \
    --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv \
    --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
    --output_dir /kaggle/working/output/efficientnet_b0 \
    --epochs 20 --batch_size 64 --seed 42 --num_workers 2 \
    --architecture efficientnet_b0 --image_size 224 --resize_size 256 \
    --label_config_version v2_log_shrunk --shrinkage_k 10 \
    --augmentation improved --label_smoothing 0.05 \
    --scheduler cosine --lr 1e-4 --lr_finetune 1e-5 \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

Then evaluate it with the same pattern as D, pointing `--output_dir` and
`--checkpoint` at `.../efficientnet_b0`. Same data protocol, same split, same
label methodology, separate artifacts. Do not train both architectures in one
automatic command.

---

## G. Export the selected checkpoint

Decide the winner using **validation** macro F1 only, then run D once on its
test split. Export the whole `models/` folder so `split_indices.json` stays next
to the checkpoint:

```python
!mkdir -p /kaggle/working/export
!cp -r /kaggle/working/output/improved_mobilenet/models /kaggle/working/export/
!cp -r /kaggle/working/output/improved_mobilenet/results /kaggle/working/export/
!cp /kaggle/working/output/improved_mobilenet/models/*.json /kaggle/working/export/
```

Replace `improved_mobilenet` with the winning run's directory.

---

## H. Download or copy the checkpoint and reports

In the Kaggle notebook sidebar: **Output → Files**, hover a file under
`/kaggle/working/export/` and use the download icon. For the whole bundle:

```python
!zip -r /kaggle/working/member_a_deliverables.zip /kaggle/working/export
```

Then download `member_a_deliverables.zip` from **Output → Files**, or use
**File → Save & Export Version → Notebook Output** to archive all outputs.

Place `best_model.pth` in the repository's `models/` folder for Member B:

```python
from src.vision_agent import VisionAgent
agent = VisionAgent(checkpoint_path="models/best_model.pth")
print(agent.score("example_house.jpg"))
```

---

## Notes and cautions

- `--label_config_version v1_legacy` reproduces the baseline label math exactly;
  the improved default is `v2_log_shrunk`. Never evaluate a v1 checkpoint with
  v2 thresholds — the checkpoint carries its own version and the code enforces it.
- `--max_images` changes the split, so smoke-test results are **not** comparable
  to full runs and must not be reported as performance.
- No metric in this guide is pre-filled: all numbers must come from an actual
  Kaggle run. See `docs/FINAL_MEMBER_A_REPORT.md` for the pending-results template.
- The score is a market-derived visual proxy. Do not claim physical-condition
  accuracy or price prediction in the project report or presentation.
