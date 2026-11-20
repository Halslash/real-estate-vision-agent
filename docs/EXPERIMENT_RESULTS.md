# Experiment results — Member A visual proxy CNN

**Status:** historical runs recorded; all new Kaggle experiments `PENDING`.
Fill cells only from actual run artifacts (`training_summary.json`,
`results/metrics.json`, checkpoint metadata). Never copy a value between rows,
never estimate, and never use `--max_images` smoke-test output.

> **Interpretation warning:** every target is a market-derived visual proxy
> (shrunken log price-per-sqft tertile), not verified physical condition and not
> a price. `condition_score = 0.5·P(average) + 1·P(high)` is a soft feature only.

## 1. Comparison table

| Run | Architecture | Label configuration | Image resolution | Loss function | Augmentation | Training configuration | Best val Macro F1 | Validation per-class F1 (low / avg / high) | Training duration | Checkpoint path |
|---|---|---|---|---|---|---|---:|---|---|---|
| Original baseline (historical) | MobileNetV2 | `v1_legacy`, K n/a | 224 | CE, auto weights, ls 0.05 | legacy | 20 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `/kaggle/working/output/models/best_model.pth` |
| Improved (incumbent, historical) | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto weights, ls 0.05 | improved | bs 64, seed 42, best epoch 15 | **0.4521** (val acc 0.4589) | PENDING | PENDING | `/kaggle/working/output/improved_mobilenet/models/best_model.pth` |
| `e0_reference` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto, ls 0.05 | improved | 30 ep, bs 64, seed 42, cosine | PENDING | PENDING | PENDING | `/kaggle/working/output/experiments/e0_reference/models/best_model.pth` |
| `e1_no_weighting` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, none, ls 0 | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e1_no_weighting/models/best_model.pth` |
| `e2_balanced_ce` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, balanced | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e2_balanced_ce/models/best_model.pth` |
| `e3_effective_number` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, eff.-num β=.9999 | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e3_effective_number/models/best_model.pth` |
| `e4_focal` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | Focal γ=2, no weights | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e4_focal/models/best_model.pth` |
| `e5_ordinal_smooth` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | Ordinal ε=.1, no weights | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e5_ordinal_smooth/models/best_model.pth` |
| `e6_efficientnet_b0` | EfficientNet-B0 | `v2_log_shrunk`, K=10 | 224 | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e6_efficientnet_b0/models/best_model.pth` |
| `e7_higher_resolution` | MobileNetV2 | `v2_log_shrunk`, K=10 | 256 (crop from 288) | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e7_higher_resolution/models/best_model.pth` |
| `e8_legacy_augmentation` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto | legacy | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e8_legacy_augmentation/models/best_model.pth` |
| `e9_shrinkage_k1` | MobileNetV2 | `v2_log_shrunk`, **K=1** | 224 | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | PENDING | `.../e9_shrinkage_k1/models/best_model.pth` |
| `e10_longer_finetune` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto | improved | 45 ep, 8 blocks, pat 8, seed 42 | PENDING | PENDING | PENDING | `.../e10_longer_finetune/models/best_model.pth` |

`...` = `/kaggle/working/output/experiments`. Peak GPU memory (where
available) is in each run's `training_summary.json`.

## 2. Historical test metrics — preserved, do not re-evaluate

Improved incumbent, **untouched test split** (label config `v2_log_shrunk`):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4384 | 0.4323 | 0.4362 | 0.4309 | 0.4327 |

Per-class test F1: low **0.4684**, average **0.3293**, high **0.4948**.

Original baseline (different label config `v1_legacy`): test accuracy **0.4065**,
macro F1 **0.4057**. This is **not** a controlled comparison with the incumbent.

Frozen ImageNet logistic probe (train→val, test untouched): val accuracy
**0.4580**, val macro F1 **0.4561**. Diagnostic only — not a CNN result.

## 3. Frozen-feature probe results

| Run | Feature source | Val accuracy | Val Macro F1 | Compared CNN val Macro F1 |
|---|---|---:|---:|---:|
| Historical improved | pretrained ImageNet | 0.4580 | 0.4561 | 0.4521 |
| `e0_reference` | pretrained ImageNet | PENDING | PENDING | PENDING |

Probe and fine-tuned CNN results must stay in separate tables and storylines.

## 4. Failed / aborted experiments

Record every run that did not complete, with its cause, here. None recorded yet.
A smoke test (`--max_images 200`) is **not** an experiment and belongs nowhere
in this document.

## 5. Selection and one-time test evaluation

1. Rank by validation macro F1; break ties with per-class F1 and support.
2. Confirm the winner's checkpoint metadata matches its table row.
3. Run the test evaluation **once**:
   `!python src/evaluate.py --output_dir <winner_output_dir>`
4. Paste test metrics into `docs/FINAL_MEMBER_A_REPORT.md`.
5. If no new run beats 0.4521 validation macro F1 convincingly, keep the
   incumbent checkpoint and record that decision here.
