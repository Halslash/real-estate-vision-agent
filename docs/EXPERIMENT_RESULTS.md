# Experiment results — Member A visual proxy CNN

**Status: FINAL.** `e0_reference_group_split` was trained and evaluated **once**
at pinned commit `bc39d102d2e9cde83144966d6e6c910926109429`. Sealed artifacts are
in `/kaggle/working/output/experiments/e0_reference_group_split/`. No other `e*`
experiment was run. Historical random-split results are preserved in §3 and are
**not comparable** with the final property-disjoint run. Numbers come only from
`training_summary.json`, `results/metrics.json`, `results/evaluation_gate.json`
and checkpoint metadata — never estimated, never copied between rows, never
from a `--max_images` smoke test.

> **Interpretation warning:** every target is a market-derived visual proxy
> (shrunken log price-per-sqft tertile), not verified physical condition and not
> a price. `condition_score = 0.5·P(average) + 1·P(high)` is a soft feature only.

## 1. Comparison table

| Run | Architecture | Label configuration | Image resolution | Loss function | Augmentation | Training configuration | Best val Macro F1 | Training duration | Checkpoint path |
|---|---|---|---|---|---|---|---:|---|---|
| **Final `e0_reference_group_split`** | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 crop / 256 resize | CE, auto β=.9999, ls 0.05 | improved | 30 ep, bs 64, seed 42, cosine, best epoch 27, **property-disjoint split** | **0.4314855** | 1446.1 s | `/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth` |
| Original baseline (historical) | MobileNetV2 | `v1_legacy`, K n/a | 224 | CE, auto weights, ls 0.05 | legacy | 20 ep, bs 64, seed 42, random split | PENDING | PENDING | `/kaggle/working/output/models/best_model.pth` |
| Improved (incumbent, historical) | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto weights, ls 0.05 | improved | seed 42, best epoch 15, random split | 0.4521 (val acc 0.4589) | PENDING | `/kaggle/working/output/improved_mobilenet/models/best_model.pth` |
| `e0_reference` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto, ls 0.05 | improved | 30 ep, bs 64, seed 42, cosine | PENDING | PENDING | `.../e0_reference/models/best_model.pth` |
| `e1_no_weighting` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, none, ls 0 | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e1_no_weighting/models/best_model.pth` |
| `e2_balanced_ce` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, balanced | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e2_balanced_ce/models/best_model.pth` |
| `e3_effective_number` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, eff.-num β=.9999 | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e3_effective_number/models/best_model.pth` |
| `e4_focal` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | Focal γ=2, no weights | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e4_focal/models/best_model.pth` |
| `e5_ordinal_smooth` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | Ordinal ε=.1, no weights | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e5_ordinal_smooth/models/best_model.pth` |
| `e6_efficientnet_b0` | EfficientNet-B0 | `v2_log_shrunk`, K=10 | 224 | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e6_efficientnet_b0/models/best_model.pth` |
| `e7_higher_resolution` | MobileNetV2 | `v2_log_shrunk`, K=10 | 256 (crop from 288) | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e7_higher_resolution/models/best_model.pth` |
| `e8_legacy_augmentation` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto | legacy | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e8_legacy_augmentation/models/best_model.pth` |
| `e9_shrinkage_k1` | MobileNetV2 | `v2_log_shrunk`, **K=1** | 224 | CE, auto | improved | 30 ep, bs 64, seed 42 | PENDING | PENDING | `.../e9_shrinkage_k1/models/best_model.pth` |
| `e10_longer_finetune` | MobileNetV2 | `v2_log_shrunk`, K=10 | 224 | CE, auto | improved | 45 ep, 8 blocks, pat 8, seed 42 | PENDING | PENDING | `.../e10_longer_finetune/models/best_model.pth` |

`...` = `/kaggle/working/output/experiments`. `PENDING` means **not run** — only
the final `e0_reference_group_split` row has live Kaggle results. Peak GPU memory
(717.5 MB) and resolved device (CUDA) are in its `training_summary.json`.

## 2. Final model — `e0_reference_group_split` (property-disjoint split)

MobileNetV2, `v2_log_shrunk`, K=10, 224 px crop / 256 resize, CE + auto class
weights β=0.9999, label smoothing 0.05, improved augmentation, seed 42, 30
epochs, cosine schedule, early-stopping patience 5 (not triggered). Best epoch
**27**, validation Macro F1 **0.4314855**. Duration 1446.1 s, peak GPU memory
717.5 MB, device CUDA.

One-time test evaluation on **2,308** untouched samples:

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4281 | 0.4277 | 0.4296 | 0.4240 | 0.4233 |

Per-class test F1: low **0.4582**, average **0.3539**, high **0.4600**.
Average-class recall **0.3070** (weak).

Artifacts: `models/best_model.pth`, `models/training_summary.json`,
`models/split_indices.json`, `results/metrics.json`, `results/evaluation_gate.json`,
`results/classification_report.json`, `results/evaluation_report.md` and the
confusion-matrix figures. Split and preflight SHA-256 values are bound in
`training_gate.json` and re-verified in the evaluation gate. Test data was used
for nothing during selection.

**The test split was evaluated once and is sealed. Never reuse it for selection
or tuning.**

**Scientific limitations.** Labels are market-derived visual proxy classes
(shrunken log price-per-sqft tertiles), **not** verified physical-condition or
aesthetic labels, and the model is **not a direct price estimator**. The
`average` class is weak (recall **0.3070**). `confidence` is the maximum class
probability, **not** a calibrated confidence; `condition_score` is a heuristic
soft feature, not a probability of a real-world event. See `docs/MODEL_CARD.md`.

## 3. Historical random-split results — preserved, not comparable

These predate the property-disjoint split. The historical test set shares street
addresses and byte-identical images with training; the final test set does not.
They are therefore **not a controlled comparison** — the lower final F1 reflects
the stricter protocol, not a proven regression.

Improved incumbent, historical **untouched test split** (`v2_log_shrunk`):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4384 | 0.4323 | 0.4362 | 0.4309 | 0.4327 |

Per-class test F1: low **0.4684**, average **0.3293**, high **0.4948**.

Original baseline (different label config `v1_legacy`): test accuracy **0.4065**,
macro F1 **0.4057**. This is **not** a controlled comparison with the incumbent.

Frozen ImageNet logistic probe (train→val, test untouched): val accuracy
**0.4580**, val macro F1 **0.4561**. Diagnostic only — not a CNN result.

## 4. Frozen-feature probe results

| Run | Feature source | Val accuracy | Val Macro F1 | Compared CNN val Macro F1 |
|---|---|---:|---:|---:|
| Historical improved | pretrained ImageNet | 0.4580 | 0.4561 | 0.4521 |
| Final `e0_reference_group_split` | pretrained ImageNet | PENDING | PENDING | 0.4314855 |

Probe and fine-tuned CNN results must stay in separate tables and storylines.
No probe was run for the final property-disjoint model.

## 5. Failed / aborted experiments

Record every run that did not complete, with its cause, here. None recorded yet.
A smoke test (`--max_images 200`) is **not** an experiment and belongs nowhere
in this document.

## 6. Selection and one-time test evaluation

1. The only new Kaggle run was `e0_reference_group_split` with the documented
   `e0_reference` configuration on the verified property-disjoint split.
2. Its checkpoint metadata was verified against `training_gate.json`, the split
   and preflight SHA-256 values, and `training_summary.json` (best epoch 27,
   val Macro F1 0.4314855).
3. The test split was evaluated **once** through the gated cell in
   `docs/MEMBER_A_KAGGLE_READINESS.md`; metrics and the one-time seal are in
   `results/metrics.json` and `results/evaluation_gate.json`.
4. Its validation Macro F1 is below the historical random-split incumbent's
   0.4521, but the two use different split protocols and are not comparable.
   No further test evaluation is permitted.
