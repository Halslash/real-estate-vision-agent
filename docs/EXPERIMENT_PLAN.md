# Experiment plan — Member A CNN visual proxy

All experiments share: seed 42, the same 70/15/15 stratified split, batch size
64, ImageNet pretrained weights, ImageNet normalization, validation macro F1 for
model selection, and the untouched test set evaluated **once** for the final
winner only. Each run writes to a distinct `--output_dir`; the original baseline
checkpoint is never overwritten.

> No results are recorded here yet. Every "Result" column is intentionally left
> as **PENDING** until the command has actually been executed on Kaggle GPU.
> Never fill a cell with an estimate or a number copied from another experiment.

## Shared command skeleton

Replace `<EXPERIMENT>` and the changed flags to build a command. See
`docs/KAGGLE_GUIDE.md` for fully copy-pasteable commands and real Kaggle paths.

```python
!python src/train.py \
    --csv_path <CSV> --images_dir <IMAGES> \
    --output_dir /kaggle/working/output/<EXPERIMENT> \
    --epochs <EPOCHS> --batch_size 64 --seed 42 --num_workers 2 \
    --architecture <ARCH> --image_size <SIZE> --resize_size <RESIZE> \
    --label_config_version <LABEL> --shrinkage_k <K> \
    --augmentation <AUG> --label_smoothing <LS> --scheduler <SCHED> \
    --lr 1e-4 --lr_finetune 1e-5 \
    --id_col image_id --price_col price --sqft_col sqft --city_col citi
```

## Experiments

| ID | Objective | Changed parameters vs reference | Output directory | Primary metric | Result (PENDING) | Decision rule |
|---|---|---|---|---|---|---|
| E0 | Baseline diagnostics | Existing checkpoint; evaluation + reports only, no retraining | `/kaggle/working/output` (baseline folder) | Test macro F1, predicted-class histogram, train/val gap | PENDING | Establish reference; check prior collapse. Test output here is descriptive only. |
| E1 | Label hygiene | `--label_config_version v2_log_shrunk --shrinkage_k 10`; log-ratio tertiles + shrunken city medians; everything else baseline | `output/experiments/E1_labels` | **Validation** macro F1 | PENDING | Keep v2 if validation macro F1 improves over baseline; otherwise keep the original labels and investigate the probe first. |
| E2 | Augmentation alignment | `--augmentation improved` (no rotation, no aggressive crop, restrained jitter) | `output/experiments/E2_augmentation` | Validation macro F1 | PENDING | Keep if equal or better than E1; reduces train/inference mismatch regardless of metric. |
| E3 | Schedule + capacity | `--epochs 20 --scheduler cosine` (10+10 split, early stopping now active) | `output/improved_mobilenet` (primary run combines E1+E2+E3) | Validation macro F1 | PENDING | Keep if validation improves; watch train/val gap for overfitting. This is the default primary run in the Kaggle guide. |
| E4 | Regularisation | `--label_smoothing 0.1` on the best of E1–E3 | `output/experiments/E4_smoothing` | Validation macro F1 + confidence distribution | PENDING | Keep if validation/calibration improves. Remember: smoothing softens `confidence`; it is not certainty. |
| E5 | Frozen-feature probe | `src/probe.py --feature_source pretrained`; train→val logistic regression only | writes `results/probe_results.json` in the chosen run | Probe validation macro F1 | PENDING | Probe ≈ CNN → feature/label ceiling; probe ≫ CNN → training-recipe limitation. No test data. |
| E6 | Architecture comparison | `--architecture efficientnet_b0`, all data/label/split settings identical to the primary run | `output/efficientnet_b0` | Validation macro F1 | PENDING | Adopt only if clearly better than the best MobileNetV2 validation result. |
| E7 | Shrinkage sensitivity | `--shrinkage_k 3`, then `--shrinkage_k 30`, holding E3 settings | `output/experiments/E7_k3`, `E7_k30` | Validation macro F1 | PENDING | Choose the most stable value; document the choice. Do not tune on test. |
| E8 | Image resolution | `--image_size 299 --resize_size 320`, holding the best label/training settings | `output/experiments/E8_299` | Validation macro F1 | PENDING | Keep only if the gain justifies ~2× Kaggle GPU time. Checkpoint preprocessing metadata changes; do not mix with 224px checkpoints. |

Multi-seed runs are deliberately **not** planned: `--seed` also fixes the
70/15/15 split, so changing it would break the fixed-split comparability that
all experiments above rely on.

## Decision protocol

1. Run E0 to establish what the existing baseline actually does on the test set.
2. Use **validation** macro F1 to choose among E1–E4 and E6–E8. The test split is
   touched only once, for the final selected configuration.
3. Run E5 early: if the probe already reaches the CNN's validation score, the
   label/feature ceiling is the likely cause and further architecture or
   resolution work (E6, E8) has low expected value.
4. Record the exact command, checkpoint path, output directory and measured
   metrics for every run in `docs/FINAL_MEMBER_A_REPORT.md`.
5. Never report smoke-test numbers (`--max_images 200`) as performance.

## Scientific guardrails

- The target remains a market-derived proxy. A better macro F1 means the image
  carries more signal about *relative price per square foot in this market*; it
  does **not** prove a physical-condition or aesthetic classifier.
- `label_config_version` is stored in every checkpoint. Do not reinterpret a
  legacy checkpoint with v2 thresholds, or a 224px checkpoint with 299px
  preprocessing.
- Report limitations and baselines (uniform/random and majority) alongside every
  headline metric.
