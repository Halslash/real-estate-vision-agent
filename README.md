# Real Estate Vision Agent — Member A (final)

**Projet 10: Évaluation Immobilière Multi-Modale (PropTech).**

This repository is **Member A's completed contribution**: an exterior-image CNN
that emits a bounded visual-proxy score and a full 3-class probability vector.
**Member B (structural model + final price estimator) is not implemented.**
The full pipeline is specified but pending in
`docs/FULL_VALUATION_IMPLEMENTATION_PLAN.md`.

> **Read first — scientific scope.** The target is a **market-derived visual
> proxy** (a tertile of log price-per-square-foot relative to a shrunken city
> median), **not** a verified physical-condition or aesthetic grade and **not** a
> price estimate. `condition_score` is a soft feature, `confidence` is the maximum
> class probability (**not calibrated**), and no end-to-end price prediction exists
> here.

## Results at a glance

| Run | Label config | Validation Macro F1 | Test Macro F1 | Status |
|---|---|---:|---:|---|
| **`e0_reference_group_split` (selected)** | `v2_log_shrunk`, K=10 | **0.4314855** | **0.4240** | Trained and evaluated once on a property-disjoint split |
| `e11_v3_quantile_labels` | `v3_balanced_quantile` | — | — | Configured only; no checkpoint or result |
| `e12_v3_balanced_ce` | `v3_balanced_quantile` + balanced CE | — | — | Configured only; no checkpoint or result |

`e0` one-time test evaluation (**2,308** samples): accuracy **0.4281**, macro
precision **0.4277**, macro recall **0.4296**, macro F1 **0.4240**, weighted F1
**0.4233**; per-class F1 low **0.4582**, average **0.3539**, high **0.4600**
(average-class recall **0.3070**). The historical random-split incumbent
(validation F1 `0.4521`, test F1 `0.4309`) is preserved in
`docs/EXPERIMENT_RESULTS.md` but is **not comparable** — its test set shared
addresses/images with training. `e1`–`e10` were not run.

**Selection:** validation Macro F1 only (fixed labels `[0,1,2]`). `e0_reference_group_split`
is the only run with measured validation evidence under the leakage-resistant
protocol; e11/e12 have no checkpoints or validation results and were not selected.
No test metric was used for selection or tuning. e0's test split was evaluated
once, is sealed, and must never be reused. An e11 test evaluation was **not**
performed; doing so later would be a subsequent look at an already-inspected test
set, not a fresh evaluation.

## What this repository contains

```
Exterior photo
  -> RGB + Resize(256) + CenterCrop(224) + ImageNet normalization
  -> ImageNet-pretrained MobileNetV2 -> dropout -> 3-class softmax
  -> P(low), P(average), P(high)
  -> condition_score = 0.5*P(average) + 1.0*P(high), clipped to [0, 1]
```

Two-phase training in `src/train.py`: frozen backbone head-only phase, then
fine-tuning of the last 4 blocks; validation macro F1 drives early stopping in
phase 2. The final model used **30 epochs total** (not 20), batch size 64, AdamW
`1e-4` / `1e-5`, cosine schedule, cross-entropy with auto class weights
(effective-number β = 0.9999), label smoothing `0.05`, improved augmentation,
seed 42, best epoch **27**. Pretrained ImageNet weights are mandatory; all label
statistics are fitted on training rows only.

Dataset: `socal2.csv` (~15,474 rows) with images `<image_id>.jpg`, expected on
Kaggle under `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/`.
The final split is `property_address_plus_exact_hash` (seed 42): **10,829 train /
2,337 validation / 2,308 test**, with a local `status=PASS` preflight showing no
address or exact-image-hash group crossing splits. Perceptual near-duplicates at
different addresses were not grouped.

## Installation

```bash
git clone https://github.com/Halslash/real-estate-vision-agent.git
cd real-estate-vision-agent
pip install -r requirements.txt
```

## Checkpoint included in the repository

**Cloning the repository is sufficient to obtain the weights.** The selected
final checkpoint is tracked directly at
`checkpoints/member_a_cnn_e0_reference_group_split_best.pth` (**9,171,467 bytes**,
SHA-256 `9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b`).
The global `*.pth` exclusion in `.gitignore` is unchanged; this single checkpoint
was explicitly added and is tracked by Git.

Verify after cloning:

```bash
sha256sum checkpoints/member_a_cnn_e0_reference_group_split_best.pth
# expected: 9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b
git ls-files --error-unmatch checkpoints/member_a_cnn_e0_reference_group_split_best.pth
```

A GitHub Release is an **additional, optional** distribution route, not the only
way to obtain the model: tag `v1.0.0-member-a-cnn`, asset
`member_a_cnn_e0_reference_group_split_best.pth`, URL
`https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth`
(prepared but not yet published). See `docs/MODEL_RELEASE.md` for release notes,
publishing and download verification.

## Inference example

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="checkpoints/member_a_cnn_e0_reference_group_split_best.pth")
result = agent.score("example_house.jpg")          # agent.score_batch(paths, batch_size=64)
print(result)
```

Returns `condition_score`, `label`, `probabilities` (`low`/`average`/`high`),
`confidence`, `model`, `score_type = "market_derived_visual_proxy"`. Inference
needs only the checkpoint (no CSV, dataset, OOF file or pretrained download).
Invalid paths/corrupt images raise errors; no fallback score is fabricated.

## Training and reproduction (Kaggle GPU)

The final run used the pinned cells in `docs/MEMBER_A_KAGGLE_READINESS.md`
(source commit `bc39d102d2e9cde83144966d6e6c910926109429`): a fail-closed
preflight, then training with
`--split_file /kaggle/working/output/leakage_resistant_split/split_indices.json`
and `--output_dir /kaggle/working/output/experiments/e0_reference_group_split`,
then the one-time evaluation cell. Do **not** rerun the test evaluation. For the
config-driven experiments (including e11/e12, which have not been run):

```bash
python src/run_experiment.py --list
python src/run_experiment.py --name e0_reference --dry-run
python src/run_experiment.py --name e0_reference             # train only
```

Local checks: `python -m unittest discover -s tests -v`. Weights, splits and
sealed metrics are excluded from Git by design.

## Documentation

| Document | Purpose |
|---|---|
| `docs/ACADEMIC_REPORT.md` | **Main academic report** — method, results, limitations, integration needs. |
| `docs/MODEL_CARD.md` | Intended use, prohibited interpretations, model details. |
| `docs/EXPERIMENT_RESULTS.md` | Final + historical results and provenance. |
| `docs/FINAL_MEMBER_A_REPORT.md` | Detailed 13-section technical report. |
| `docs/MEMBER_A_TECHNICAL_HANDOFF.md` | `VisionAgent`, probability vector, OOF row contract and leakage rules. |
| `docs/FULL_VALUATION_IMPLEMENTATION_PLAN.md` | Pending end-to-end architecture; distinguishes proxy from price prediction. |
| `docs/MODEL_RELEASE.md` | Tracked-checkpoint details, SHA-256 verification, optional release notes and commands. |
| `docs/MEMBER_A_KAGGLE_READINESS.md` | Pinned preflight / training / one-time evaluation cells. |
| `docs/KAGGLE_GUIDE.md`, `docs/MEMBER_A_LEAKAGE_RESISTANT_SPLIT.md` | Supporting commands and split rationale. |
| `docs/archive/` | Superseded audit/status snapshots kept for provenance. |

## Scope and limitations

Implemented: the CNN visual proxy, self-contained checkpoint/inference API, OOF
contract utilities, evaluation scripts, configuration-driven experiments, tests,
and documentation. Pending: group-aware OOF scores, structural-feature model,
final price estimator, integration, and any Member B evaluation. The existing
`oof_visual_scores.csv`/row-level-fold path is unsafe; Member B must regenerate
group-aware folds and train fold-specific CNNs. The final e0 checkpoint is
**prohibited** from scoring outer-test rows that overlap its training data.
