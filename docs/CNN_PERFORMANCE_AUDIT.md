# CNN performance audit — Member A visual proxy

**Date:** 2026-11-20
**Scope:** `src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`,
`src/probe.py`, `src/vision_agent.py`, experiment configuration and all docs.
**Methods:** full source read, git state check, `python -m compileall`, local
`unittest` run, dry-run of the experiment runner. **No model was trained or
evaluated locally** (no torch/Tesla GPU here); all performance numbers below are
pre-existing Kaggle measurements preserved verbatim, not new results.

Evidence labels: **[OK]** verified in source; **[CONFIRMED]** factual defect or
property; **[HYPOTHESIS]** plausible but unmeasured, requires a GPU experiment.

---

## 1. Executive summary

The pipeline is conceptually sound: train-only label statistics, self-contained
checkpoints, untouched test split and a stable inference API. The problems are
mostly in **measurement and experiment discipline**, not in the CNN itself:

- the selection metric could silently change label support between epochs;
- documentation claimed "all results PENDING" while measured results existed;
- no leakage diagnostics, no out-of-fold scores for Member B, no loss/architecture
  experiments, no timing/VRAM record and no tests;
- the weak `average` class is real but **not** evidence of a broken model — it is
  an expected consequence of a middle-tertile market proxy.

None of the fixes change the architecture, the `v2_log_shrunk` label definition,
the score formula or the historical checkpoints.

---

## 2. Pre-existing measured results (retained, not reproduced)

| Run | Split | Accuracy | Macro F1 | Notes |
|---|---|---:|---:|---|
| Original baseline (`v1_legacy`) | test | 0.4065 | 0.4057 | Different label methodology — **not** a controlled comparison with the rows below |
| Improved MobileNetV2 (`v2_log_shrunk`) | validation | 0.4589 | 0.4521 | Best epoch 15 |
| Improved MobileNetV2 (`v2_log_shrunk`) | test | 0.4384 | 0.4309 | Macro precision 0.4323, macro recall 0.4362, weighted F1 0.4327 |
| Frozen ImageNet logistic probe | validation | 0.4580 | 0.4561 | Train→val only; test untouched |

Improved per-class **test** F1: low `0.4684`, average `0.3293`, high `0.4948`.

The frozen probe (0.4561) is essentially level with the CNN's validation F1
(0.4521). **Interpretation [HYPOTHESIS]:** the ceiling is likely in the
feature/label signal, not in the classifier head. This is why the experiment
plan prioritises label, loss and data/leakage questions over head tuning.

---

## 3. Findings

### 3.1 Correctness defects (fixed)

1. **[CONFIRMED] Variable-label macro F1 during checkpoint selection.**
   `run_epoch` called `f1_score(..., average="macro", zero_division=0)` without
   `labels=`. When a class is absent from an epoch's true *and* predicted labels,
   sklearn scores only 2 classes, so `val_macro_f1` values from different epochs
   were not on the same scale. `evaluate.py` and `probe.py` already passed fixed
   labels, so only training selection was affected. **Fix:** `labels=[0,1,2]`
   in `src/train.py`. This does not modify any saved checkpoint; it makes future
   runs comparable. On large runs where all three classes appear every epoch the
   numeric result is unchanged.

2. **[CONFIRMED] Documentation/results mismatch.** Every results doc said
   "PENDING" despite the measured numbers in §2. **Fix:** results are recorded
   with provenance; new runs stay explicitly `PENDING` until executed.

3. **[CONFIRMED] No test suite.** **Fix:** `tests/` with stdlib `unittest`
   covering config/CLI contracts (runs everywhere) and checkpoint loading,
   single/batch scoring, invalid paths and output validity (skip cleanly without
   torch; run on Kaggle).

4. **[CONFIRMED] No experiment isolation or reproducibility record.**
   **Fix:** `src/run_experiment.py` + `configs/experiments.json`; per-run output
   dirs, protected historical names, full resolved config and command log in
   `experiment_run.json`.

5. **[CONFIRMED] No duration / GPU-memory record.** **Fix:**
   `training_summary.json` now contains `training_duration_seconds`,
   `peak_gpu_memory_mb` and `resolved_device`.

### 3.2 Verified-correct properties (no change)

6. **[OK] Train-only statistic fitting.** `fit_preprocessing_stats` receives
   only `train_records`; clipping bounds, global/city medians, shrinkage and
   tertile thresholds never see validation or test rows. Version and full stats
   are saved into the checkpoint.

7. **[OK] Preprocessing consistency.** Train transform (improved recipe) and
   eval transform both use `Resize(256) → CenterCrop(224) → ImageNet normalize`;
   only flip + mild colour jitter differ at train time. `VisionAgent.load()`
   rebuilds the transform from checkpoint metadata, so train/inference geometry
   cannot drift.

8. **[OK] Checkpoint self-containment and API stability.** `model.py` stores
   architecture, class map, preprocessing, label stats, config, best epoch,
   metrics and history. `VisionAgent` exposes `VisionAgent(device=...)`,
   `.load()`, `.score()`, `.score_batch()` with the required output fields,
   handles RGB/grayscale/RGBA and never invents fallback probabilities.

9. **[OK] Shrinkage `K=10`.** The shrunk median is
   `(n·median_city + K·median_global)/(n+K)`, so a one-property city cannot
   define its own reference. This is defensible; sensitivity is still tested
   (`e9_shrinkage_k1`) because the correct K is an empirical question.

### 3.3 Open risks (measured on Kaggle, not here)

10. **[CONFIRMED] No leakage diagnostics.** `dataset.py` deduplicates by
    normalized image-id stem only. There is no exact-byte/perceptual-duplicate
    check and no street/city grouping in the split. Two photos of the same
    property or two listings on the same street can land in different splits
    and inflate held-out scores. **Fix:** read-only `src/data_audit.py` reports
    CSV→image mismatches, exact hashes, near-duplicate dHashes across splits and
    street/city groups spanning splits. It changes no data.

11. **[CONFIRMED] Member B in-sample leakage risk.** If Member B trains on CNN
    scores produced by the same model that saw those images during training,
    the structural model will over-trust the visual feature. **Fix:** documented
    OOF protocol in `docs/MEMBER_B_INTEGRATION.md`, implemented by
    `src/make_folds.py` and `src/oof_predict.py` (see §4).

12. **[HYPOTHESIS] Weak `average` class (test F1 0.3293).** Plausible causes:
    (a) the middle tertile is definitionally a heterogeneous residual bin;
    (b) shrinkage pushes borderline cities toward the global median and
    concentrates ties near thresholds; (c) the market proxy is noisy for
    mid-priced homes. Targeted experiments: `e1`–`e5`, `e9`.

13. **[HYPOTHESIS] Overlap/noisy labels.** Tertiles of a continuous proxy force
    near-identical properties into different classes. Focal loss (`e4`) and
    ordinal smoothing (`e5`) directly test whether soft targets help.

14. **[HYPOTHESIS] Image–CSV matching.** The loader only accepts `<image_id>.jpg`.
    `data_audit.py` reports other extensions and missing files diagnostically;
    no silent extension substitution was added to the training path.

15. **[CONFIRMED] Deferred, deliberately not implemented:** regression head,
    calibration/decision-threshold tuning and conformal-style uncertainty.
    A regression head would change the 3-class checkpoint/score contract that
    Member B depends on; threshold tuning on validation is legitimate but must
    not be applied to the test split or to the `condition_score` definition.
    They are documented as future work, not reported as experiments.

---

## 4. New tooling (what changed in code)

| File | Purpose |
|---|---|
| `src/losses.py` | CE, focal, ordinal-smooth; `none`/`auto`/`balanced`/`effective_number` weighting |
| `src/data_audit.py` | Read-only CSV/image/duplicate/geographic leakage report |
| `src/make_folds.py` | K OOF folds from the **training split only**; test stays empty |
| `src/oof_predict.py` | Honest OOF `condition_score` CSV; fails closed on coverage/leakage errors |
| `src/run_experiment.py` | Config-driven train(/probe/eval) runner with output isolation |
| `configs/experiments.json` | 11 controlled experiments, seed 42, shared split/labels |
| `tests/` | Config/CLI + VisionAgent/loss tests |
| `src/train.py` | `--split_file`, loss/weight flags, fixed-label F1, duration/VRAM, metadata |
| `src/dataset.py` | `validate_labeled_split(require_test=...)` for OOF folds |

---

## 5. Bottom line

The existing improved checkpoint remains the incumbent. A new model replaces it
**only** if it wins on validation macro F1 with per-class support reported, and
then receives exactly one test evaluation. Until GPU runs exist, no performance
claim is made beyond the historical numbers in §2.
