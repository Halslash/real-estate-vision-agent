# Experiment plan — Member A optimization pass

**Status:** planned, code implemented and locally checked; **no Kaggle run executed**.
**Companion:** `docs/MEMBER_A_OPTIMIZATION_AUDIT.md` (why), `docs/MEMBER_A_OPTIMIZATION_RESULTS.md` (status/results table).
**Baseline checkpoint:** annotated tag `member-a-optimization-baseline` at `7ce74b6`.

> Interpretation warning: every target is a market-derived visual proxy, not a
> physical-condition label or a price. `condition_score` is a soft feature only.

---

## 1. Non-negotiable protocol

1. **One fixed split for every run.** Seed 42, 70/15/15 stratified, regenerated
   identically by `make_stratified_split`. No new split per experiment.
2. **Test split is touched exactly once**, after model selection, for the single
   winner. Never use test accuracy/F1 to choose an experiment. `e0`–`e12` are
   train+validation only.
3. **Primary selection metric: validation Macro F1 over all 3 classes** (fixed
   `labels=[0,1,2]`). Report accuracy, macro/weighted precision/recall/F1,
   per-class precision/recall/F1/support, confusion matrix and train/val curves
   for every run. A run with a missing class gets Macro F1 with that class scored
   as 0, never silently dropped.
4. **Train-only fitting.** Clipping bounds, city medians, shrinkage and tertile
   thresholds come from the training partition only. v3 is no exception.
5. **No threshold tuning on validation for `condition_score`.** It stays
   `0.5·P(average) + 1.0·P(high)` in every run.
6. **No grid search.** Each experiment changes one factor (or one declared
   combination). Fixed seeds and full resolved configs are logged by
   `run_experiment.py` in `<output_dir>/experiment_run.json`.
7. **Historical outputs are read-only.** Never write to
   `/kaggle/working/output/models`, `.../improved_mobilenet`, `.../results` or
   `.../output`. `run_experiment.py` refuses those names.
8. **Member B interface is frozen.** `VisionAgent.score(image_path)` and the OOF
   CSV columns stay byte-compatible. OOF generation is a separate protocol and
   never reads the test split.

---

## 2. Experiment queue (bounded, priority order)

All rows use the shared defaults in `configs/experiments.json` unless noted
(seed 42, MobileNetV2, 224px, improved augmentation, CE + label smoothing 0.05,
`class_weight=auto`, 30 epochs, cosine, patience 5).

| Order | ID | One factor changed | Question answered | Priority |
|---:|---|---|---|---|
| 1 | `e0_reference` | none (fresh run on current code) | Establish an artifact-backed controlled reference; the brief's `0.4563/epoch 28` is unverifiable. | **Must run first** |
| 2 | `e0_reference --probe` | frozen ImageNet probe | Re-test the probe ≈ CNN ceiling hypothesis on current code. | **Must run first** |
| 3 | `e11_v3_quantile_labels` | label rule v2 → v3 | Does distinct-value tertiling starve a class? Direct matched comparison to `e0_reference`. | **Highest-value new run** |
| 4 | `e12_v3_balanced_ce` | v3 + `class_weight=balanced` | Once cuts are balanced, is explicit inverse-frequency weighting still useful? | High |
| 5 | `e2_balanced_ce` | weighting only, v2 labels | Separates the loss effect from the label effect (pairs with #4). | High |
| 6 | `e4_focal` | focal loss, v2 labels | Do hard/noisy proxy examples benefit? | Medium |
| 7 | `e5_ordinal_smooth` | ordinal soft targets, v2 labels | Does the ordered nature of low/average/high help? | Medium |
| 8 | `e9_shrinkage_k1` | K=10 → 1 | Is city shrinkage concentrating borderline cases? | Medium |
| 9 | `e1_no_weighting`, `e3_effective_number`, `e6_efficientnet_b0`, `e7_higher_resolution`, `e8_legacy_augmentation`, `e10_longer_finetune` | one each | Already-defined ablations; run only if time/GPU budget allows. | Optional |

**Stop rule:** stop after row 4 if `e11` shows no class-support problem and no
Macro F1 gain over `e0_reference` — that would falsify the main hypothesis and
make the rest of the label/loss queue low-value. Do not "keep tuning until it
wins".

**Leakage gate (run before interpreting any result):**
`!python src/data_audit.py`. If substantial cross-split duplicate/near-duplicate
or street/city overlap is found, record it; the held-out numbers are optimistic
and the primary conclusion becomes a data problem, not a model problem.

---

## 3. Seed variability (required, not optional)

A single seed's Macro F1 is not a result. Run two extra replicates of the
**reference** and of the **best label variant** with identical settings except
`--seed`, writing to distinct output dirs (these are deliberately direct commands
so the config's "all seed 42" invariant stays intact):

```python
# reference replicates
!python src/train.py --seed 43  --output_dir /kaggle/working/output/experiments/e0_reference_seed43
!python src/train.py --seed 123 --output_dir /kaggle/working/output/experiments/e0_reference_seed123

# best label variant replicates (use e11 if it wins; otherwise swap the label flag)
!python src/train.py --seed 43  --label_config_version v3_balanced_quantile \
    --output_dir /kaggle/working/output/experiments/e11_v3_seed43
!python src/train.py --seed 123 --label_config_version v3_balanced_quantile \
    --output_dir /kaggle/working/output/experiments/e11_v3_seed123
```

Report mean ± range (not just the best seed) for both arms. If the arms' ranges
overlap, the label change is **not** experimentally validated.

---

## 4. Exact Kaggle commands

```python
# A. setup (run once per notebook session)
!git clone <YOUR_REPOSITORY_URL> /kaggle/working/vision_agent
%cd /kaggle/working/vision_agent
!python src/inspect_csv.py
!python src/data_audit.py                       # leakage gate — read-only
!python -m unittest discover -s tests -v        # torch tests run here

# B. inspect the runner, then execute the queue in the order above
!python src/run_experiment.py --list
!python src/run_experiment.py --name e0_reference --dry-run
!python src/run_experiment.py --name e0_reference --probe
!python src/run_experiment.py --name e11_v3_quantile_labels --probe
!python src/run_experiment.py --name e12_v3_balanced_ce
!python src/run_experiment.py --name e2_balanced_ce
# ...remaining rows from section 2 as budget allows

# C. seed replicates — section 3 commands verbatim

# D. frozen-probe-only diagnostic on any run (no weights touched)
!python src/probe.py \
    --checkpoint /kaggle/working/output/experiments/e0_reference/models/best_model.pth \
    --output_dir /kaggle/working/output/experiments/e0_reference
```

Every run writes `experiment_run.json` (resolved config + exact command),
`models/training_summary.json`, `models/training_history.csv`, curves and
confusion matrices into its **own** directory. Record the peak GPU memory and
duration from the summary in the results table.

---

## 4b. Fail-closed notebook cells (copy verbatim, in order)

These cells make accidental reuse impossible: a guard cell fails the whole
notebook if **any** planned output path already exists. The code itself also
fails closed (`run_experiment.py --check` / default, `train.py` default,
`probe.py --force`, `label_audit.py --force`), so there are two independent
barriers against overwriting historical checkpoints or earlier outputs. Protected
names (`baseline`, `improved_mobilenet`, `models`, `results`, `output`) are
rejected by the runner regardless.

**Cell 0 — guard (run first; re-run after each successful stage):**
```python
import os
ROOT = "/kaggle/working/output/experiments"
PLANNED = [
    f"{ROOT}/e0_reference", f"{ROOT}/e11_v3_quantile_labels",
    f"{ROOT}/e12_v3_balanced_ce", f"{ROOT}/e2_balanced_ce",
    f"{ROOT}/e0_reference_seed43", f"{ROOT}/e0_reference_seed123",
    f"{ROOT}/e11_v3_seed43",  f"{ROOT}/e11_v3_seed123",
]
ARTIFACTS = ["models/best_model.pth", "experiment_run.json",
             "results/metrics.json", "results/probe_results.json",
             "results/probe_results_checkpoint.json"]
def guard(paths):
    hits = [os.path.join(d, a) for d in paths for a in ARTIFACTS
            if os.path.exists(os.path.join(d, a))]
    if hits:
        raise RuntimeError("Refusing to proceed; prior run artifacts exist:\n  "
                           + "\n  ".join(hits))
    print(f"OK: {len(paths)} output dirs clean.")
guard(PLANNED)
```

**Cell 1 — real-data label distribution + tie/assignment audit (train-only
thresholds; no training, no writes to any run dir):**
```python
!python src/label_audit.py \
    --output /kaggle/working/output/label_audit/label_audit.json
```
Read the printed v2/v3 train counts, `rows_in_tie_groups`,
`rows_on_a_threshold` and `n_changed` (v2→v3). This is the evidence for the tie
hypothesis. If `label_audit.json` already exists, the script refuses to
overwrite it (pass `--force` deliberately only if you intend to replace it).
Val/test rows in that report are **descriptive only** and must never select a
model or a label version.

**Cell 2 — leakage gate (read-only; run before trusting any held-out number):**
```python
!python src/data_audit.py --split_file "{ROOT}/e0_reference/models/split_indices.json"
```
Inspect `/kaggle/working/output/data_audit/data_audit.json`. Substantial
cross-split duplicate/street/city overlap makes the held-out numbers optimistic;
record that and treat the conclusion as a data problem, not a model problem.

**Cell 3 — preflight every experiment (prints resolved commands, writes nothing):**
```python
!python src/run_experiment.py --name e0_reference --check
!python src/run_experiment.py --name e11_v3_quantile_labels --check
!python src/run_experiment.py --name e12_v3_balanced_ce --check
```

**Cell 4 — run reference and the two label experiments independently:**
```python
!python src/run_experiment.py --name e0_reference --probe
!python src/run_experiment.py --name e11_v3_quantile_labels --probe
!python src/run_experiment.py --name e12_v3_balanced_ce
guard(PLANNED)  # confirm no unexpected reuse
```

**Cell 5 — probe ceiling diagnosis on the reference (both feature sources,
distinct output files; train→val only, no weights changed):**
```python
!python src/probe.py \
    --checkpoint {ROOT}/e0_reference/models/best_model.pth \
    --output_dir {ROOT}/e0_reference \
    --feature_source pretrained
!python src/probe.py \
    --checkpoint {ROOT}/e0_reference/models/best_model.pth \
    --output_dir {ROOT}/e0_reference \
    --feature_source checkpoint
```
`pretrained` writes `results/probe_results.json`; `checkpoint` writes
`results/probe_results_checkpoint.json` — the two cannot overwrite each other.
Repeat both with the winning label variant's checkpoint if `e11`/`e12` is chosen.
Compare val macro F1: near-equal probe ≈ CNN ⇒ likely feature/label ceiling;
probe ≫ CNN ⇒ likely fine-tuning-recipe limitation.

**Cell 6 — seed replicates for the winner only (section 3 commands), then
aggregate mean ± range on validation Macro F1.**

**Cell 7 — select on validation Macro F1 + seed ranges + leakage gate, then the
single one-time test evaluation (section 5). Run evaluate.py for the winner only.**

---

## 5. The one-time test evaluation (after selection only)

Selection happens on validation Macro F1 + seed ranges + the leakage gate. Only
then:

```python
!python src/evaluate.py \
    --output_dir /kaggle/working/output/experiments/<WINNER>
```

This evaluates the untouched test split **once** and writes `metrics.json`,
per-class metrics and a confusion matrix. Do not re-run it for a losing model,
do not compare two models on test, and do not copy its numbers into a validation
cell. The historical test Macro F1 `0.4309` is already spent and is not a
selection target.

---

## 6. Member B OOF (separate protocol, after selection)

```python
!python src/make_folds.py --n_splits 5
# train each fold with --split_file, then:
!python src/oof_predict.py --output_dir /kaggle/working/output/oof
```

Folds come from the canonical **training split only** (`test: []` in each split
file). `oof_contract.py` fails closed on overlap, wrong fold ownership, missing
coverage or any test-id leakage. The output CSV keeps the required columns
`image_id, fold, condition_score, p_low, p_average, p_high, label, confidence,
model, checkpoint`. OOF scores must come from fold checkpoints that never saw
the row they score.

---

## 7. Results cells to fill (all `PENDING` until a real run)

| ID | Val acc | Val Macro F1 | Per-class F1 (low/avg/high) | Support | Best epoch | Duration | Peak VRAM | Seed range |
|---|---:|---:|---|---|---|---:|---:|---|
| e0_reference | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e11_v3_quantile_labels | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e12_v3_balanced_ce | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| (others) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Fill only from `training_summary.json`, `metrics.json`, `training_history.csv`
and the checkpoint metadata. Never estimate, never copy between rows, never use
`--max_images` smoke-test output as a result.
