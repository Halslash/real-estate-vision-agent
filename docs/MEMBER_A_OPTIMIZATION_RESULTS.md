# Member A optimization results — status and evidence ledger

**Date:** 2026-11-20
**Status summary:** code **Implemented** and **Locally tested**; experiments
**Ready for Kaggle**; **no result is Experimentally validated** because no GPU,
dataset or checkpoint was available locally. No new performance number is claimed.

Legend (required by the brief):
- **Implemented** — code/config exists.
- **Locally tested** — exercised with CPU-only checks here (exact commands in §3).
- **Ready for Kaggle** — exact commands prepared; not yet run.
- **Experimentally validated** — would require a real Kaggle run + artifacts.

---

## 1. Deliverable status

| Item | Status | Evidence |
|---|---|---|
| Recoverable baseline checkpoint | **Implemented** | annotated Git tag `member-a-optimization-baseline` at `7ce74b6`; message records that Member B working-tree changes were left untouched. |
| Full source/config/doc audit | **Implemented** | `docs/MEMBER_A_OPTIMIZATION_AUDIT.md` (§1 number verification, §2 ranked diagnosis). |
| v3 opt-in balanced-tertile label rule | **Implemented + Locally tested** | `src/label_thresholds.py`, `src/dataset.py` (`_fit_v3_stats`), 9 pure-numpy tests. |
| Real-data v2/v3 distribution + tie audit | **Ready for Kaggle** | `src/label_audit.py` + `tie_summary`/`threshold_tie_impact`/`label_change_summary`; never run on the mounted data. |
| v3 accepted by train/evaluate | **Implemented + Locally tested** | `src/train.py` CLI `choices`, `src/evaluate.py` version guard; config contract tests pass. |
| Controlled experiments e11/e12 | **Implemented + Locally tested** | `configs/experiments.json`; `--list` and `--dry-run` verified. |
| Experiment protocol + guarded Kaggle cells | **Implemented** | `docs/MEMBER_A_EXPERIMENT_PLAN.md` §4b (fail-closed guard cell, audit cells, preflight). |
| Fail-closed overwrite protection | **Implemented + Locally tested** | `run_experiment.py --check`/artifact guard, `train.py --allow_existing`, `probe.py --force`, `label_audit.py --force`; historical names refused. |
| Leakage audit on real data | **Ready for Kaggle** | `src/data_audit.py` exists but has never been run on the mounted dataset. |
| Frozen-probe ceiling re-test | **Ready for Kaggle** | commands in plan §2/§4. |
| Any CNN training / validation result | **Ready for Kaggle** (not validated) | no artifact exists locally or in git history. |
| One-time test evaluation of winner | **Ready for Kaggle** (not validated) | gated on selection; plan §5. |
| Member B OOF scores | **Ready for Kaggle** (not validated) | `oof_contract` tests pass; real OOF not generated. |

---

## 2. What actually changed in this pass

Minimal, opt-in, reversible (full rationale in the audit doc):

- **`src/label_thresholds.py` (new)** — pure numpy tertile rules. `unique_index_tertiles`
  reproduces v2 for diagnostics; `sample_quantile_tertiles` is the v3 rule;
  `class_counts` reports resulting support. Added `tie_summary` (rows vs distinct
  values, largest tie group), `threshold_tie_impact` (rows sitting exactly on a
  cut) and `label_change_summary` (v2→v3 transition matrix) for the tie hypothesis.
- **`src/dataset.py`** — adds `LABEL_V3 = "v3_balanced_quantile"`. v3 reuses the
  v2 proxy value and shrunken city medians; `_fit_v3_stats` is layered on the
  untouched `_fit_v2_stats` and replaces only `class_threshold_low/high`.
  **Default remains v2**, so every existing checkpoint keeps its interpretation.
- **`src/train.py`, `src/evaluate.py`** — accept the new version string.
- **`configs/experiments.json`** — add `e11_v3_quantile_labels` (matched to `e0`
  on everything but the label rule) and `e12_v3_balanced_ce` (v3 + balanced CE).
- **`src/label_audit.py` (new)** — read-only audit printing real v2/v3 train-only
  thresholds, class counts/support and tie/assignment effects on each split.
- **`src/probe.py`, `src/train.py`, `src/run_experiment.py`** — fail-closed overwrite
  guards (distinct probe output per feature source, `--check` preflight, artifact
  detection) so re-running the notebook cannot silently reuse an earlier output.
- **`tests/test_label_thresholds.py` (new)** — 9 tests: v2 formula fidelity, v3
  balance under heavy ties, deterministic ordered cuts, non-finite filtering,
  fail-closed degenerate input, and the new tie/assignment diagnostics.

**Not changed:** default label version, v1/v2 code paths, `VisionAgent.score`,
OOF contract/columns, `condition_score`, test split, historical outputs, Member B
files, the original 11 experiments.

---

## 3. Local verification performed (CPU only)

```text
python3 -m compileall -q src tests                  -> OK
python3 -m unittest discover -s tests -v            -> Ran 42 tests: OK (skipped=10)
python3 src/run_experiment.py --list                -> 13 experiments listed
python3 src/run_experiment.py --name e11_v3_quantile_labels --dry-run -> OK
python3 src/run_experiment.py --name e12_v3_balanced_ce --dry-run     -> OK
python3 src/run_experiment.py --name e0_reference --check            -> OK (preflight)
```

The 10 skips are torch/torchvision-dependent `VisionAgent` and loss tests;
they run on Kaggle. Synthetic threshold checks (labelled **[SYNTHETIC]** in the
audit, **not** project data) show v2 can produce `[2324, 649, 27]` where v3 gives
`[711, 1201, 1088]` under heavy proxy-value rounding. This motivates the
experiment; it is not evidence that the real dataset has that tie mass.

---

## 4. Results tables — all cells `PENDING`

No value may be copied from the historical docs into these rows. The historical
incumbent (val Macro F1 0.4521 / epoch 15, test 0.4309) is a **separate documented
reference**, not a controlled comparison.

**Status of every experiment row below: Ready for Kaggle** — the code/config is
*Implemented* and *Locally tested*, but no Kaggle run exists, so none is
*Experimentally validated*. The single test-evaluation row is also
*Ready for Kaggle* and gated on selection.

| Run | Architecture | Label config | Resolution | Loss / weights | Seed | Val acc | Val Macro F1 | Per-class F1 (low/avg/high) | Support | Best ep | Duration | Peak VRAM |
|---|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|
| e0_reference | MobileNetV2 | v2_log_shrunk | 224 | CE / auto | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e0_reference_seed43 | MobileNetV2 | v2_log_shrunk | 224 | CE / auto | 43 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e0_reference_seed123 | MobileNetV2 | v2_log_shrunk | 224 | CE / auto | 123 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e11_v3_quantile_labels | MobileNetV2 | v3_balanced_quantile | 224 | CE / auto | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e11_v3_seed43 | MobileNetV2 | v3_balanced_quantile | 224 | CE / auto | 43 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e11_v3_seed123 | MobileNetV2 | v3_balanced_quantile | 224 | CE / auto | 123 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e12_v3_balanced_ce | MobileNetV2 | v3_balanced_quantile | 224 | CE / balanced | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e2_balanced_ce | MobileNetV2 | v2_log_shrunk | 224 | CE / balanced | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e4_focal | MobileNetV2 | v2_log_shrunk | 224 | Focal / none | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e5_ordinal_smooth | MobileNetV2 | v2_log_shrunk | 224 | Ordinal / none | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| e9_shrinkage_k1 | MobileNetV2 | v2_log_shrunk, K=1 | 224 | CE / auto | 42 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

### Test split (filled once, for the winner only, after selection)

| Run | Test acc | Test Macro F1 | Macro precision | Macro recall | Weighted F1 | Per-class F1 | Evaluation date |
|---|---:|---:|---:|---:|---:|---|---|
| winner (TBD) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

---

## 5. How to fill the cells (provenance rule)

For each `<output_dir>`:
- `experiment_run.json` → architecture, label config, loss/weights, seed, exact command.
- `models/training_summary.json` → best epoch, best validation metrics, duration, peak VRAM, resolved device.
- `models/training_history.csv` → train/val loss, accuracy and macro-F1 curves (attach to the report, not invented here).
- `metrics.json` and confusion-matrix figures from `evaluate.py` → **test** metrics only, once.

If an artifact is missing, write `MISSING`, not an estimate.

---

## 6. Decision rule (pre-registered)

1. Run the leakage audit; record its verdict before trusting any held-out number.
2. Compute validation Macro F1 with all three classes fixed, plus per-class support.
3. Require `e11_v3_quantile_labels` to beat `e0_reference` on **validation Macro F1**
   *and* to show a materially more balanced class support. If seed ranges overlap,
   call the label change **not validated** and keep v2.
4. Only a run that passes step 3 (or an existing-config run that wins outright)
   becomes the winner; only that run gets the single test evaluation.
5. Keep the incumbent if no new run wins. Never select on test.

---

## 7. Risks the operator must know

- The brief's `e0_reference` numbers (`0.4563`, accuracy `~0.4606`, epoch `28`)
  have **no supporting artifact** in this repo; `0.4521`/epoch 15 is the documented
  incumbent. Treat the former as unverified.
- v3 is an **opt-in** label methodology change; it must never be applied silently
  to a historical checkpoint or to Member B's OOF labels.
- No GPU work was performed here. Every Kaggle cell is `PENDING` by design;
  "Ready for Kaggle" is not "validated".
- Member B's uncommitted files were left untouched, and nothing was pushed.
