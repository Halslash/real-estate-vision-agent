# Final academic readiness audit — Member A visual proxy CNN

**Date:** 2026-11-20
**Purpose:** a final, independent classification of what is actually true about
this contribution. It deliberately separates code that was inspected, numbers
that were really measured on Kaggle, code that exists but has never been run,
and open limitations. Nothing here is a new experimental result.

**Method:** full source and documentation read, `python -m compileall -q src
tests`, local `python -m unittest discover -s tests -v`, experiment-runner
`--list` / `--dry-run`, and git state check. The local machine has **no torch,
torchvision, dataset or checkpoints**, so no training, inference or Kaggle
evaluation was performed by this audit.

> Interpretation warning: the target is a market-derived visual proxy, not
> verified physical condition and not a price. See `docs/MODEL_CARD.md`.

---

## Category 1 — Verified / completed (checked locally, no GPU needed)

| Item | Evidence | Verdict |
|---|---|---|
| Whole repository inspected before changes | file tree, all `src/`, `tests/`, `configs/` and `docs/` read; git clean at `87785a6` | **Verified** |
| Historical checkpoints and result folders preserved | no write to `/kaggle/working/output/models` or `.../improved_mobilenet`; protected names rejected by `run_experiment.py` and tested | **Verified** |
| Stable inference API | `VisionAgent(device=...)`, `.load()`, `.score()`, `.score_batch(paths, batch_size=64)` with all six required output fields; RGB/grayscale/RGBA; no fabricated fallback scores | **Verified in source + tests** |
| Fixed 3-label selection F1 | `train.py::run_epoch` uses `labels=list(range(NUM_CLASSES))` | **Verified** |
| Per-class precision/recall/F1, support and confusion matrices | `evaluate.py` uses fixed labels, raw + row/column-normalized matrices and figures | **Verified** |
| Train-only statistics | `fit_preprocessing_stats` receives only `train_records`; version/stats saved in checkpoint | **Verified** |
| Preprocessing consistency | train/eval/agent transforms built from the same saved config | **Verified** |
| 11 isolated experiments | `configs/experiments.json`, `run_experiment.py`, distinct output dirs, protected names | **Verified** |
| Loss / weighting alternatives | focal, ordinal-smooth, effective-number, balanced/none/auto in `losses.py` | **Verified in source** |
| OOF isolation invariants | `oof_contract.py` + `tests/test_oof_contract.py`: fold ownership, no train/val overlap, no test leakage, no self-training row, complete one-per-row coverage | **Verified (pure logic tests)** |
| Member B CSV now carries probabilities | `oof_predict.py` writes `p_low/p_average/p_high` alongside `condition_score`, `label`, `confidence`, IDs and provenance | **Verified in source + tests** |
| Demonstration builder | `demonstrate.py` copies originals and emits class probabilities, normalized score, and confidence labelled "max predicted class probability (NOT calibrated)" | **Verified in source + tests** |
| Syntax and local tests | `compileall` and `unittest` pass locally (torch-dependent tests skip cleanly) | **Verified** |
| Documentation set | 13-section final report, experiment plan/results, model card, Kaggle guide, integration guide, audit docs | **Verified** |

## Category 2 — Tested with Kaggle results (really measured, historical)

These numbers predate this audit and are **preserved, not reproduced**. They are
the only performance claims that may be stated as measured.

| Run | Split | Accuracy | Macro F1 | Notes |
|---|---|---:|---:|---|
| Original baseline (`v1_legacy`) | test | 0.4065 | 0.4057 | Different label config; not a controlled comparison |
| Improved incumbent (`v2_log_shrunk`) | validation | 0.4589 | 0.4521 | Best epoch 15 |
| Improved incumbent (`v2_log_shrunk`) | test | 0.4384 | 0.4309 | Precision 0.4323, recall 0.4362, weighted F1 0.4327 |
| Frozen ImageNet probe | validation | 0.4580 | 0.4561 | Diagnostic only, not a CNN result |

Incumbent per-class test F1: low `0.4684`, average `0.3293`, high `0.4948`.
The probe-vs-CNN parity is a **hypothesis-generating observation**, not proof of
a feature ceiling.

## Category 3 — Implemented but not executed (no result may be claimed)

| Item | What exists | Why it is not a result yet |
|---|---|---|
| 11 controlled experiments (`e0`–`e10`) | config + runner + isolated dirs | No Kaggle GPU run occurred locally |
| Frozen probe re-run | `src/probe.py` | Must run on Kaggle with the new split |
| Leakage audit | `src/data_audit.py` | Needs the real dataset and saved split |
| Winner's one-time test evaluation | `src/evaluate.py` | Must run **once**, after selection is locked |
| 5-fold OOF training and scoring | `make_folds.py`, `oof_predict.py` | Needs 5 Kaggle fold checkpoints |
| Live real-image demonstration | `src/demonstrate.py` | Needs the checkpoint and dataset; HTML/CSV not generated |
| Winner selection / possible checkpoint replacement | documented rule | Impossible before validation results exist |

All such values are marked `PENDING` in `docs/EXPERIMENT_RESULTS.md` and
`docs/FINAL_MEMBER_A_REPORT.md`. A smoke test with `--max_images` is never an
experiment.

## Category 4 — Remaining limitations (open, not fixed by this work)

1. **No local runtime evidence.** torch/torchvision, the dataset and all
   checkpoints are absent locally. Torch-dependent tests were skipped locally;
   they must be run on Kaggle (`python -m unittest discover -s tests -v`).
2. **Labels are market proxies.** Price-per-sqft tertiles inherit location,
   demand, timing and property confounds. There are no human condition labels.
3. **The weak `average` class is real** (test F1 0.3293) and may be inherent to
   a heterogeneous middle tertile; experiments test, but do not assume, a fix.
4. **Leakage is unmeasured.** Duplicate photos or same-street listings may span
   splits, so held-out numbers may be optimistic until `data_audit.py` runs.
5. **No calibration or uncertainty.** `confidence` is max soft probability, not
   a calibrated confidence; no conformal or temperature-scaling claim is made.
6. **Single-split, no significance testing.** Small validation differences must
   be treated cautiously; per-class support must accompany any F1 comparison.
7. **OOF cost.** Five fold trainings are required before Member B can use honest
   training-row scores; until then, in-sample leakage remains a live risk.
8. **Deliberate deferrals.** Regression head, validation-tuned thresholds and
   group-aware folds are future work, not hidden experiments.

---

## Final verdict

**The contribution is academically ready to *run and defend*, not yet ready to
claim new results.** The methodology, controls, isolation checks and reporting
structure are verified in code and local tests; the only measured performance is
the preserved historical set in Category 2. Category 3 items are executable but
unexecuted, and Category 4 limitations must be stated plainly to the examiner.
The incumbent checkpoint stays the default until Kaggle validation evidence
justifies a change.
