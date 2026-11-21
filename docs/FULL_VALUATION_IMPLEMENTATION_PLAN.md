# Full valuation pipeline — implementation plan (PENDING)

**Status: PLAN ONLY.** No structural model, price estimator, integration, or
Member B experiment has been implemented in this repository. The exact original
valuation prompt is not present in the repository or its git history, so this
document describes the required architecture without pretending to reproduce
verbatim wording.

## Scope boundary

| Component | Owner | Status |
|---|---|---|
| Image-side visual-proxy CNN, `VisionAgent`, OOF row contract | Member A | **Implemented and finalized** (`e0_reference_group_split`) |
| Group-aware OOF visual scores for Member B | Member B | **Pending** — no OOF CNN trained, no CSV generated |
| Structural-feature model (bed/bath/sqft/city and related attributes) | Member B | **Pending** |
| Final price estimator combining structural and image-derived features | Member B | **Pending** |
| End-to-end evaluation and price-error reporting | Member B | **Pending** |

The CNN is a **market-derived visual proxy**, not verified physical condition and
not a price estimate. Its `condition_score`/probabilities are *features* for a
later estimator; they are never the final prediction.

## Required end-to-end architecture

```
Exterior image ──> [Member A CNN, OOF-only features] ──┐
                                                       ├──> [Member B structural model
CSV structural attributes ──────────────────────────────┘      + visual features]
                                                                  │
                                                                  v
                                                  [final price estimator]
                                                  price (EUR/USD as in CSV)
```

### 1. Data and splits (pending work)

- Use the dataset and column contract documented in `docs/ACADEMIC_REPORT.md`.
- Member B defines its **outer** train/validation/test split with the same
  property-group discipline as Member A (normalised address + exact image hash;
  never split a group across partitions). The final e0 checkpoint's train ids
  are a row-level subset of the full dataset; they are **not** a valid outer-test
  boundary.
- Every feature fitted on data (imputers, encoders, scalers, target transforms,
  CNN statistics) must be fitted on the relevant training partition only.

### 2. Visual features (pending work)

- Generate OOF probability vectors for **training rows only** using group-aware
  folds restricted to outer-train data (`src/oof_contract.py` defines
  `OOF_COLUMNS`; `src/train.py --split_file` refits label statistics per fold).
- Train **fold-specific CNNs**; each held-out row must be scored by a checkpoint
  that never saw that row. The existing row-level `src/make_folds.py` output and
  any older `oof_visual_scores.csv` are unsafe and must be regenerated.
- **Never** use the final `e0_reference_group_split` checkpoint to score outer-test
  rows that overlap its training data. Score outer-test only with a CNN trained
  without those rows.
- Merge `condition_score`, `p_low`, `p_average`, `p_high`, `label`, `confidence`
  and provenance columns into Member B's training frame by `image_id` (see
  `docs/MEMBER_A_TECHNICAL_HANDOFF.md`).

### 3. Structural model and final estimator (pending work)

- Build structural features from `bed`, `bath`, `sqft`, `citi`/city statistics and
  any other documented attributes; keep preprocessing scikit-learn pipelines
  inside the training fold.
- Train the structural model and a separate final price estimator that consumes
  both structural features and the OOF visual features. Do not tune or select on
  the test set.
- Compare against baselines (structural-only, visual-only, combined) using
  validation metrics appropriate to regression (MAE/RMSE/R² and subgroup
  errors); report per-group and per-city behaviour where support permits.

### 4. Evaluation discipline

- Member A's e0 test split was evaluated **once** and is sealed. Member B must
  define and seal its own outer test **before** model selection, evaluate it
  **once**, and never reuse it for tuning.
- Keep the proxy-class metrics (macro F1) separate from price-error metrics; a
  better proxy F1 is not proof of better price prediction.
- Record exact commits, split manifests, checkpoint SHAs, commands, metrics and
  timing for every run, as Member A did in `docs/MEMBER_A_KAGGLE_READINESS.md`.

## Deliverables still required

1. Group-aware OOF CSV with full probability vector and fold provenance.
2. Structural-feature model artifacts and validation results.
3. Final price-estimator artifacts, one-time outer-test results, and a report
   that distinguishes proxy classification from price estimation.
4. End-to-end reproduction instructions and an honest limitations section.

None of these exist yet; this file is a specification, not evidence of work.
