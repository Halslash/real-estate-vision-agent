# Leakage-resistant split — simplified workflow (Member A)

**Status: split generation and read-only audit only. No model was trained and the
test split was not evaluated through a model.** Numbers below come from the
public Kaggle dataset reconstructed locally (the same 15,474 valid rows as the
label audit). Regenerate on the Kaggle mount with the single cell in
`docs/MEMBER_A_KAGGLE_READINESS.md`.

## 1. Why the random split is not acceptable

On the seed-42 random split, **614 validation rows (26.5%) and 643 test rows
(27.7%) were byte-identical to a training image**, and 1,384 exact-duplicate
groups crossed splits. Validation and test estimates are therefore optimistic.
The old ~0.46 validation Macro F1 must not be presented as generalization
evidence or compared with results from a clean split.

## 2. What the default workflow does

`src/group_split.py` is opt-in; the random split remains `train.py`'s default.
It builds connected components over **two cheap evidence types**:

1. the **normalized full street address** (`street` is a full address such as
   `1317 Van Buren Avenue`, not just a street name), canonicalized for case,
   punctuation and common abbreviations;
2. **byte-identical image files** (SHA-256) — repeated photos of the same
   listing, including stock images reused at different addresses.

Whole groups are then assigned to train/val/test (70/15/15) with the project's
existing price-per-sqft tertile stratification and two-step `70/30 -> 50/50`
recipe, seed 42. No proxy label is fitted during splitting. `train.py` consumes
the result through the existing `--split_file` flag; architecture, labels,
preprocessing, loss, optimizer and all hyperparameters are unchanged.

`--near_duplicates` is an **optional, OFF-by-default** flag that adds global
64-bit dHash edges (Hamming <= 5). It was the Kaggle bottleneck (all-pairs
comparison) and is not a prerequisite for a defensible split. The separate
`src/data_audit.py` near-duplicate check is a cheap **same-street** diagnostic;
it is not a global guarantee. Neither is run by the default split command.

## 3. Local verification of the simplified split (seed 42)

| Split | Records | Share |
|---|---:|---:|
| train | 10,829 | 70.0% |
| val | 2,337 | 15.1% |
| test | 2,308 | 14.9% |
| **total** | **15,474** | **100%** |

Groups: **12,353** (address edges 3,076; exact-hash edges 45; dHash edges 0 —
option off). Largest group: 10 rows. Group-size distribution: 1 → 9,326;
2 → 2,981; 3 → 21; 4 → 15; 5 → 5; 7 → 4; 10 → 1.

`src/data_audit.py` on all 15,474 images:

| Check | Result |
|---|---:|
| Missing image files | 0 |
| Exact-duplicate groups across splits | **0** |
| Exact-duplicate rows in val/test matching train | **0** |
| Same-street near-duplicate pairs across splits (diagnostic only) | 0 |
| Raw `street` strings spanning splits | 1 — the literal `"Address not provided"` sentinel (train 18, val 4, test 1) |
| City groups spanning splits | 332 (expected; city is not a leakage unit) |

**Placeholder addresses:** 23 rows have an unusable address (23 sentinels).
Their ids are written explicitly to `group_manifest.json` under
`missing_address_ids`. They are not merged into one fake property; they start
as singletons and are grouped only when their image bytes are identical.

Class counts, with tertile thresholds fitted on the **training split only**
(a post-hoc check, never used to build the split):

| Label version | train | val | test |
|---|---|---|---|
| v2_log_shrunk | [3691, 3563, 3575] | [817, 770, 750] | [774, 785, 749] |

All three classes are present in every split. CNN labels are **market-derived
visual-quality proxies** (price-per-sqft tertiles relative to a shrunken city
median), **not** verified physical-condition grades.

## 4. What the split guarantees

- No normalized address group crosses train/val/test.
- No byte-identical image group crosses train/val/test.
- Every valid record appears exactly once. `train.py`'s
  `validate_supplied_split()` rejects unknown, duplicate, overlapping or omitted
  ids before training starts.
- Deterministic with seed 42; the random/default path is untouched.
- Preprocessing statistics and label thresholds are fitted on training data
  only; the test split is not used for fitting or model selection.

## 5. What it does not guarantee

- **Visually similar photos at different addresses are not exhaustively
  excluded.** Global dHash detection is off by default; the audit's
  near-duplicate check only compares images on the same raw street string. Do
  not claim near-duplicate leakage is eliminated — claim only that no
  *identical* image and no *same-address* image crosses splits.
- Placeholder addresses (23 rows) cannot serve as property keys and are
  reported, not silently merged.
- City overlap is descriptive and expected; city is too coarse to be a
  property identifier.
- Group stratification uses raw price-per-sqft tertiles rather than the final
  train-fitted proxy labels, so class proportions are reasonable, not exact.
- This is a local reconstruction of the public dataset. Regenerate and re-audit
  on Kaggle before treating the split as the canonical artifact.

## 6. Reproduction and verdict

One Kaggle cell (generate → audit → verify → save, no training) is in
`docs/MEMBER_A_KAGGLE_READINESS.md`. For the later training step, pass
`--split_file .../leakage_resistant_split/split_indices.json` to `src/train.py`
with a **new** output directory and the original hyperparameters (including
`--epochs 30`, since the parser default is 20). Do not reuse old checkpoints or
the old split, and do not evaluate the test split until selection on validation
is complete.

**Verdict:** this simplified split is sufficient for a credible student-project
evaluation because it removes the dominant leakage source (repeated photos of
the same property) with a cheap, reproducible, documented step. The random
split must not be reported as reliable generalization evidence.

Skipped/not checked: no model training, no GPU work, no test-set inference. The
full suite has one **pre-existing, unrelated** failure in
`tests/test_vision_agent.py::test_invalid_counts_are_rejected` (a list passed to
`compute_class_weights`, which calls `.tolist()`); it was left untouched.
