# Leakage-resistant split — audit, design and verification (Member A)

**Status: audit and split design only. No model was trained and the test split was
not evaluated through a model.** All numbers below come from the public Kaggle
dataset reconstructed locally (same 15,474 rows as the completed label audit);
the same commands are ready to run on the Kaggle mount.

---

## 1. Available grouping identifiers (actual data, 15,474 rows)

| Field | Missing/blank | Distinct | Assessment |
|---|---:|---:|---|
| `image_id` | 0 | 15,474 | **Unique per row, not per property.** 1:1 with the image filename; cannot be used as a property key. |
| `street` | 0 | 12,401 raw / **12,376 normalized** | **Full street address** (house number + street): 15,437/15,474 values (99.8%) start with a number. This is the only usable property identifier. |
| `citi` / `n_citi` | 0 | 415 | City. Too coarse: grouping by city would merge hundreds of unrelated properties; 339 cities legitimately appear in multiple splits. |
| `bed`, `bath`, `sqft`, `price` | 0 | 12 / 32 / 3,571 / 2,320 | Listing attributes, not identifiers; useful only as supporting evidence. |

Evidence that `street` is a property key with multiple images per property:

- 3,008 normalized addresses cover 6,106 rows; 2,958 of those groups (98.3%)
  have exactly the same `price` **and** `sqft`.
- Of 2,986 byte-identical image groups, **99.1% share one normalized address**,
  99.2% share one price, 99.6% share one sqft, 98.9% share all metadata.
  The duplicates are overwhelmingly repeated photos of the same listing.
- Address formatting is inconsistent but reconcilable: 24 raw strings collapse
  to a normalized key (case, `Ave`/`Avenue`, `St`/`Street`, punctuation). Only
  three addresses appear in more than one city; two are visibly the same
  property listed across a city boundary and one is the sentinel below.
- **Unusable identifiers:** 23 rows have `street == "Address not provided"`.
  They are not merged into a single fake property. They start as singletons and
  are grouped only if their image bytes or dHash prove a relationship.

**Conclusion:** a reliable property identifier exists (normalized full address),
but it must be augmented with image evidence — 27 byte-identical groups (79 rows)
span different addresses, i.e. different properties photographed with the same
stock image.

## 2. Chosen strategy and why

`src/group_split.py` (new, opt-in) builds **connected components** over three
edge types, so a group can never cross a split:

1. same normalized full address (case/punctuation/abbreviation canonicalized);
2. identical image SHA-256 (guarantees zero exact-duplicate leakage);
3. near-duplicate dHash (64-bit, Hamming ≤ 5), searched **globally**, not only
   within one street string as the existing audit does.

It then performs group-level 70/15/15 splitting with the same temporary
price-per-sqft tertile stratification and the same two-step `70/30 -> 50/50`
`train_test_split` recipe as `dataset.make_stratified_split`. No proxy label is
fitted during splitting; group strata use only the pre-existing temporary pps
bins. Seed 42, output confined to `/kaggle/working/output/leakage_resistant_split`.

**Deliberately not done:** no city-level grouping (would merge unrelated
properties), no model/label/preprocessing/optimizer/hyperparameter changes, no
change to `dataset.py`, and no change to `train.py`'s training behaviour. The
only edit to `train.py` is a fail-closed guard on `--split_file` that rejects
leaking, duplicate or incomplete id lists. The random split remains the default;
the new split is consumed through the already-supported `train.py --split_file`.
No existing experiment, checkpoint or historical result was overwritten.

## 3. Generated split

| Split | Records | Share |
|---|---:|---:|
| train | 10,847 | 70.10% |
| val | 2,281 | 14.74% |
| test | 2,346 | 15.16% |
| **total** | **15,474** | 100% |

Groups: **12,310**. Size distribution: 1 → 9,281; 2 → 2,974; 3 → 24;
4 → 17; 5 → 7; 6 → 1; 7 → 3; 9 → 1; 10 → 1; 24 → 1 (largest group is the
sentinel-related component, 0.16% of rows).

Class counts and proportions, with tertile thresholds fitted on the new TRAIN
split only (post-hoc check, never used to build the split):

| Version | train | val | test |
|---|---|---|---|
| v2_log_shrunk | [3680, 3581, 3586] (33.9/33.0/33.1%) | [814, 689, 778] (35.7/30.2/34.1%) | [829, 745, 772] (35.3/31.8/32.9%) |
| v3_balanced_quantile | [3616, 3614, 3617] (33.3/33.3/33.3%) | [805, 696, 780] (35.3/30.5/34.2%) | [811, 754, 781] (34.6/32.1/33.3%) |

All three classes are present in every split and no class exceeds 36% of any
split. Class balance is therefore reasonable, not exact.

## 4. Leakage audit of the proposed split

`src/data_audit.py` ran on **all 15,474 images**, no `--max_images`:

| Check | Random seed-42 split | Proposed group split |
|---|---:|---:|
| Missing image files | 0 | 0 |
| Exact-duplicate groups across splits | 1,384 | **0** |
| Near-duplicate pairs across splits (same street, dHash ≤ 5) | 1,391 | **0** |
| Street groups spanning splits | 1,399 | 1 (the literal `"Address not provided"` sentinel) |
| City groups spanning splits | 356 | 339 (expected; city is not a leakage unit here) |
| Val rows byte-identical to a train image | 614 (26.5%) | **0** |
| Test rows byte-identical to a train image | 643 (27.7%) | **0** |

Additional checks on the new split:

- Groups spanning more than one split: **0**; normalized addresses spanning
  more than one split: **0** (from `group_manifest.json`).
- Independent global dHash scan across **all** addresses (not just same-street):
  **0** cross-split pairs at Hamming ≤ 5.
- Unassignable/sentinel addresses: 23 rows (train 14, val 5, test 4); none is
  byte-identical or near-duplicate to a row in another split.
- No records were skipped; all 15,474 IDs are covered exactly once.

## 5. Limitations

- Near-duplicate grouping uses a fixed Hamming ≤ 5 threshold on a 64-bit dHash;
  visually similar photos with a larger distance are not caught. The global scan
  is stronger than the shipped same-street audit but is still a hash heuristic.
- Connected components are transitive: one borderline near-duplicate edge can
  merge several properties. Here the effect is tiny (43 extra edges, largest
  component 24 rows), but it is a real trade-off.
- Group-level stratification uses raw price-per-sqft tertiles, not the final
  train-fitted proxy labels; class proportions are good but not exactly equal.
- The one spanning "street" group is the literal sentinel string, not a shared
  address; city overlap remains descriptive and expected under random group
  assignment.
- This is a local reconstruction of the public dataset. The split must be
  regenerated on the Kaggle mount (deterministic with seed 42) before being
  treated as the canonical artifact.

## 6. Reproduction commands (Kaggle, no training)

```python
!python src/group_split.py \
    --output_dir /kaggle/working/output/leakage_resistant_split --seed 42
!python src/data_audit.py \
    --split_file /kaggle/working/output/leakage_resistant_split/split_indices.json \
    --output /kaggle/working/output/leakage_resistant_split/data_audit.json
!python -m unittest discover -s tests -v
```

The generated `split_indices.json` is consumed by the existing
`train.py --split_file` mechanism (hardened to reject leaking, duplicate or
incomplete id lists); architecture, labels, preprocessing, loss, optimizer,
epochs and all other settings stay identical to the seed-42 recipe.

## 7. Verdict and what must happen before retraining

**Yes — this dataset can support a credible evaluation, but only with the
group split (or an equivalent property-disjoint split).** The old random split
cannot: ~27% of test and ~26.5% of validation rows were byte-identical to a
training image.

Before any retraining:

1. Regenerate `split_indices.json` on Kaggle and re-run the leakage audit there;
   require 0 cross-split exact duplicates and 0 cross-split near-duplicates.
2. Train **from scratch** with `--split_file .../leakage_resistant_split/split_indices.json`
   into a **new** output directory. Never reuse the old checkpoints or split.
3. Recompute all preprocessing statistics and proxy-label thresholds on the new
   training split only (the code already does this). Do not copy the old
   `class_thresholds.json` / `city_medians.json`.
4. Re-run the seed-stability replicates (43 and 123) on the new split before
   comparing label variants. Include `--epochs 30` explicitly — `train.py`'s
   parser default is 20 and would silently change the protocol.
5. Treat the old ~0.46 validation Macro F1 as **invalid** for generalization
   claims. Do not compare new results to it as if it were a like-for-like
   baseline, and do not evaluate the test split until selection on validation
   is complete.

No test evaluation, model selection on test, or significance claim is made here.

Skipped/not checked: no model training, no GPU work, no test-set inference; the
full suite has one **pre-existing, unrelated** failure in
`tests/test_vision_agent.py::test_invalid_counts_are_rejected` (a list passed to
`compute_class_weights` which calls `.tolist()`), visible only when torch is
installed; it was left untouched. Risk: the group split is only as good as the
address field — if Kaggle's mounted CSV differs from the public copy, regenerate
and re-audit before use.
