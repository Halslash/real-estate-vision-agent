# Member A academic report — market-derived visual proxy CNN

**Status: FINAL (Member A only).** This is the main report for examination. The
final run is `e0_reference_group_split` at pinned source commit
`bc39d102d2e9cde83144966d6e6c910926109429`. It was trained and evaluated once
on a property-disjoint split; the test split is sealed. Member B (structural
model and final price estimator) is **not implemented**.

> **Interpretation warning.** The CNN estimates a **market-derived visual
> proxy**: a tertile of log price-per-square-foot relative to a shrunken city
> median. It does **not** verify physical condition or aesthetics, and it is
> **not** a price estimate. A higher `condition_score` does not prove a better
> maintained property.

---

## 1. Problem and objective

The project (Projet 10, *Évaluation Immobilière Multi-Modale*) asks whether an
exterior-property photograph adds useful signal to a conventional structural
price model. The full objective is a multimodal valuation pipeline: image-derived
features plus structural attributes feeding a final price estimator. This
repository delivers only the image branch.

## 2. Member A's responsibility

Member A provides a reproducible transfer-learning classifier over exterior
photos, a stable inference interface (`VisionAgent`), an out-of-fold (OOF)
probability-row contract, and the documentation needed to integrate the visual
feature safely. Member A does **not** implement a Random Forest, a final Linear
Regression, an end-to-end price predictor, or any Member B experiment. Those
components are specified as pending in
`docs/FULL_VALUATION_IMPLEMENTATION_PLAN.md`.

## 3. Data and label construction

- Dataset: `socal2.csv` (~15,474 rows) with exterior images named
  `<image_id>.jpg`, on Kaggle under
  `/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/`.
- Columns are detected case-insensitively (`image_id`, `street`, `citi`, `bed`,
  `bath`, `sqft`, `price`); images are decoded lazily and converted to RGB.
- **Label versions.** `v1_legacy` clips raw `price/sqft` to train-only 1st/99th
  percentiles and tertiles the ratio against the raw city median. The default
  `v2_log_shrunk` clips `log(price/sqft)`, shrinks each city median toward the
  global median with `k = 10` pseudo-counts
  (`m̂_c = (n_c·m_c + k·m_global)/(n_c + k)`), and tertiles the observed distinct
  training values of `clipped_log_pps − m̂_c`. `v3_balanced_quantile` replaces
  only the two cut rules with sample-weighted tertiles and was configured but
  never trained.
- All clipping bounds, medians, shrinkage and thresholds are fitted on the
  **training partition only** and stored in the checkpoint; validation and test
  rows never influence label statistics.

The three classes are `low`, `average`, `high` **proxy tertiles**, not human
condition grades.

## 4. Leakage-resistant split

The final model uses `property_address_plus_exact_hash` (seed 42): grouped by
normalized full street address and byte-exact image SHA-256, with near-duplicate
dHash grouping disabled. Counts are **10,829 train / 2,337 validation / 2,308
test**. A local, read-only preflight returned `status=PASS`, with all validity
checks true: every valid id appears once, and no address or exact-hash group
crosses splits. This local evidence is recorded in the pinned Kaggle preflight
cell in `docs/MEMBER_A_KAGGLE_READINESS.md` and in the cached
`verification.json`; the final training cell re-verified the same artifacts on
Kaggle. Perceptual near-duplicates at different addresses remain untested.

The historical random split allowed the same addresses and byte-identical images
into multiple partitions, so its held-out scores were optimistic. Historical and
final numbers are therefore **not directly comparable**.

## 5. CNN architecture and training

Image → RGB → `Resize(256)` → `CenterCrop(224)` → ImageNet normalization →
ImageNet-pretrained MobileNetV2 → dropout → 3-logit head → softmax →
`condition_score = 0.5·P(average) + 1.0·P(high)`, clipped to `[0, 1]`.

Two-phase training (`src/train.py`): frozen backbone/head-only feature extraction
then fine-tuning of the last 4 blocks at a lower learning rate. Selected
configuration: **30 epochs total** (not 20), batch size 64, AdamW with
`lr = 1e-4` / `lr_finetune = 1e-5`, cosine schedule, cross-entropy with auto
class weights (effective-number β = 0.9999 rule), label smoothing `0.05`,
`improved` augmentation, early-stopping patience 5 (not triggered), seed 42 with
deterministic cuDNN. Model selection used **validation Macro F1 over fixed labels
`[0, 1, 2]`** only. Best epoch **27**; reported training duration 1,446.1 s,
peak VRAM 717.5 MB on CUDA (from the Kaggle `training_summary.json`, not locally
re-measured).

## 6. Experimental comparison

| Run | Label config | Loss / weights | Validation Macro F1 | Checkpoint |
|---|---|---|---:|---|
| **`e0_reference_group_split` (final)** | `v2_log_shrunk`, K=10 | CE + auto β=.9999, ls 0.05 | **0.4314855** | reported at `/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth` |
| `e11_v3_quantile_labels` | `v3_balanced_quantile` | CE + auto, ls 0.05 | **not run — no checkpoint** | absent |
| `e12_v3_balanced_ce` | `v3_balanced_quantile` | CE + balanced, ls 0.05 | **not run — no checkpoint** | absent |

All three use MobileNetV2, 224 px crop / 256 resize, 30 epochs, seed 42 and the
same property-disjoint protocol in `configs/experiments.json`; e11 and e12 are
configured and locally dry-run checked, but no Kaggle run, validation metric, or
checkpoint exists for them. Experiments `e1`–`e10` were also not run. The
historical random-split incumbent's validation F1 of `0.4521` is preserved in
`docs/EXPERIMENT_RESULTS.md` but is not comparable with the final run. **The
selected model is `e0_reference_group_split`**: it is the only run with measured
validation evidence under the leakage-resistant protocol, so e11/e12 cannot be
selected, and no test result was used for selection.

## 7. Final results

One-time test evaluation on **2,308** samples (reported from the sealed Kaggle
run; not re-executed locally):

| Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.4281 | 0.4277 | 0.4296 | 0.4240 | 0.4233 |

Per-class test F1: low **0.4582**, average **0.3539**, high **0.4600**;
average-class recall **0.3070**. The disparity between final and historical
numbers reflects the change in split protocol, not a demonstrated regression.
The test split was evaluated **once** and sealed in `results/evaluation_gate.json`;
it must never be reused for model selection or tuning. An additional e11 test
evaluation would be a *subsequent* look at an already-inspected test set and is
not scientifically necessary for this report; it was not performed.

## 8. Inference interface

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="path/to/best_model.pth")
result = agent.score("house.jpg")          # or agent.score_batch(paths, batch_size=64)
print(result)
```

The result is a dict with `condition_score`, `label`, `probabilities`
(`low`/`average`/`high`), `confidence`, `model` and `score_type`. Inference needs
only the self-contained checkpoint: no CSV, dataset, OOF file, or pretrained
weight download. `confidence` is the maximum class probability and is **not**
calibrated. The full OOF row contract (`image_id`, `fold`, `condition_score`,
`p_low`, `p_average`, `p_high`, `label`, `confidence`, `model`, `checkpoint`) and
the fold/outer-test rules are in `docs/MEMBER_A_TECHNICAL_HANDOFF.md`.

## 9. Limitations

1. Labels are market proxies inherited from prices; location, demand and timing
   confounds are unavoidable, and there are no human condition annotations.
2. The `average` class is weak (test F1 `0.3539`, recall `0.3070`) — expected for
   a heterogeneous middle tertile, but a real limitation.
3. `condition_score` is a heuristic soft feature, not a probability of a
   real-world event; `confidence` is not calibrated; no uncertainty bounds or
   threshold tuning are claimed.
4. Near-duplicate perceptual images at different addresses were not grouped;
   single-split results without significance testing should be read cautiously.
5. The test split has been inspected once and is sealed; no further test-based
   selection, tuning, or unreported re-evaluation is permitted.
6. Member B is pending; nothing here demonstrates an end-to-end price estimate.
7. Final result artifacts are stored on Kaggle; only local, structural evidence
   (configs, code, preflight report) could be verified in this repository.

## 10. Future integration requirements

Member B must generate **group-aware OOF folds restricted to its outer training
data**, train fold-specific CNNs that never see their held-out rows, and score
outer-test rows with a CNN trained without those rows. The final
`e0_reference_group_split` checkpoint is explicitly **prohibited** from scoring
outer-test rows that overlap its own training data. Structural features and the
visual proxy/OOF probabilities should then feed a separately evaluated price
estimator; see `docs/FULL_VALUATION_IMPLEMENTATION_PLAN.md` and
`docs/MEMBER_A_TECHNICAL_HANDOFF.md`.
