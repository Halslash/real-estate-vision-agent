# CNN Performance Audit & Improvement Plan

**Project:** Member A — Real Estate Vision Agent (MobileNetV2 visual-quality proxy)
**Scope:** Static audit of `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/vision_agent.py`, `README.md`
**Date:** 2026-11-20
**Constraints observed:** No Python execution, no tests, no package installs, no Kaggle access, no dataset or weight downloads, no project-file modifications, no checkpoint changes. No evaluation results have been measured by this audit; all test-side numbers referenced below are *not* available.

**Evidence labels used throughout:**
- **[OK]** — verified correct in the source.
- **[CONFIRMED]** — a factual property of the code or of the user-provided training log.
- **[HYPOTHESIS]** — plausible but unmeasured; requires a Kaggle experiment.

---

## 1. Executive summary

The implementation is **technically sound**. Architecture, ImageNet normalization, train-only statistic fitting, split discipline, freeze/unfreeze logic, checkpoint selection, evaluation reconstruction and inference are all implemented correctly. No data leakage was found in label construction or split construction. The lazy-decoding change is a loading optimization and does not alter labels or splits.

The ~39.6% validation accuracy and 0.3961 macro F1 are **not explained by an implementation bug**. They are consistent with a **low-signal, noisy-label problem**: the target is a tertile of price-per-square-foot relative to city median, a quantity driven mostly by location, micro-market, lot, renovations and property attributes that are largely invisible in an exterior photograph. For a balanced 3-class task, chance accuracy is ~33.3%; the model finishes ~6.3 points above chance. That is a real but weak learned signal.

Two code-level weaknesses are worth correcting because they add avoidable label noise:
1. **[CONFIRMED]** city medians are computed with **no minimum group size and no shrinkage to the global median**, so cities with one or two training samples produce unstable ratio denominators;
2. **[CONFIRMED]** training augmentation (`RandomResizedCrop` scale 0.8–1.0 + 10° rotation) differs from evaluation preprocessing (`Resize(256)` + `CenterCrop(224)`), which is a train/inference distribution mismatch that may be unhelpful for upright architectural imagery.

The highest-value next step is **diagnosis, not tuning**: add predicted-class distribution, train-vs-validation gap, learning curves and a frozen-feature linear probe. Without those, underfitting and label noise cannot be distinguished. After that, improve label quality (log-ratio + shrunken city medians), then add a learning-rate schedule and longer fine-tuning, then compare architectures.

The model must continue to be described as a **market-derived visual proxy**. It is not a validated physical-condition or aesthetic-quality classifier, and a higher score must not be interpreted as "the house is in objectively better condition."

---

## 2. Phase 1 — Existing implementation review

### 2.1 Summary table

| # | Audit question | Finding | Status |
|---|---|---|---|
| 1 | Architecture / pretrained weights | `MobileNet_V2_Weights.IMAGENET1K_V1`, head `Dropout(0.2) → Linear(1280, 3)`; correct in-feature inference | [OK] |
| 2 | ImageNet normalization | Standard mean/std, used in train, val, test and inference from saved config | [OK] |
| 3 | Preprocessing / augmentation | Mismatch between train crop/rotation and eval center crop; rotation questionable for houses | [CONFIRMED] difference; [HYPOTHESIS] harm |
| 4 | Proxy-label math | `clip pps → pps / city median → tertile of train ratios`; applied unchanged to val/test | [OK] |
| 5 | City-relative labels | No min-count or shrinkage on city medians → noisy denominators for small cities | [CONFIRMED] code property; [HYPOTHESIS] impact |
| 6 | Class separation | Tertiles separate the *ratio* by construction; visual separability unmeasured | [CONFIRMED] construction; [HYPOTHESIS] visual |
| 7 | Split leakage | Split by `image_id`, duplicates removed, stats fitted on train only; no leakage found | [OK] |
| 8 | Schedule / optimizer / weight decay | AdamW 1e-4 head, 1e-5 fine-tune, no LR scheduler, 5+5 epochs; class weighting effectively inactive | [CONFIRMED] |
| 9 | Freeze / unfreeze | Correct `requires_grad` handling; correct optimizer param selection | [OK] |
| 10 | Checkpoint selection | Best validation macro F1 saved; epoch 9 selected | [OK] |
| 11 | Useful features vs priors | Macro F1 above chance suggests some discrimination, but priors not ruled out | [HYPOTHESIS] |

### 2.2 Architecture and pretrained weights — [OK]

`src/model.py:19–52` builds `models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)`, reads `model.classifier[1].in_features` (1280 for MobileNetV2) and replaces the classifier with `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 3))`. This is the standard, correct transfer-learning pattern. Pretrained-weight failure raises explicitly instead of silently using random weights. `build_model_from_checkpoint()` reconstructs the same architecture with `pretrained=False` and loads the saved state dict, so inference requires no download.

Note (minor, not a defect): `IMAGENET1K_V2` weights are marginally stronger than `V1` for a similar cost, but this is not a plausible explanation for a ~6-point-over-chance result.

### 2.3 Normalization — [OK]

`src/dataset.py:40–41` defines the canonical ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Both `build_train_transform()` and `build_eval_transform()` apply them, and the saved `preprocessing` dictionary (with `image_size`, `resize_size`, `mean`, `std`) is used by `evaluate.py` and `VisionAgent` (`src/vision_agent.py:50`). Normalization is consistent end to end.

### 2.4 Preprocessing and augmentation — [CONFIRMED difference, harm unproven]

Training (`src/dataset.py:478–487`): `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(±10°)`, `ColorJitter(0.2, 0.2, 0.2, 0.05)`.
Evaluation/inference (`src/dataset.py:490–502`): `Resize(256)`, `CenterCrop(224)`.

- **[CONFIRMED]** The spatial transforms differ: training samples can be cropped anywhere from 80% to 100% of the image and rotated; evaluation always takes a fixed center crop with no rotation. This is a common torchvision default, not a correctness error, but it is a distribution shift.
- **[HYPOTHESIS]** ±10° rotation is questionable for exterior property photos, where upright orientation is semantically meaningful; it creates unrealistic tilted houses and may add label-irrelevant variance.
- **[HYPOTHESIS]** `scale=(0.8, 1.0)` can crop off roofline, façade edge or yard — potentially discarding the very cues the label is supposed to reflect.
- **[OK]** Horizontal flip and mild color jitter are reasonable for this domain. Grayscale/RGBA/palette inputs are converted to RGB at decode time.

### 2.5 Proxy-label generation — [OK]

`load_and_validate_records()` computes `price_per_sqft = price / sqft` and rejects non-positive/non-finite values and duplicate IDs. `fit_preprocessing_stats()` is called **only on `train_records`** (`src/train.py:174–181`) and computes:
- `clip_p1`, `clip_p99` from training pps;
- clipped pps and a global median;
- per-city medians of clipped pps;
- `ratio = clipped_pps / city_median` (global median fallback for missing/unseen cities);
- tertile thresholds from the sorted **distinct** training ratios (`src/dataset.py:388–401`).

`compute_labels()` applies the *saved* thresholds to train, val and test identically. No statistic is refitted on validation or test. This is mathematically coherent and leakage-free.

One subtlety worth recording: stratification bins in `make_stratified_split()` use `pd.qcut` over the **full** dataset's pps (`src/dataset.py:283–285`). This uses the label-defining quantity of all rows to balance the split. That is ordinary stratified splitting (which always uses all labels) and does **not** leak into model fitting or threshold estimation. **[OK]**

### 2.6 City normalization — [CONFIRMED code property]

`src/dataset.py:374–379` computes `city_medians = {city: median(values) ...}` with **no minimum sample count and no shrinkage**. Consequences:
- A city with 1 training property gets a median equal to that property's own pps, so its ratio is exactly `1.0` regardless of the image.
- A city with 2 properties yields two ratios symmetric around 1.
- Unseen cities fall back to the global median, which is reasonable.

**[HYPOTHESIS]** In a 15,474-row SoCal dataset with a long tail of small cities, this injects systematic noise into labels for those rows, weakening the image–label correlation. Magnitude is unmeasured. The persisted `models/city_medians.json` plus the CSV allow measuring the city-count distribution on Kaggle without touching the model.

Also **[CONFIRMED]**: `CITY_ALIASES` lists `"city"` before `"citi"`, and the exact-match loop runs before fuzzy matching, so with the reported CSV the column `citi` is selected deterministically over `n_citi`. The `street` column exists in the CSV but is **not used** anywhere in the code — it is available for a group-aware leakage diagnostic.

### 2.7 Class separation and balance — [CONFIRMED construction]

Tertiles of a continuous ratio guarantee approximately equal class sizes in training, so the chance baseline is ~33.3% accuracy. The weighted-CrossEntropy branch (`src/train.py:224–231`) activates only if `max(weight)/min(weight) > 1.5`; with near-equal tertiles this should not trigger. **[CONFIRMED]** Class weighting is therefore very unlikely to be a factor — verify from the logged line `Using weighted CrossEntropyLoss: ...` and `model_config.json`.

Whether the three classes are *visually* separable is unknown. **[HYPOTHESIS]** They may not be: price-per-sqft residuals largely reflect non-visual factors.

### 2.8 Split integrity — [OK]

- Duplicate `image_id` rows are skipped, so one image cannot appear in two splits (`src/dataset.py:233, 242`).
- `make_stratified_split()` produces disjoint 70/15/15 id lists with an explicit overlap assertion (`src/dataset.py:318–325`), seeded with 42.
- `fit_preprocessing_stats()` sees only training rows.
- `validate_labeled_split()` fails loudly on empty splits or missing training classes.
- Evaluation reuses `split_indices.json` from the checkpoint's folder and the saved column names; it does not regenerate a split.

No leakage found. **[HYPOTHESIS]** A different kind of risk remains: random (non-grouped) splitting may place near-duplicate exterior photos from the same tract/street in both train and test. That would *inflate* held-out scores rather than explain a low one, but it should be checked because it affects how results are interpreted.

### 2.9 Training schedule, optimizer, weight decay — [CONFIRMED]

- `--epochs 10` is split as `phase1_epochs = 5`, `phase2_epochs = 5` (`src/train.py:239–240`). The user's epoch table is consistent: epochs 1–5 frozen head, 6–10 fine-tuning.
- Phase 1: AdamW on classifier only, `lr=1e-4`, `weight_decay=1e-4`.
- Phase 2: unfreeze last 4 blocks, AdamW on all trainable parameters, `lr=1e-5`, same weight decay. Early stopping with patience 5, but with only 5 fine-tune epochs it effectively cannot trigger before the end.
- **[CONFIRMED]** There is **no learning-rate scheduler** anywhere in `train.py`.
- **[OK]** Batch size 64, seed set for Python/NumPy/Torch/CUDA.
- **[HYPOTHESIS]** Five fine-tune epochs at 1e-5 may be too few/too conservative. The validation curve was still rising at epoch 9 (39.51 → 39.60) and dipped slightly at 10 (39.47), which is more suggestive of a plateau at low signal than of a confidently converged model.

### 2.10 Freeze / unfreeze — [OK]

`freeze_backbone()` disables gradients on `model.features`, enables them on the classifier. `unfreeze_last_blocks()` unfreezes the final `n_blocks` feature blocks (with `n_blocks=4` out of 19 `features` modules, starting at index 15) and keeps the classifier trainable. Phase 2's optimizer selects `[p for p in model.parameters() if p.requires_grad]`. This is correct.

### 2.11 Checkpoint selection — [OK]

`run_phase()` saves on strict improvement in validation macro F1 (`> best + 1e-6`) and writes `best_model.pth` with full metadata. Epoch 9 was selected, consistent with the reported peak. Selecting on macro F1 for a balanced 3-class problem is appropriate. **[HYPOTHESIS]** With ~2,321 validation samples, macro F1 has sampling noise; repeated seeds would be needed to know whether epoch 9 vs 8 is meaningful.

### 2.12 What is the model learning? — [HYPOTHESIS, needs diagnostics]

Macro F1 0.3961 is above the ~0.33 uniform-guess baseline and far above the ~0.17 macro F1 of a majority-class-only predictor on balanced classes. That indicates it is producing non-trivial predictions across all three classes rather than collapsing to a prior. However, this cannot be confirmed without the confusion matrix and predicted-class histogram from the held-out test set. **[HYPOTHESIS]** A model can achieve ~39% accuracy by predicting mostly the two largest classes; the normalized confusion matrix will settle this.

---

## 3. Why ~39.6%? Evidence-based explanation

### 3.1 Baselines

For a balanced 3-class problem:
- Random/uniform guessing: ~33.3% accuracy, ~0.333 macro F1.
- Always predicting the majority class: ~33.3% accuracy, ~0.167 macro F1.
- Observed best: **39.60% accuracy, 0.3961 macro F1** → **+6.3 accuracy points over chance**.

This is a weak but real signal. It is the expected shape of a result where the label contains only a small image-dependent component.

### 3.2 Curve analysis

| Phase | Epochs | Val accuracy trajectory | Interpretation |
|---|---|---|---|
| 1 — frozen head | 1–5 | 33.78 → 35.50 → 37.14 → 35.07 → 36.62 | Slow, noisy learning of the linear head on fixed ImageNet features |
| 2 — fine-tune | 6–10 | 37.66 → 38.95 → 39.51 → 39.60 → 39.47 | Steadier gains, then plateau |

**[CONFIRMED]** Both phases contributed roughly +2.8 accuracy points. **[CONFIRMED]** The curve flattens in the last three epochs. This is compatible with either (a) the model approaching the ceiling of information available in the labels/images, or (b) underfitting due to a short schedule. Current evidence cannot distinguish them — see §4.3.

### 3.3 Ranked explanation

1. **[HYPOTHESIS, highest confidence]** **Label signal is intrinsically weak.** Price-per-sqft relative to city median is dominated by within-city location, neighborhood, school district, lot, renovations and market timing. An exterior photo exposes only a fraction of these. The Bayes-optimal image-only classifier may itself be only modestly above chance. This is the most parsimonious explanation for a slow climb to ~39%.
2. **[CONFIRMED code property; impact hypothesized]** **Small-city median noise** adds label jitter (§2.6).
3. **[HYPOTHESIS]** **Short fine-tuning schedule** with no LR schedule (§2.9).
4. **[HYPOTHESIS]** **Augmentation mismatch / rotation** adds train-only noise (§2.4).
5. **[HYPOTHESIS]** **224px resolution** may be insufficient for subtle material/condition cues.
6. **[HYPOTHESIS]** **Non-grouped split** may contain near-duplicates, which would make test numbers optimistic rather than explain the low validation number.

Explicitly ruled out by evidence: wrong normalization, wrong classifier dimensions, broken freeze/unfreeze, a different split between train and evaluation, refitting statistics on validation/test, and class-weighting pathology.

---

## 4. Phase 2 — Evaluation and diagnostics

### 4.1 What `evaluate.py` already produces — [OK]

From `src/evaluate.py:182–252`:
- Test accuracy;
- Macro precision, macro recall, macro F1;
- Weighted F1;
- Per-class precision / recall / F1 / support;
- Raw confusion matrix and row-normalized confusion matrix (PNG + JSON);
- `classification_report.json` and `metrics.json`.

It correctly reloads the exact split and saved statistics, rebuilds the same architecture, and runs in inference mode. No test data is used for selection.

**Run to obtain the held-out numbers:** `!python src/evaluate.py` on Kaggle. No test results are reported in this document because none have been measured.

### 4.2 Missing diagnostics

| Diagnostic | Why it matters | Where to add / how to derive |
|---|---|---|
| **Predicted class distribution** | Distinguishes "weak but balanced signal" from "predicts two priors" | Count `y_pred` by class; add to `metrics.json`. Also compare to true class counts |
| **Train-vs-validation gap, per epoch** | Distinguishes underfitting from overfitting | Already recoverable from `results/training_history.csv` (`train_accuracy`, `val_accuracy`, both losses); surface as a summary and plot |
| **Learning curves (PNG)** | Visualise convergence and plateau | Plot `training_history.csv` columns versus epoch |
| **Majority-class / uniform baseline on the test set** | Gives the correct chance reference for *this* split | Trivial counters in `evaluate.py` |
| **Frozen-feature linear probe** | Decisive: if a logistic regression on frozen ImageNet features reaches ~39% too, the ceiling is the features/labels, not the head; if it reaches much more, training is at fault | Small sklearn probe on pooled `model.features` embeddings from train, evaluated on val |
| **Per-city / per-group error breakdown** | Tests the small-city-median hypothesis | Join predictions back to `citi` from the CSV |
| **Street/group-overlap audit** | Tests near-duplicate leakage risk | Count shared `street` (or exact-image-hash) groups across saved splits |
| **Calibration / confidence histogram** | The proxy score is used as a soft feature; overconfidence would harm Member B | Bin `max(softmax)` vs accuracy |
| **Multiple seeds (val only)** | Quantifies whether epoch-9 selection is noise | Re-run training with seeds 0/1/2, compare val macro F1 mean ± std |
| **Test-set confusion matrix normalized by prediction** | Complements row-normalized matrix for prior-collapse checks | Column-normalize the existing matrix |

### 4.3 Underfitting vs noisy labels — decision rules

Use these *after* running the added diagnostics:

- If train accuracy is only slightly above val accuracy (e.g., both ~40%): **underfitting or label ceiling**. Add capacity/schedule/longer fine-tune (§5.2, §5.3) and run the linear probe.
- If train accuracy is much higher (e.g., 70%+ vs 40% val): **overfitting** — strengthen regularization/augmentation, reduce trainable blocks, add label smoothing.
- If the frozen-feature linear probe also plateaus near ~39%: **the labels/features are the ceiling**; further model tuning has low expected value. Prioritize proxy-label redesign and architecture change.
- If the probe reaches materially higher val macro F1: **the training recipe is the bottleneck**; prioritize schedule and fine-tuning changes.
- If the predicted-class histogram is highly skewed: **prior collapse**; revisit weighting/label smoothing.

---

## 5. Phase 3 — Improvement opportunities

Each proposal is assessed for mechanism, downside, cost and fit. Nothing here is automatically appropriate — the diagnostics in §4.2 should gate the expensive items.

### 5.1 Better augmentations

| Proposal | Why it might help | Disadvantages | Cost | Appropriate here? |
|---|---|---|---|---|
| Remove/reduce `RandomRotation(10°)` | Houses are upright; rotation creates unrealistic inputs | Slightly less invariance | Negligible | **Yes — first cheap experiment** |
| Align train crop with eval (`Resize`+`CenterCrop` or mild translation instead of `RandomResizedCrop(0.8–1.0)`) | Reduces train/inference mismatch | Less scale robustness | Negligible | **Yes** |
| Mild `RandomErasing` | Forces use of context | Can erase the house | Negligible | Low priority |
| MixUp / CutMix with soft labels | Regularizes against noisy labels | Soft labels muddy already-proxy targets; needs loss changes | Low–medium | Only after label-noise evidence |
| Stronger color jitter | Robustness to lighting | Can destroy material cues tied to price | Negligible | No strong rationale |

### 5.2 Resolution

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| 299×299 or 320×320 | More detail for façade/material cues | ~1.8–2× compute and memory | Medium | **Reasonable experiment** on Kaggle GPU |
| 384+ | Still more detail | 3×+ compute; MobileNetV2 not tuned for it | High | Not first; only if 299 helps |

Note: `build_eval_transform` already reads `image_size`/`resize_size` from saved config, so resolution is a config-level change in training (`IMAGE_SIZE`, `RESIZE_SIZE`) rather than an architecture rewrite — but it changes the checkpoint's preprocessing metadata, so old and new checkpoints must not be mixed.

### 5.3 Fine-tuning strategy and schedule

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| Cosine LR decay (or `ReduceLROnPlateau`) after warmup | Curve still moving at epoch 9; decay helps settle | One more hyperparameter | Negligible | **Yes — high benefit/cost** |
| Longer fine-tune phase (e.g., 15–25 total epochs) | More adaptation of backbone | Overfitting risk; more GPU time | Low–medium | **Yes, with early stopping** (patience currently effectively inactive at 5 fine-tune epochs) |
| Unfreeze more blocks / discriminative LRs (head 1e-4, mid 3e-5, deep 1e-5) | More capacity where it matters | More parameters to overfit | Low | Worth one run |
| Freeze BatchNorm in fine-tune | Stabilizes small-batch BN stats | Can slow adaptation | Negligible | Low priority at batch 64 |

### 5.4 Alternative architectures

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| EfficientNet-B0 (ImageNet) | Similar FLOPs to MobileNetV2, usually stronger features; 224 default | New checkpoint path; different normalization still ImageNet | Low–medium | **Best architecture comparison** |
| ResNet50 (ImageNet) | Strong, well-understood baseline | ~3–4× compute, larger checkpoint | Medium–high | Only if B0 suggests architecture matters |
| ConvNeXt-T / Swin-T | Stronger still | High compute, more tuning | High | Not justified before cheaper wins |
| Domain-specific pretraining | Would close the domain gap | Requires large labelled/self-supervised corpus | Very high | Out of scope |

### 5.5 Loss and output formulation

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| Label smoothing (0.05–0.1) | Reduces overconfidence under noisy proxy labels | Slightly different probability semantics for `condition_score` | Negligible | **Yes** |
| Class weighting | Handles class imbalance | Classes are near-balanced tertiles; almost certainly inactive | Negligible | **No — evidence says unnecessary** |
| Ordinal loss (two binary thresholds `low|≥average`, `≤average|high`) | Classes are ordered; CE ignores order | Needs new loss and output handling | Low–medium | Interesting secondary experiment |
| Focal loss | Focuses on hard examples | Can amplify label noise | Low | Not recommended with noisy labels |

### 5.6 Proxy-label and city normalization redesign

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| **Shrink city medians** to global by sample count (e.g. `median = (n*city_median + k*global)/(n+k)`, or require n≥10) | Directly addresses the confirmed small-city noise | Adds one hyperparameter | Negligible | **Yes — highest-value label fix** |
| Use `log(price_per_sqft)` before clipping/tertiles | Reduces right-skew, stabilizes thresholds | Changes label semantics slightly | Negligible | **Yes** |
| Winsorize within city rather than globally | Better outlier control per market | Circular if city groups are tiny | Low | After shrinkage |
| **Residual labels**: regress pps on `bed`, `bath`, `street`/location, use residual tertiles | Attempts to isolate the non-structural component that visuals might explain | May just capture unobserved location; requires careful train-only fit | Medium | **Strong scientific experiment**, not a default |
| Group-aware split by `street`/city | Tests near-duplicate leakage | Changes split → not comparable to current checkpoint | Low | Diagnostic only; keep main split fixed |
| Human condition annotations | Would make labels ground truth | Not available; out of scope | N/A | Not feasible |

### 5.7 Continuous regression alternative

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| Regress clipped/log ratio with MSE/Huber, threshold predictions for the proxy score | Uses far more information than 3 bins; often learns better features | New loss/head, output scaling, checkpoint and `VisionAgent` changes; sensitive to outliers | Medium | **Worth a standalone experiment**, not a replacement without evidence |
| Multi-task: CE class + regression auxiliary | Combines both signals | More loss balancing | Medium | Secondary |

### 5.8 Other

| Proposal | Why | Downside | Cost | Appropriate? |
|---|---|---|---|---|
| Test-time augmentation (hflip + multi-crop average) | Cheap accuracy bump | Extra inference time; changes `VisionAgent` semantics | Low | After core fixes |
| Weight averaging (EMA / SWA) | Stabilizes noisy validation selection | Extra state | Low | Optional |
| Hyperparameter grid search | Might find a better point | Wasteful without diagnostics | High | **No — diagnose first** |
| Increase epochs alone | Curve still rising | User explicitly wants scientific improvement; likely small gain | Low | Only as part of a scheduled run |

---

## 6. Phase 4 — Prioritized action plan

### Priority 1 — Critical correctness / robustness (do first)
1. **[CONFIRMED]** Add minimum-group-size **shrinkage for city medians** and use `log(price_per_sqft)` for clipping and tertiles. This is the only confirmed code-level source of avoidable label noise. Fit on train only; persist new thresholds in the checkpoint — do **not** overwrite the existing best checkpoint (use a new `--output_dir`).
2. **[CONFIRMED]** Add the missing diagnostics from §4.2 (predicted-class distribution, train-vs-val summary + plot, learning curves, baselines). These are prerequisites for every later decision.
3. **[CONFIRMED]** Run the street/group overlap audit to quantify near-duplicate leakage risk before trusting test numbers.

### Priority 2 — High-impact, low-cost improvements
4. Remove `RandomRotation`, reduce crop-scale range, and align training crop with evaluation preprocessing.
5. Add cosine LR decay with warmup and extend the fine-tune phase with functioning early stopping (current patience cannot trigger within 5 epochs).
6. Add label smoothing (0.05–0.1).
7. Keep the existing 70/15/15 split and seed 42 fixed for all of the above so results are comparable.

### Priority 3 — Experiments worth running on Kaggle GPU
8. Frozen-feature linear probe (decisive underfitting-vs-ceiling test).
9. EfficientNet-B0 at 224, then 299 if 224 is promising.
10. Continuous log-ratio regression as a separate experiment.
11. Residual-after-structural proxy labels (ridge on `bed`/`bath`/location, train-only) — highest scientific upside if visuals carry signal beyond price/sqft.
12. Multi-seed validation runs (3 seeds) to quantify selection noise.

### Priority 4 — Optional architecture comparisons
13. ResNet50, then ConvNeXt-T / Swin-T **only if** EfficientNet-B0 shows that architecture is the limiting factor.

### Priority 5 — Unlikely to justify their cost
14. Class weighting (near-balanced classes; weighting code is effectively inactive).
15. Broad hyperparameter sweeps before diagnostics exist.
16. Resolutions ≥384, self-supervised pretraining, or heavy TTA before the label ceiling is understood.
17. Simply increasing epochs with no schedule change.

---

## 7. Proposed Kaggle experiment plan

All experiments: fixed seed 42, fixed 70/15/15 split, new `--output_dir` per run, evaluate on the untouched test set **once** per selected validation winner. Do not overwrite `best_model.pth`.

| ID | Experiment | Change | Primary metric | Cost | Decision rule |
|---|---|---|---|---|---|
| E0 | Diagnostics baseline | Add §4.2 diagnostics; run `evaluate.py` on existing checkpoint | Test macro F1 + predicted-class histogram + train/val gap | ~0 GPU (eval only) | Establishes reference and prior-collapse check |
| E1 | Label hygiene | Shrunken city medians + log pps | Val macro F1 | ~1 training run | Keep if val macro F1 improves; else revert |
| E2 | Augmentation fix | No rotation, eval-aligned crop | Val macro F1 | ~1 run | Keep if improves or equals E1 |
| E3 | Schedule | Cosine decay + 20 total epochs, early stopping active | Val macro F1 | ~2× E1 cost | Keep if improves; watch train/val gap |
| E4 | Regularization | Label smoothing 0.1 on best of E1–E3 | Val macro F1, confidence histogram | ~1 run | Keep if calibration/val improves |
| E5 | Probe | Logistic regression on frozen features | Val macro F1 | Low | If probe ≈ model → label/feature ceiling; deprioritize tuning |
| E6 | Architecture | EfficientNet-B0 224 (then 299) | Val macro F1 | Medium | Keep if clearly better than best MobileNet run |
| E7 | Label redesign | Residual-after-structural tertiles | Val macro F1 | Medium | Keep only if signal improves without leakage |
| E8 | Objective | Log-ratio regression (Huber) | Val MAE + thresholded macro F1 | Medium | Independent track; compare proxy usefulness |
| E9 | Robustness | 3 seeds for the winning recipe | Mean ± std val macro F1 | 3× | Reject unstable wins |

---

## 8. Risks and limitations

- **Scientific meaning.** Labels are market-derived proxies, not human-annotated condition or aesthetics. Price per square foot is confounded by location, demand, property characteristics and market timing. A higher `condition_score` does **not** mean the house is in objectively better physical condition, nor does the model predict final price.
- **No measured test results.** This audit reports no held-out metrics because none were provided or measured. All test-side statements are conditional on running `evaluate.py`.
- **Static analysis only.** No code was executed; findings about behavior under the real dataset (city-count distribution, duplicate-image rate, actual class counts, train/val gap) remain unmeasured.
- **Checkpoint safety.** Label or preprocessing changes alter threshold/size metadata. Use new output directories; never overwrite the trained checkpoint while experimenting.
- **Split stability.** Do not change the 70/15/15 split or seed for comparability. A grouped split is a diagnostic, not a drop-in replacement.
- **Inference contract.** `VisionAgent.score()` returns `condition_score = 0.5·P(average) + 1.0·P(high)`, a probability-derived proxy with no physical calibration. Keep that contract and wording.
- **Compute cost.** Higher resolution and larger architectures increase Kaggle GPU time; the plan orders cheap diagnostics before expensive comparisons.

---

## 9. Final verdict

**Is the current model technically sound?** Yes. Architecture, pretrained weights, normalization, train-only statistic fitting, split discipline, freeze/unfreeze, checkpointing, evaluation and inference are correctly implemented. No data leakage was found. Two robustness gaps are worth fixing: unshrunk city medians and the train/eval augmentation mismatch. Neither is a bug that invalidates the result.

**Are the results scientifically meaningful?** Only in a narrow sense. ~39.6% accuracy and 0.3961 macro F1 are above chance for a balanced 3-class task, so the exterior image carries *some* information about the market-derived target. But the target is a weak proxy dominated by non-visual confounders, so this result should be reported as "limited image-derived signal about relative price per square foot in this market," **not** as evidence of a working physical-condition classifier and not as a failure of engineering. The modest number is most plausibly a label-signal ceiling, with a short schedule and small-city median noise as secondary contributors.

**What should be improved first?** In order: (1) add the missing diagnostics and run the baseline evaluation; (2) fix label hygiene with shrunken city medians and log-ratio tertiles; (3) align augmentation and add a LR schedule with longer fine-tuning plus label smoothing; (4) run the frozen-feature probe to decide whether further model tuning is worthwhile; (5) then compare EfficientNet-B0 and, only if justified, more expensive architectures or a continuous-regression objective.

**Bottom line:** keep the implementation, improve the evidence and the labels before changing the model.
