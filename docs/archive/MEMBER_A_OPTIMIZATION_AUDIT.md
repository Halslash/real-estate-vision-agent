# Member A optimization audit — CNN visual proxy

**Date:** 2026-11-20
**Scope:** all of `src/`, `tests/`, `configs/experiments.json`, `docs/`, git state and
the recoverable baseline tag `member-a-optimization-baseline` (HEAD `7ce74b6`).
**Method:** full source/doc read, `compileall`, local `unittest`, experiment-runner
`--dry-run`, git-history search, and analytic (synthetic) threshold checks.
**No GPU, no dataset and no checkpoint files were available locally**: no model was
trained, inferred or evaluated. Every performance number below is either a
*pre-existing documented measurement* or a *synthetic diagnostic*, and is labelled
as such.

Evidence labels: **[VERIFIED]** checked against source/history here; **[DOCUMENTED]**
pre-existing Kaggle measurement preserved from repo docs, not reproduced;
**[SYNTHETIC]** produced by a local simulation, not the project dataset;
**[HYPOTHESIS]** plausible, requires a GPU experiment.

> Interpretation warning: the target is a **market-derived visual proxy** — a
> tertile of log(price/sqft) relative to a shrunken city median. It is **not** a
> physical-condition grade, an aesthetic score or a price. `condition_score` is a
> soft feature, not a calibrated probability. See `docs/MODEL_CARD.md`.

---

## 1. Baseline number verification (do this before trusting any target)

The optimization brief supplied `e0_reference` val Macro F1 `0.4563`, val accuracy
`~0.4606`, best epoch `28`, incumbent `0.4521`, historical test `0.4309`,
`~15,474` images. Verification result:

| Claimed value | Status | What the repository actually contains |
|---|---|---|
| e0_reference val Macro F1 `0.4563` | **NOT VERIFIABLE** | No run named `e0_reference` has ever been executed. `e0_reference` is a *planned* config entry (`configs/experiments.json`) whose result cell is `PENDING`. The string `0.4563` does not occur anywhere in the working tree or git history. |
| val accuracy `~0.4606`, best epoch `28` | **NOT VERIFIABLE** | No `training_summary.json`, `metrics.json`, `training_history.csv` or `.pth` exists in the working tree **or anywhere in git history**. These numbers cannot be tied to an artifact. |
| incumbent val Macro F1 `0.4521` | **[DOCUMENTED]** | `docs/CNN_PERFORMANCE_AUDIT.md` §2, `docs/EXPERIMENT_RESULTS.md`, `docs/FINAL_ACADEMIC_READINESS_AUDIT.md`: improved MobileNetV2, val acc `0.4589`, **best epoch 15** (not 28). |
| historical test Macro F1 `0.4309` | **[DOCUMENTED]** | Same sources: test acc `0.4384`, macro precision `0.4323`, macro recall `0.4362`, weighted F1 `0.4327`; per-class test F1 low `0.4684` / average `0.3293` / high `0.4948`. |
| ~15,474 images | **NOT VERIFIABLE locally** | Plausible for the mounted Kaggle dataset, but the CSV is absent here. `inspect_csv.py` must confirm it on Kaggle. |
| frozen-probe val Macro F1 `0.4561` | **[DOCUMENTED]** | `docs/CNN_PERFORMANCE_AUDIT.md` §2; val acc `0.4580`, test untouched. |

**Conclusion:** the only defensible incumbent is the historical `v2_log_shrunk`
MobileNetV2 at **val Macro F1 0.4521 / epoch 15** (and test 0.4309, already spent —
it must not be reused for selection). The brief's `0.4563 / epoch 28` figures are
treated as unverifiable and are **not** used as a selection target. A fresh
`e0_reference` run is required to establish a controlled, artifact-backed
reference on the current code.

---

## 2. Why performance is weak — ranked diagnosis

### 2.1 The ceiling appears to be signal/labels, not the classifier head
**[DOCUMENTED + HYPOTHESIS]** A frozen ImageNet logistic probe reaches val Macro
F1 `0.4561` vs the fine-tuned CNN's `0.4521`. When a linear probe on frozen
features matches or beats the trained model, extra head tuning is low-value: the
information is not separable in the image features **or** the labels are noisy.
This is why the plan prioritises label quality, loss geometry and leakage checks
over architecture head tweaks. It remains a hypothesis until `e0_reference` and
the probe are re-run on the current code.

### 2.2 The v2 tertile rule can starve a class when the proxy value has ties
**[VERIFIED code] + [SYNTHETIC evidence]** `_fit_v2_stats` (`src/dataset.py`) cuts
at positions `len(unique)//3` and `2*len(unique)//3` of the *distinct sorted*
proxy values. A value repeated 500 times counts **once**, so heavy rounding of
price or square footage can push both boundaries into a narrow band and leave
one class with almost no samples.

Synthetic check (local simulation, **not** the project data), 3000 lognormal
values rounded to 0.1:
- v2 distinct-value cuts → class counts `[2324, 649, 27]`;
- v3 sample-quantile cuts → `[711, 1201, 1088]`.

With milder rounding over 15,474 simulated values, v2 gave `[5237, 5227, 5010]`
and v3 `[5158, 5158, 5158]`. So the defect is **conditional on tie mass**, which
is exactly why it must be measured on the real CSV (`data_audit.py` reports the
proxy-value distribution; the e11 vs e0 class support is the decisive evidence).
This is the concrete, justified improvement implemented here: **v3
`v3_balanced_quantile`** uses training-sample-weighted tertiles, still fitted on
the train partition only, still assigning identical proxy values to the same
class (no rank-based label laundering).

### 2.3 The `average` class is definitionally hard
**[DOCUMENTED + HYPOTHESIS]** Test F1 for `average` is `0.3293` vs `0.4684`/`0.4948`
for the outer classes. A middle tertile of a continuous market proxy is a
heterogeneous residual bin: visually similar houses land on either side of a
thin boundary. This is not proof of a broken model. `e4_focal` (hard-example
focus) and `e5_ordinal_smooth` (soft targets for the ordered classes) directly
test whether the loss geometry helps. v3 also reduces the chance that `average`
is *artificially* small.

### 2.4 City shrinkage concentrates borderline cases
**[VERIFIED code] + [HYPOTHESIS]** `K=10` shrinks each city median toward the
global training median, so properties from small cities cluster near the same
reference and pile up around a threshold. The rule is defensible (a one-listing
city cannot define its own reference); `e9_shrinkage_k1` tests the sensitivity.
No silent change to `K` was made.

### 2.5 Held-out scores may be optimistic (leakage diagnostics not yet run)
**[VERIFIED code] + [HYPOTHESIS]** `make_stratified_split` groups only by label.
`src/data_audit.py` exists but has **not been executed** on Kaggle: exact-byte and
perceptual near-duplicates, and street/city groups spanning train/val/test, can
inflate validation and test numbers. If confirmed, the true generalisation gap is
*worse* than 0.4521/0.4309, and no loss/label tweak will fix it. This audit is a
prerequisite, not optional.

### 2.6 Training recipe is already broadly covered
**[VERIFIED]** `configs/experiments.json` already defines 11 controlled runs:
weighting schemes (`e1`–`e3`), focal/ordinal losses (`e4`–`e5`), architecture
(`e6`), resolution (`e7`), augmentation (`e8`), shrinkage (`e9`) and a longer
fine-tune (`e10`). Adding a dense hyper-parameter grid would be wasteful; the
two new runs (`e11`, `e12`) target the one untested structural factor — the
tertile rule — under the same split and seed.

---

## 3. Changes made in this optimization pass

| File | Change | Reversibility |
|---|---|---|
| `src/label_thresholds.py` | **New**, pure numpy: `unique_index_tertiles` (v2 reference), `sample_quantile_tertiles` (v3), `class_counts`. No I/O, no torch. | New file; deletable. |
| `src/dataset.py` | Add `LABEL_V3 = "v3_balanced_quantile"`; v3 shares the v2 proxy formula and city shrinkage; `_fit_v3_stats` rebuilds on the **untouched** v2 fitter and replaces only the two thresholds. Default remains `LABEL_V2`. | v1/v2 behaviour unchanged; opt-in only. |
| `src/train.py` | Accept `v3_balanced_quantile` in `--label_config_version`. | One `choices` entry. |
| `src/evaluate.py` | Accept v3 checkpoints in the version guard. | One tuple entry. |
| `configs/experiments.json` | Add `e11_v3_quantile_labels` and `e12_v3_balanced_ce`. | New config keys only. |
| `tests/test_label_thresholds.py` | **New**, 5 local tests: v2 formula fidelity, v3 balance under heavy ties, deterministic/ordered cuts, non-finite handling, fail-closed degeneracy. | New file; deletable. |

**Deliberately unchanged:** the default label version, all v1/v2 threshold code,
`VisionAgent.score` / `score_batch` and the OOF CSV contract, `condition_score`,
the untouched test split, historical checkpoints/results, all Member B files, and
the existing 11 experiment definitions.

---

## 4. Residual limitations and risks

- **[HYPOTHESIS]** v3 cannot fully balance classes if a single rounded proxy
  value holds a large share of the training mass — tied values must share a
  label. The magnitude is unknown until the real CSV is audited.
- **[VERIFIED]** No experiment has been executed in this pass. `Ready for Kaggle`
  is not `Experimentally validated`; see `docs/MEMBER_A_OPTIMIZATION_RESULTS.md`.
- **[VERIFIED]** The baseline numbers are undocumented by artifacts; a fresh
  `e0_reference` is the only way to obtain an honest controlled reference.
- **[HYPOTHESIS]** Even a perfectly balanced proxy is still a noisy market label;
  a modest Macro F1 may be close to the achievable ceiling. Report that honestly
  rather than tuning on the test split.
- **[VERIFIED]** Local tests skip all torch-dependent cases (no torch installed);
  those must run on Kaggle with `python -m unittest discover -s tests -v`.

---

## 5. Bottom line

The weak score is most likely a **label/signal ceiling** (probe ≈ CNN) aggravated
by a **conditional v2 tie-induced class imbalance** and unverified leakage. The
smallest justified code change is an opt-in v3 label rule plus two controlled
experiments; everything else was already in the harness. No performance claim is
made here beyond the historical documented numbers in §1.
