# Member A technical handoff — CNN visual proxy

**Status:** final, documentation-only. This is the interface and leakage contract
between Member A (the visual-proxy CNN) and Member B (structural model + final
price prediction). **Member B is not yet implemented; no Member B code is
assumed to exist.** Nothing here retrains, evaluates, or implements Member B.

> Interpretation warning: the visual target is a **market-derived visual proxy**
> (a tertile of prices per square foot relative to a shrunken city median),
> **not** a human-annotated physical-condition or aesthetic grade, and **not** a
> price. See `docs/MODEL_CARD.md`.

---

## 1. What Member A provides

| Item | Detail |
|---|---|
| Final checkpoint | `/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth` |
| Run | `e0_reference_group_split`, pinned commit `bc39d102d2e9cde83144966d6e6c910926109429` |
| Config | MobileNetV2, `v2_log_shrunk` (K=10), 224 px crop / 256 resize, CE + auto class weights β=0.9999, label smoothing 0.05, seed 42, 30 epochs, best epoch 27 |
| Result | Validation Macro F1 **0.4314855**; one-time test Macro F1 **0.4240** on **2,308** samples (property-disjoint split) |
| Reusable source (unchanged) | `src/vision_agent.py`, `src/oof_contract.py`, `src/oof_predict.py`, `src/train.py --split_file` |
| Provenance to record | `training_gate.json`, `models/{split_indices.json,training_summary.json,model_config.json,preprocessing_stats.json,city_medians.json,class_thresholds.json}`, `results/{metrics.json,evaluation_gate.json}` |

No Member A source, configuration, checkpoint, split or sealed metric is modified
by this handoff.

## 2. Inference interface — `VisionAgent`

Public API (`src/vision_agent.py:52-215`), imported from the repository root:

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="/path/to/best_model.pth", device="auto")
result = agent.score("/path/to/exterior.jpg")          # one image
results = agent.score_batch(paths, batch_size=64)      # list, order preserved
```

- `VisionAgent(checkpoint_path=...)` loads immediately; the legacy
  `VisionAgent(device="cuda").load(path)` also works.
- `device` accepts `"auto"` (default), `"cpu"`, `"cuda"` or a `torch.device`.
  `"auto"` uses CUDA when available, else CPU; explicit `"cuda"` fails closed if
  CUDA is unavailable. No download is attempted at inference.
- Inputs: paths to exterior photographs. RGB, grayscale, RGBA and palette images
  are converted to RGB. Missing paths raise `FileNotFoundError`, undecodable
  images raise `ValueError`, and invalid/non-finite model output raises
  `RuntimeError` — **no fabricated fallback score**.
- Gradients are disabled and the model is in eval mode, so scoring is
  deterministic for a given checkpoint and device.
- Inference needs **no CSV, no dataset, no label refit and no training code** —
  everything required is stored inside the checkpoint.

## 3. Output contract and score definition

`score()` / `score_batch()` return exactly these keys (`src/vision_agent.py:158-180`):

| Key | Meaning |
|---|---|
| `condition_score` | `0.5·P(average) + 1.0·P(high)`, clipped to `[0,1]` (`src/vision_agent.py:168-171`) |
| `label` | Argmax class: `low` \| `average` \| `high` |
| `probabilities` | Full vector `{"low": …, "average": …, "high": …}` (renormalized, sums to ~1) |
| `confidence` | `max(P)` — **maximum class probability, NOT calibrated confidence** |
| `model` | Architecture name, e.g. `mobilenet_v2` |
| `score_type` | Always `"market_derived_visual_proxy"` |

Pass through the **full probability vector** (`p_low`, `p_average`, `p_high`),
not only the scalar score. The score is a soft, probability-derived feature, not
a physical measurement, aesthetic rating, calibrated probability or price.
Member B owns any downstream thresholding or interpretation.

## 4. Checkpoint and preprocessing requirements

- The checkpoint is self-contained: `model_state_dict`, `idx_to_label`,
  `preprocessing`, `label_stats`, architecture, and training config
  (`VisionAgent.load()`, `src/vision_agent.py:88-116`). Reconstruction uses
  `pretrained=False` + the saved state dict; nothing is fetched at inference.
- Eval transform (`build_eval_transform`): RGB → Resize(256) → CenterCrop(224) →
  ImageNet normalization. This is derived from checkpoint metadata, so a fold
  checkpoint trained with different settings carries its own correct transform.
- Label statistics were fitted on **train rows only**: clip `log(price/sqft)` to
  train p1/p99, global median, city medians shrunk toward it with `K=10`, and
  tertiles from the observed distinct train ratios. They are frozen in the
  checkpoint and must never be recomputed from a different split at inference.
- **Do not reuse a checkpoint's frozen `label_stats` for a different split.**
  `train.py --split_file` refits statistics on the ids in that file's `train`
  list (`src/train.py:147-149`), which is what fold-specific training needs.

## 5. OOF utilities and expected row contract

Member A supplies the interface, **not** a ready OOF file.

- `src/oof_contract.py` (pure, torch-free, unit-tested):
  - `OOF_COLUMNS` = `image_id, fold, condition_score, p_low, p_average, p_high,
    label, confidence, model, checkpoint` (`src/oof_contract.py:11-22`).
  - `make_oof_row(image_id, fold, result, checkpoint)` builds one row from a
    `VisionAgent` result, validating the score/probabilities
    (`src/oof_contract.py:65-98`).
  - `check_fold_spec` fails closed on train/val overlap, test-id leakage, wrong
    fold ownership or a fold training on its own held-out row
    (`src/oof_contract.py:25-49`).
  - `check_coverage` requires exactly one validation assignment per owned row,
    refusing partial output (`src/oof_contract.py:52-62`).
- `src/oof_predict.py` reads a fold root with `oof_index.json`, `folds/fold_k.json`
  and `<root>/fold_k/models/best_model.pth`, runs the checks above, scores each
  held-out set with its own fold checkpoint, and writes `oof_visual_scores.csv`
  with exactly `OOF_COLUMNS` and one row per owned id
  (`src/oof_predict.py:59-113`).

**Critical:** the existing `oof_visual_scores.csv` and the `make_folds.py` output
are **unsafe and must not be reused**. `make_folds.py` uses row-level `KFold`,
so same-street rows can straddle folds and `check_fold_spec` only checks row ids;
the CNNs that produced those scores were trained on a random row split of the
full data and may have seen a future outer-test image (Member B audit §5.1).

## 6. Future group-aware fold requirements (Member B's responsibility)

Member A makes **no code change** for these (Member B audit §9: "NONE"). Member B
must supply the fold structure and run the training:

1. Define a group-aware outer split from the **raw CSV** (Member A's records drop
   `street`), using the group key `f"{city}||{street}"`; assert no group appears
   on both sides of the outer split before sealing it.
2. Regenerate inner OOF **from outer-train only**. Build group-aware fold JSONs
   with `train`, `val` and `test: []` (from the raw group key, not
   `make_folds.py`), so no held-out group appears in that fold's train set.
3. For each fold, train a CNN with `train.py --split_file <fold_split.json>`,
   which refits label statistics on fold-train ids only, then score that fold's
   held-out rows with `VisionAgent` / `oof_predict.py`.
4. Train one final CNN on **outer-train only** to score the outer-test rows once.
   At the documented settings this is **5 fold CNNs + 1 final CNN = 6 trainings**
   (Kaggle GPU); none has been run here.
5. Keep outer test sealed: no fitting, selection, imputation or threshold tuning
   on it; evaluate once at the end.

## 7. CRITICAL warning — do not score outer test with the final checkpoint

**The final `e0_reference_group_split` checkpoint must NOT be used to score
Member B's outer-test rows if those rows overlap its training data.** Its train
ids are a random row subset of the full dataset, so Member B's group-aware
outer-test rows will generally already be represented. Using it for outer-test
scoring would make the held-out result optimistically biased and invalid.

Member B must instead train **fold-specific CNNs on the appropriate
outer-training data only**, and score each row with a CNN that never saw that
row (or its group). The final checkpoint is the delivered artifact for the
*Member A* result; it is not a universal scorer for another team's outer split.
OOF scores and experiment validation metrics are different protocols and must
never be mixed.

## 8. Limitations to carry forward

- Market-derived proxy labels, confounded by location/demand/timing — not
  verified physical condition, not aesthetics, not a direct price estimator.
- Weak `average` class (test F1 `0.3539`, recall `0.3070`).
- `confidence` is max probability, not calibrated; no uncertainty estimates.
- Property-disjoint final split (address + exact SHA-256); near-duplicate dHash
  grouping was disabled, so perceptually similar photos at different addresses
  remain untested.
- Final test set was evaluated **once** and sealed in `results/evaluation_gate.json`;
  **never** reuse it for model selection or tuning.

## References

- `docs/MODEL_CARD.md` — model details, final vs historical results, limitations.
- `docs/EXPERIMENT_RESULTS.md` — final results and preserved historical random-split results.
- `docs/FINAL_MEMBER_A_REPORT.md` — full Member A report and handoff summary.
- `docs/MEMBER_B_PRE_IMPLEMENTATION_AUDIT.md` — reuse verdicts and leakage register.
- `docs/MEMBER_A_KAGGLE_READINESS.md` — exact pinned training and one-time evaluation cells.
