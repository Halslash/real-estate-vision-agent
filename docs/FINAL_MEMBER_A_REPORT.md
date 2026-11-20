# Final Member A report — CNN visual proxy

**Status:** implementation and experiment harness complete; new Kaggle GPU runs
`PENDING`. Historical results are preserved below and were not re-measured.

> **Interpretation warning:** all targets are market-derived visual proxies
> (shrunken log price-per-sqft tertiles), **not** human-annotated physical
> condition or aesthetic grades, and **not** prices. See `docs/MODEL_CARD.md`.

## 1. Objective and scope

Member A delivers a CNN that turns a property exterior photo into a normalized
visual proxy feature for Member B. Scope is strictly Member A: no Random Forest,
no Linear Regression, no final price prediction. The incumbent checkpoint is
`/kaggle/working/output/improved_mobilenet/models/best_model.pth`.

## 2. Method

- **Backbone:** MobileNetV2 ImageNet V1 (default); EfficientNet-B0 for one
  controlled comparison. Fresh 3-class head, 224×224 default.
- **Labels:** `v2_log_shrunk` — tertiles of clipped `log(price/sqft)` minus a
  city log-median shrunk toward the global training median with `K=10`.
  Everything is fitted on the **train split only**.
- **Training:** two phases (frozen head → last blocks unfrozen), AdamW, cosine
  LR, weighted CE by default, validation macro F1 (fixed 3 labels) for
  checkpoint selection, early stopping in phase 2.
- **Output:** `condition_score = 0.5·P(average) + 1·P(high) ∈ [0,1]`.

## 3. What was wrong and what changed

See `docs/CNN_PERFORMANCE_AUDIT.md` for the full evidence-backed audit. Short
version: variable-label macro F1 during selection, stale "PENDING" docs despite
measured results, no leakage diagnostics, no OOF path for Member B, no
loss/architecture experiments, no timing/VRAM record, no tests.

Changes: fixed-label selection F1; config-driven runner with 11 isolated
experiments; focal/ordinal/effective-number losses; read-only leakage audit;
OOF fold generation and honest OOF scoring; duration/VRAM metadata; tests;
this documentation set. Architecture, labels, score formula and historical
checkpoints are unchanged.

## 4. Historical measured results (preserved verbatim)

| Run | Split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Improved MobileNetV2 (incumbent) | validation | 0.4589 | 0.4521 (best epoch 15) |
| Improved MobileNetV2 (incumbent) | **test** | 0.4384 | 0.4309 |
| Original baseline (`v1_legacy`) | test | 0.4065 | 0.4057 |
| Frozen ImageNet probe | validation | 0.4580 | 0.4561 (diagnostic only) |

Incumbent test detail: macro precision 0.4323, macro recall 0.4362, weighted F1
0.4327; per-class F1 low 0.4684 / average 0.3293 / high 0.4948.

The baseline used a different label configuration, so its numbers are **not** a
controlled comparison. The probe being level with the CNN suggests a
feature/label ceiling rather than a head-tuning problem.

## 5. New experiments (no results yet)

Eleven controlled experiments are defined in `configs/experiments.json` and
listed in `docs/EXPERIMENT_PLAN.md`. Live results, failures and the selection
decision belong in `docs/EXPERIMENT_RESULTS.md`. No new number is claimed here.

**Selection rule:** validation macro F1 with per-class support. Final test
evaluation runs **once**, for the winner only. If no new model convincingly
beats 0.4521 validation macro F1, the incumbent checkpoint stays in production.

## 6. Member B hand-off

Use `src.vision_agent.VisionAgent` (see `docs/MEMBER_B_INTEGRATION.md`).
When training Member B's structural model, use the OOF `condition_score` CSV for
training rows so the visual feature is never in-sample; use the selected final
checkpoint only for held-out/test rows.

## 7. Limitations

- Proxy labels inherit all price-per-sqft confounds (location, demand, timing,
  property characteristics). The weak average class likely reflects a
  heterogeneous middle tertile, not a fixable software bug.
- No human condition labels, no calibrated condition probability, no
  uncertainty estimates.
- Geographic/perceptual duplicate leakage has not been measured yet; run
  `src/data_audit.py` on Kaggle before trusting held-out scores as absolute.
- Regression and validation-threshold experiments are deliberately deferred.
