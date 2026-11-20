# Member B integration guide — `VisionAgent`

Boundary contract between Member A (CNN visual proxy) and Member B (structural
Random Forest + final Linear Regression). Member B needs no training code, CSV or
CNN internals.

> **Read first.** `condition_score` is a **normalized market-derived visual
> proxy**, not a physical-condition grade, aesthetic score, calibrated
> probability or price. See `docs/MODEL_CARD.md`.

## 1. Load and score

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="/kaggle/working/output/improved_mobilenet/models/best_model.pth")
result = agent.score("/path/to/exterior.jpg")
batch = agent.score_batch(paths, batch_size=64)
```

Also supported: `agent = VisionAgent(device="cuda")` then `agent.load(path)`.
`device` accepts `"auto"`, `"cpu"`, `"cuda"` or a `torch.device`; CPU is used
automatically when CUDA is unavailable. The checkpoint supplies architecture,
class mapping and preprocessing; nothing is downloaded at inference time.

## 2. Output contract (unchanged)

```python
{
  "condition_score": 0.0..1.0,                 # 0.5*P(average) + 1.0*P(high)
  "label": "low" | "average" | "high",         # argmax of 3-class probabilities
  "probabilities": {"low": ..., "average": ..., "high": ...},  # sum 1
  "confidence": float,                         # max probability
  "model": "mobilenet_v2" | "efficientnet_b0",
  "score_type": "market_derived_visual_proxy",
}
```

`score()`/`score_batch()` accept RGB, grayscale and RGBA images and path-like
objects. Invalid paths raise `FileNotFoundError`, undecodable files raise
`ValueError`, and non-finite model outputs raise `RuntimeError` — there is no
fabricated fallback score. `score_batch` returns results in input order.

Append `condition_score` as **one soft feature** to Member B's structural
features. Do not use `label` as a hard condition verdict.

## 3. Leakage risk — the important part

If Member B trains on `condition_score` values produced by the **same CNN that
saw those images during training**, the visual feature is in-sample. The
structural model will over-trust it, and any offline validation will look better
than reality. Two additional risks from the data itself:

- duplicate / near-duplicate photos split across train and test;
- multiple listings on the same street (or city) across splits.

Run once on Kaggle before relying on held-out numbers:

```python
!python src/data_audit.py --split_file \
   /kaggle/working/output/experiments/e0_reference/models/split_indices.json
```

## 4. Practical OOF strategy for training rows

The CNN's **training rows must be scored out-of-fold**; its untouched test rows
must be scored only by the final selected checkpoint.

```python
# 1) Build folds from the CANONICAL TRAIN split only (test is never folded).
!python src/make_folds.py --n_folds 5 --seed 42 \
    --output_dir /kaggle/working/output/oof

# 2) Train one CNN per fold, reusing each fold's saved split.
!python src/train.py --split_file /kaggle/working/output/oof/folds/fold_0.json \
    --output_dir /kaggle/working/output/oof/fold_0 --seed 42
# ... repeat for fold_1 .. fold_4 (same flags, only split_file/output_dir change)

# 3) Score each held-out fold with its own checkpoint -> OOF CSV.
!python src/oof_predict.py --output_dir /kaggle/working/output/oof
# writes /kaggle/working/output/oof/oof_visual_scores.csv
```

`oof_visual_scores.csv` columns: `image_id`, `fold`, `condition_score`, `label`,
`confidence`, `model`, `checkpoint`. Every training row appears exactly once and
is scored only by a model that never trained on it. The scripts fail closed on
missing checkpoints, overlapping folds, incomplete coverage or any test-id
leakage. OOF folds are a **different protocol** from the 70/15/15 experiments:
never mix OOF scores with experiment validation metrics.

Member B then:

- joins `oof_visual_scores.csv` to its **training** rows by `image_id`;
- uses the selected final checkpoint via `VisionAgent` for validation/test rows;
- keeps the OOF split consistent with its own cross-validation where possible;
- never trains on scores from a checkpoint that saw the same image.

## 5. Readiness

The API and OOF tooling are implemented and unit-tested (Kaggle torch tests:
`python -m unittest discover -s tests`). No checkpoint export or Member B run
was performed locally. The incumbent checkpoint remains the default until a new
experiment wins validation macro F1 and receives its single test evaluation.
