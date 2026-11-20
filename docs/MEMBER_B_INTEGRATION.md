# Member B integration guide — `VisionAgent`

This guide is the boundary contract between Member A (CNN visual proxy) and
Member B (structural Random Forest + final Linear Regression). Member B does
**not** need training code, the dataset CSV, or knowledge of the CNN internals.

> **Read this first.** `condition_score` is a **normalized market-derived visual
> proxy**, not a measured physical-condition grade, not an aesthetic score and
> not a price. Treating it as ground truth or as a calibrated probability of
> "good condition" is a prohibited interpretation. See `docs/MODEL_CARD.md`.

---

## 1. Load the trained checkpoint

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(
    checkpoint_path="/kaggle/working/output/improved_mobilenet/models/best_model.pth"
)
```

The constructor reads the architecture, class mapping, image preprocessing and
all label metadata from the checkpoint. No pretrained weights are downloaded at
inference time. CPU is used automatically when CUDA is unavailable; you can pin
the device explicitly:

```python
agent = VisionAgent(checkpoint_path="...", device="cpu")   # or "cuda" / "auto"
result = agent.score("exterior.jpg")
```

The legacy two-step form still works for the original baseline checkpoint:

```python
agent = VisionAgent()
agent.load("models/best_model.pth")
```

---

## 2. Score one exterior image

```python
result = agent.score("path/to/property_exterior.jpg")
print(result)
```

Return structure (stable contract; numbers below are **illustrative only**):

```python
{
    "condition_score": 0.73,
    "label": "high",
    "probabilities": {"low": 0.10, "average": 0.24, "high": 0.66},
    "confidence": 0.66,
    "model": "mobilenet_v2",
    "score_type": "market_derived_visual_proxy",
}
```

| Field | Meaning |
|---|---|
| `condition_score` | `0.5 * P(average) + 1.0 * P(high)`, always in `[0, 1]`. |
| `label` | Argmax proxy class: `low`, `average`, `high`. |
| `probabilities` | Softmax class probabilities; sum to ~1. With label smoothing they are deliberately softened, so `confidence` is **not** calibrated certainty. |
| `confidence` | Maximum class probability. Useful for filtering, not a guarantee. |
| `model` | Architecture reconstructed from the checkpoint (`mobilenet_v2` or `efficientnet_b0`). |
| `score_type` | Always `market_derived_visual_proxy` — a reminder of the interpretation limit. |

Inference is deterministic for a fixed checkpoint and device (eval mode,
gradients disabled). RGB conversion happens automatically for grayscale, RGBA
and palette images.

---

## 3. Score many images

```python
paths = [f"/kaggle/input/.../socal_pics/{image_id}.jpg" for image_id in df["image_id"]]
results = agent.score_batch(paths, batch_size=64)

df["condition_score"] = [r["condition_score"] for r in results]
df["visual_label"] = [r["label"] for r in results]
df["visual_confidence"] = [r["confidence"] for r in results]
```

`score_batch` returns results in input order and raises on any invalid path or
corrupted image; it never substitutes a fabricated score.

---

## 4. Handle missing or invalid images

Wrap calls where the image may be absent (the agent raises clear exceptions):

```python
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="models/best_model.pth")

def safe_score(path):
    try:
        return agent.score(path)["condition_score"]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Missing file, undecodable image, or invalid model output.
        print(f"Skipping {path}: {exc}")
        return None  # Member B decides how to impute; do NOT invent a score here.

scores = [safe_score(p) for p in paths]
```

Recommended policy: keep `None` as missing and let Member B's pipeline decide
whether to impute with the training-set mean or add a missingness indicator.
Do not fill with a fabricated "average" CNN score.

---

## 5. Use the score as an additional feature

Member B owns all of the following; Member A deliberately does not implement it.

```python
# --- Member B's pipeline (illustrative only, NOT part of this repository) ---
import os
from src.vision_agent import VisionAgent

agent = VisionAgent(checkpoint_path="models/best_model.pth")
images_dir = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/"

df["condition_score"] = [
    agent.score(os.path.join(images_dir, f"{image_id}.jpg"))["condition_score"]
    for image_id in df["image_id"]
]

# Structural feature matrix X_struct: bed, bath, sqft, city, ...
# X_final = concat[X_struct, df[["condition_score"]]]
# RandomForest -> its prediction -> LinearRegression -> final price estimate.
```

Practical notes for Member B:

- Score **each image once** and reuse the value; do not refit the CNN.
- Use the same checkpoint (and therefore the same label methodology) for every
  row so the feature has a consistent meaning.
- `condition_score` is relative to the SoCal price-per-sqft market used to build
  the labels. It is not portable to another market without retraining.
- Do not use the test split to decide how much weight to give this feature.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Load a checkpoint first` | No checkpoint provided. | Pass `checkpoint_path=` or call `.load(path)`. |
| `FileNotFoundError` on image | Wrong base directory or filename. | Confirm `<image_id>.jpg` exists under `--images_dir`. |
| `ValueError: Could not read image` | Corrupt/unsupported file. | Skip the row via the `safe_score` pattern above. |
| `ValueError: Unsupported checkpoint architecture` | Checkpoint from an older/incompatible package. | Use the package version that produced the checkpoint. |
| Probabilities look "flat" | Label smoothing (0.05) plus a weak proxy label. | Expected; do not interpret as model confidence. |

---

## 7. Contract summary

- Stable API: `VisionAgent(checkpoint_path=...)`, `.score(path)`, `.score_batch(paths)`.
- Stable fields: `condition_score`, `label`, `probabilities`, `confidence`, `model`, `score_type`.
- No dataset CSV, no pretrained download, no training imports at inference time.
- Prohibited interpretations are documented in `docs/MODEL_CARD.md`.
