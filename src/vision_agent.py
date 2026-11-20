"""Independent inference interface for Member B.

    from src.vision_agent import VisionAgent

    agent = VisionAgent()
    agent.load("models/best_model.pth")
    result = agent.score("example_house.jpg")

The caller never needs to know that MobileNetV2 is used internally. No CSV, no
dataset and no training-time code is needed for inference: everything required
is stored inside the checkpoint.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import build_eval_transform  # noqa: E402
from src.model import build_model_from_checkpoint, load_checkpoint  # noqa: E402


class VisionAgent:
    """Scores a house exterior image with a market-derived visual-quality proxy."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.transform = None
        self.idx_to_label: Dict[int, str] = {}
        self.preprocessing: Dict[str, Any] = {}
        self.checkpoint: Dict[str, Any] = {}

    def load(self, weights_path: str) -> "VisionAgent":
        """Load checkpoint, model architecture, class mapping and preprocessing."""
        checkpoint = load_checkpoint(weights_path, map_location="cpu")
        self.preprocessing = checkpoint.get("preprocessing", {})
        self.idx_to_label = {
            int(index): str(name)
            for index, name in checkpoint["idx_to_label"].items()
        }
        self.transform = build_eval_transform(self.preprocessing)
        self.model = build_model_from_checkpoint(checkpoint, device=self.device)
        self.checkpoint = checkpoint
        return self

    def score(self, image_path: str) -> Dict[str, Any]:
        """Return condition score, predicted label, class probabilities and confidence.

        Raises:
            RuntimeError: if ``load`` has not been called.
            FileNotFoundError: if the image path does not exist.
            ValueError: if the image is corrupted or not readable.
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Call VisionAgent.load(weights_path) before score().")
        if not isinstance(image_path, (str, os.PathLike)):
            raise TypeError("image_path must be a string or a path-like object.")
        if not os.path.isfile(os.fspath(image_path)):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")  # handles grayscale, RGBA, palette, etc.
        except Exception as exc:
            raise ValueError(f"Could not read image {image_path!r}: {exc}") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        context = (
            torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
        )
        with context():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        probabilities = np.asarray(probabilities, dtype=float)
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Model produced invalid probabilities.")
        probabilities = probabilities / total  # numerical safety; sums to 1
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities = probabilities / probabilities.sum()

        prob_dict = {
            self.idx_to_label.get(i, str(i)): float(value)
            for i, value in enumerate(probabilities)
        }
        predicted_index = int(np.argmax(probabilities))
        label = self.idx_to_label.get(predicted_index, str(predicted_index))

        # Continuous proxy: P(average) contributes 0.5, P(high) contributes 1.0.
        score = (
            0.5 * float(prob_dict.get("average", 0.0))
            + 1.0 * float(prob_dict.get("high", 0.0))
        )
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "condition_score": score,
            "label": label,
            "probabilities": prob_dict,
            "confidence": float(np.max(probabilities)),
        }
