"""Independent inference interface for Member B.

    from src.vision_agent import VisionAgent

    agent = VisionAgent(checkpoint_path="models/best_model.pth")
    result = agent.score("path/to/property_exterior.jpg")

The caller never needs to know which CNN is used internally. No CSV, no dataset
and no training-time code is needed for inference: everything required is stored
inside the checkpoint. Gradients are disabled and the model is in eval mode, so
inference is deterministic for a given checkpoint and device.

SCIENTIFIC NOTE: ``condition_score`` is a NORMALIZED MARKET-DERIVED VISUAL
PROXY, not a calibrated physical-condition measurement and not a price estimate.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import build_eval_transform  # noqa: E402
from src.model import build_model_from_checkpoint, get_architecture, load_checkpoint  # noqa: E402

SCORE_TYPE = "market_derived_visual_proxy"
PathLike = Union[str, os.PathLike]


def _resolve_device(requested: Optional[Union[str, torch.device]]) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested, torch.device):
        device = requested
    else:
        device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use device='cpu' or device='auto' for CPU fallback."
        )
    return device


class VisionAgent:
    """Score exterior images with a market-derived visual proxy CNN.

    Supports both the original MobileNetV2 checkpoint and checkpoints produced
    by the improved (log-ratio / shrunken-city) label methodology, as well as
    EfficientNet-B0 checkpoints. The architecture and all preprocessing are read
    from the checkpoint itself.
    """

    def __init__(
        self,
        checkpoint_path: Optional[PathLike] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Create an agent, optionally loading a checkpoint immediately.

        Args:
            checkpoint_path: Path to a self-contained ``.pth`` checkpoint. The
                legacy pattern ``VisionAgent().load(path)`` still works.
            device: ``"auto"`` (default), ``"cpu"``, ``"cuda"`` or a torch.device.
        """
        self.device = _resolve_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.transform = None
        self.idx_to_label: Dict[int, str] = {}
        self.preprocessing: Dict[str, Any] = {}
        self.label_stats: Dict[str, Any] = {}
        self.architecture = "mobilenet_v2"
        self.checkpoint: Dict[str, Any] = {}
        if checkpoint_path is not None:
            self.load(checkpoint_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, weights_path: PathLike) -> "VisionAgent":
        """Load checkpoint, model architecture, class mapping and preprocessing.

        No pretrained-weight download is attempted: reconstruction always uses
        ``pretrained=False`` and the saved state dict.

        Raises:
            FileNotFoundError: if the checkpoint does not exist.
            KeyError: if required checkpoint fields are missing.
            ValueError: if the saved architecture is unsupported.
        """
        path = os.fspath(weights_path)
        checkpoint = load_checkpoint(path, map_location="cpu")
        for key in ("model_state_dict", "idx_to_label"):
            if key not in checkpoint:
                raise KeyError(
                    f"Checkpoint {path!r} is missing required field {key!r}; "
                    "it was not produced by this project's training script."
                )
        self.preprocessing = checkpoint.get("preprocessing", {})
        self.label_stats = checkpoint.get("label_stats", {})
        self.idx_to_label = {
            int(index): str(name) for index, name in checkpoint["idx_to_label"].items()
        }
        self.architecture = get_architecture(checkpoint)
        self.transform = build_eval_transform(self.preprocessing)
        self.model = build_model_from_checkpoint(checkpoint, device=self.device)
        self.checkpoint = checkpoint
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_loaded(self) -> None:
        if self.model is None or self.transform is None:
            raise RuntimeError("Load a checkpoint first: VisionAgent(checkpoint_path=...) or .load(path).")

    @staticmethod
    def _read_rgb(image_path: PathLike) -> Image.Image:
        if not isinstance(image_path, (str, os.PathLike)):
            raise TypeError("image_path must be a string or a path-like object.")
        path = os.fspath(image_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            with Image.open(path) as image:
                return image.convert("RGB")  # grayscale, RGBA, palette, etc.
        except Exception as exc:
            raise ValueError(f"Could not read image {path!r}: {exc}") from exc

    def _probabilities(self, image_paths: Sequence[PathLike]) -> np.ndarray:
        """Return an [N, K] probability array; never fabricates fallback values."""
        self._require_loaded()
        assert self.model is not None and self.transform is not None
        tensors = [self.transform(self._read_rgb(path)) for path in image_paths]
        batch = torch.stack(tensors).to(self.device)
        context = (
            torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
        )
        with context():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RuntimeError("Model produced invalid (non-finite) probabilities.")
        totals = values.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or (totals <= 0).any():
            raise RuntimeError("Model produced invalid probability totals.")
        values = values / totals
        values = np.clip(values, 0.0, 1.0)
        values = values / values.sum(axis=1, keepdims=True)
        return values

    def _format_result(self, probabilities: np.ndarray) -> Dict[str, Any]:
        prob_dict = {
            self.idx_to_label.get(i, str(i)): float(value)
            for i, value in enumerate(probabilities)
        }
        predicted_index = int(np.argmax(probabilities))
        label = self.idx_to_label.get(predicted_index, str(predicted_index))

        # Normalized market-derived proxy: P(average) contributes 0.5, P(high) 1.0.
        # This is a probability-derived soft feature, NOT physical calibration.
        score = 0.5 * float(prob_dict.get("average", 0.0)) + 1.0 * float(
            prob_dict.get("high", 0.0)
        )
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "condition_score": score,
            "label": label,
            "probabilities": prob_dict,
            "confidence": float(np.max(probabilities)),
            "model": self.architecture,
            "score_type": SCORE_TYPE,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, image_path: PathLike) -> Dict[str, Any]:
        """Score one exterior photograph.

        Returns a dictionary with ``condition_score`` in [0, 1], the argmax
        ``label``, class ``probabilities`` (summing to ~1), ``confidence``, the
        ``model`` architecture name and ``score_type``.

        Raises:
            RuntimeError: if no checkpoint is loaded or outputs are invalid.
            FileNotFoundError: if the image path does not exist.
            ValueError: if the image cannot be decoded.
        """
        return self._format_result(self._probabilities([image_path])[0])

    def score_batch(
        self, image_paths: Sequence[PathLike], batch_size: int = 64
    ) -> List[Dict[str, Any]]:
        """Score multiple images, returning one result per input path.

        Invalid paths or corrupted images raise the same clear errors as
        :meth:`score`; results are returned in input order.
        """
        paths = list(image_paths)
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        results: List[Dict[str, Any]] = []
        for start in range(0, len(paths), int(batch_size)):
            chunk = paths[start:start + int(batch_size)]
            probabilities = self._probabilities(chunk)
            results.extend(self._format_result(row) for row in probabilities)
        return results
