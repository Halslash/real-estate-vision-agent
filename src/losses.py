"""Loss functions for Member A's controlled experiments.

All losses keep the existing 3-logit head and the ``condition_score`` contract,
so checkpoints remain compatible with ``VisionAgent``. ``compute_class_weights``
implements both the legacy "inverse frequency above a ratio" rule and
effective-number weighting (Cui et al., 2019).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_CE = "ce"
LOSS_FOCAL = "focal"
LOSS_ORDINAL_SMOOTH = "ordinal_smooth"
SUPPORTED_LOSSES = (LOSS_CE, LOSS_FOCAL, LOSS_ORDINAL_SMOOTH)

CLASS_WEIGHT_AUTO = "auto"  # original behaviour: inverse frequency above a ratio
CLASS_WEIGHT_NONE = "none"
CLASS_WEIGHT_BALANCED = "balanced"
CLASS_WEIGHT_EFFECTIVE_NUMBER = "effective_number"
SUPPORTED_CLASS_WEIGHTING = (
    CLASS_WEIGHT_AUTO,
    CLASS_WEIGHT_NONE,
    CLASS_WEIGHT_BALANCED,
    CLASS_WEIGHT_EFFECTIVE_NUMBER,
)


def compute_class_weights(
    counts,
    mode: str = CLASS_WEIGHT_NONE,
    *,
    beta: float = 0.9999,
    imbalance_ratio: float = 1.5,
) -> tuple[np.ndarray, bool]:
    """Return float32 class weights and whether weighting is actually active.

    ``auto`` preserves the original rule exactly: inverse-frequency weights are
    used only when max/min exceeds ``imbalance_ratio``. ``effective_number``
    uses ``(1-beta)/(1-beta**n_c)`` and rescales so the mean weight is 1.
    """
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid class counts: {counts!r}")
    if (values <= 0).any():
        raise ValueError(f"Every class needs at least one training sample: {counts.tolist()}")

    num_classes = values.size
    if mode == CLASS_WEIGHT_NONE:
        return np.ones(num_classes, dtype=np.float32), False
    if mode == CLASS_WEIGHT_AUTO:
        inverse = values.sum() / (num_classes * values)
        active = bool(inverse.max() / inverse.min() > float(imbalance_ratio))
        return (inverse if active else np.ones(num_classes, dtype=np.float32)), active
    if mode == CLASS_WEIGHT_BALANCED:
        inverse = values.sum() / (num_classes * values)
        return inverse.astype(np.float32), True
    if mode == CLASS_WEIGHT_EFFECTIVE_NUMBER:
        beta = float(beta)
        if not 0.0 < beta < 1.0:
            raise ValueError(f"--class_weight_beta must be in (0, 1), got {beta!r}.")
        weights = (1.0 - beta) / (1.0 - np.power(beta, values))
        weights = weights * num_classes / weights.sum()  # mean weight = 1
        return weights.astype(np.float32), True
    raise ValueError(
        f"Unknown class weighting {mode!r}; choose from {list(SUPPORTED_CLASS_WEIGHTING)}."
    )


class FocalLoss(nn.Module):
    """Multi-class focal loss; ``weight`` must already be on the target device."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"--focal_gamma must be non-negative, got {gamma!r}.")
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        index = targets.unsqueeze(1)
        ce = -log_probs.gather(1, index).squeeze(1)
        pt = probs.gather(1, index).squeeze(1)
        losses = (1.0 - pt).pow(self.gamma) * ce
        if self.weight is not None:
            losses = losses * self.weight.gather(0, targets)
        return losses.mean()


class OrdinalSmoothLoss(nn.Module):
    """Cross-entropy with a small mass moved to the adjacent *ordered* class.

    This is deliberately an ordinal-aware soft target, not a cumulative-logit
    ordinal head: it keeps the 3-class probability interface unchanged. Middle
    samples split the mass equally between low/high; boundary samples give it
    to their single neighbour.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, epsilon: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= float(epsilon) <= 0.5:
            raise ValueError(f"--ordinal_epsilon must be in [0, 0.5], got {epsilon!r}.")
        self.epsilon = float(epsilon)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        num_classes = logits.size(1)
        if num_classes != 3:
            raise ValueError("Ordinal smoothing is defined for the 3 ordered classes only.")

        half = torch.full_like(targets, 0.5, dtype=logits.dtype)
        zero = torch.zeros_like(half)
        left = torch.where(targets > 0, half, zero)
        right = torch.where(targets < num_classes - 1, half, zero)
        mass = (left + right).clamp_min(1e-6)  # 1.0 middle, 0.5 boundary
        epsilon = self.epsilon
        left = left / mass * epsilon
        right = right / mass * epsilon

        soft = torch.zeros_like(logits)
        soft.scatter_add_(1, (targets - 1).clamp(min=0).unsqueeze(1), left.unsqueeze(1))
        soft.scatter_add_(1, (targets + 1).clamp(max=num_classes - 1).unsqueeze(1), right.unsqueeze(1))
        soft.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)

        log_probs = F.log_softmax(logits, dim=1)
        losses = -(soft * log_probs).sum(dim=1)
        if self.weight is not None:
            losses = losses * self.weight.gather(0, targets)
        return losses.mean()


def make_criterion(
    name: str,
    weight: Optional[torch.Tensor] = None,
    *,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    ordinal_epsilon: float = 0.1,
) -> nn.Module:
    """Build the requested criterion. ``weight`` must be on the training device."""
    if name == LOSS_CE:
        try:
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing))
        except TypeError as exc:
            raise RuntimeError(
                "nn.CrossEntropyLoss(label_smoothing=...) is unsupported by this PyTorch; "
                "upgrade or use --loss focal / --loss ordinal_smooth with their own settings."
            ) from exc
    if name in (LOSS_FOCAL, LOSS_ORDINAL_SMOOTH):
        if float(label_smoothing) != 0.0:
            raise ValueError(
                f"--loss {name} defines its own target smoothing; pass --label_smoothing 0 "
                "with it (or use --loss ce for plain label smoothing)."
            )
    if name == LOSS_FOCAL:
        return FocalLoss(weight=weight, gamma=focal_gamma)
    if name == LOSS_ORDINAL_SMOOTH:
        return OrdinalSmoothLoss(weight=weight, epsilon=ordinal_epsilon)
    raise ValueError(f"Unknown loss {name!r}; choose from {list(SUPPORTED_LOSSES)}.")
