"""Shared MobileNetV2 architecture and checkpoint utilities.

Used by train.py (saving), evaluate.py (reconstruction) and vision_agent.py
(inference). The checkpoint is self-contained: it stores architecture metadata,
the class mapping, preprocessing configuration and all proxy-label statistics.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

ARCH_NAME = "mobilenet_v2"


def create_mobilenet_v2(num_classes: int = 3, pretrained: bool = True, progress: bool = True):
    """Build MobileNetV2 with a fresh three-class classification head.

    Transfer learning is mandatory in this project. If ImageNet weights cannot
    be downloaded we raise a clear error instead of silently training a random
    network. Inference passes ``pretrained=False`` because it loads trained
    weights from the checkpoint and needs no download.
    """
    if pretrained:
        try:
            try:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
                model = models.mobilenet_v2(weights=weights, progress=progress)
            except AttributeError:  # older torchvision versions
                model = models.mobilenet_v2(pretrained=True, progress=progress)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ImageNet pretrained weights for MobileNetV2. "
                "This project requires transfer learning and will NOT fall back "
                "to random initialization. Check Kaggle internet access or that "
                "torchvision weights are available. Original error: " + str(exc)
            ) from exc
    else:
        try:
            model = models.mobilenet_v2(weights=None, progress=progress)
        except TypeError:  # older torchvision versions
            model = models.mobilenet_v2(pretrained=False, progress=progress)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: freeze every backbone block, train only the classification head."""
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 4) -> int:
    """Phase 2: unfreeze the last ``n_blocks`` MobileNetV2 blocks.

    Earlier blocks stay frozen. Returns the index of the first unfrozen block.
    """
    total_blocks = len(model.features)
    start = max(0, total_blocks - int(n_blocks))
    for index, block in enumerate(model.features):
        for parameter in block.parameters():
            parameter.requires_grad_(index >= start)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    return start


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def make_checkpoint(
    model: nn.Module,
    *,
    idx_to_label: Dict[int, str],
    preprocessing: Dict[str, Any],
    label_stats: Dict[str, Any],
    train_config: Dict[str, Any],
    epoch: int,
    val_macro_f1: float,
    history: Any,
) -> Dict[str, Any]:
    """Assemble a self-contained, CPU-resident checkpoint dictionary."""
    return {
        "arch": ARCH_NAME,
        "num_classes": len(idx_to_label),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "idx_to_label": {int(i): str(name) for i, name in idx_to_label.items()},
        "class_to_idx": {str(name): int(i) for i, name in idx_to_label.items()},
        "preprocessing": dict(preprocessing),
        "label_stats": label_stats,
        "train_config": train_config,
        "epoch": int(epoch),
        "val_macro_f1": float(val_macro_f1),
        "history": history,
        "pytorch_version": torch.__version__,
    }


def save_checkpoint(path: str, checkpoint: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions without the weights_only argument
        return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any], device: Optional[torch.device] = None
) -> nn.Module:
    """Rebuild the exact architecture used in training and load its weights."""
    if checkpoint.get("arch", ARCH_NAME) != ARCH_NAME:
        raise ValueError(
            f"Unsupported checkpoint architecture: {checkpoint.get('arch')!r}"
        )
    model = create_mobilenet_v2(
        num_classes=int(checkpoint["num_classes"]), pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
