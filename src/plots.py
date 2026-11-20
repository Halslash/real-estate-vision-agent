"""Small shared plotting helpers (headless / Kaggle-safe)."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # headless plotting on Kaggle
import matplotlib.pyplot as plt  # noqa: E402


def plot_learning_curves(history: Sequence[Dict[str, Any]], path: str) -> None:
    """Plot loss and accuracy/macro-F1 from a training history.

    Only recorded values are plotted; missing curves are skipped rather than
    filled with invented numbers.
    """
    if not history:
        raise ValueError("Cannot plot learning curves from an empty training history.")

    epochs = [int(row.get("epoch", i + 1)) for i, row in enumerate(history)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for key, label in (("train_loss", "train loss"), ("val_loss", "val loss")):
        values = [row[key] for row in history if key in row]
        if values:
            axes[0].plot(epochs[: len(values)], values, marker="o", label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key, label in (
        ("train_accuracy", "train acc"),
        ("val_accuracy", "val acc"),
        ("val_macro_f1", "val macro F1"),
    ):
        values = [row[key] for row in history if key in row]
        if values:
            axes[1].plot(epochs[: len(values)], values, marker="o", label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
