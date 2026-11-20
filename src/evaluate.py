"""Evaluate the saved CNN on the untouched test split.

The test split is used ONLY here -- never for early stopping or model selection.
Run from the Kaggle repository root, e.g.
    !python src/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless plotting on Kaggle
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_DIR,
    HouseImageDataset,
    build_eval_transform,
    compute_labels,
    load_and_validate_records,
)
from src.model import build_model_from_checkpoint, load_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Member A's CNN on the held-out test split."
    )
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Informational only during evaluation (matches training).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Usually unnecessary: the saved test split is reused as-is.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Informational only; the optimizer is not used here.")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--price_col", type=str, default=None)
    parser.add_argument("--sqft_col", type=str, default=None)
    parser.add_argument("--city_col", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override the checkpoint path (default: <output_dir>/models/best_model.pth).")
    return parser


def plot_confusion_matrix(matrix: np.ndarray, names: List[str], path: str,
                          normalize: bool = False) -> None:
    """Save a readable confusion-matrix figure."""
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(names)))
    axis.set_yticks(range(len(names)))
    axis.set_xticklabels(names)
    axis.set_yticklabels(names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    fmt = ".2f" if normalize else "d"
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column, row, format(value, fmt),
                ha="center", va="center",
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(model_dir, "best_model.pth")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    stats = checkpoint["label_stats"]
    preprocessing = checkpoint["preprocessing"]
    idx_to_label = {int(k): str(v) for k, v in checkpoint["idx_to_label"].items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    train_config = checkpoint.get("train_config", {})

    # The split belongs to the checkpoint, so look next to the checkpoint first.
    # By default that is <output_dir>/models/split_indices.json.
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    split_path = os.path.join(checkpoint_dir, "split_indices.json")
    if not os.path.isfile(split_path):
        split_path = os.path.join(model_dir, "split_indices.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            "Split file not found next to the checkpoint or in "
            f"{model_dir}. Run train.py first (it writes models/split_indices.json)."
        )
    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_ids = set(split["test"])
    if not test_ids:
        raise ValueError("The saved test split is empty.")

    # Rebuild ONLY the test records. Labels come from the saved training-time
    # statistics, and the saved column names are passed explicitly so detection
    # cannot drift between training and evaluation.
    detected = stats.get("columns", {})
    records, skipped, _ = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col or detected.get("id") or train_config.get("id_col"),
        price_col=args.price_col or detected.get("price") or train_config.get("price_col"),
        sqft_col=args.sqft_col or detected.get("sqft") or train_config.get("sqft_col"),
        city_col=args.city_col or detected.get("city") or train_config.get("city_col"),
        max_images=args.max_images,
    )
    test_records = [r for r in records if r["image_id"] in test_ids]
    if not test_records:
        raise RuntimeError(
            "No test records matched the saved split. Make sure --csv_path, "
            "--images_dir and --max_images match the training run."
        )
    compute_labels(test_records, stats)
    print(f"Skipped records while reloading data: {skipped}")
    print(f"Evaluating on {len(test_records)} untouched test images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model_from_checkpoint(checkpoint, device=device)

    dataset = HouseImageDataset(
        test_records, transform=build_eval_transform(preprocessing)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    inference_context = (
        torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    )
    with inference_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            y_true.extend(labels.tolist())

    labels_order = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels_order, zero_division=0
    )
    weighted_f1 = float(f1_score(
        y_true, y_pred, average="weighted", labels=labels_order, zero_division=0
    ))
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels_order, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels_order)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float), row_sums, out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    per_class = {
        class_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i in labels_order
    }
    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "labels": class_names,
        "test_samples": len(test_records),
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    report = classification_report(
        y_true, y_pred, labels=labels_order, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report["confusion_matrix"] = metrics["confusion_matrix"]
    report["normalized_confusion_matrix"] = metrics["normalized_confusion_matrix"]

    plot_confusion_matrix(matrix, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    plot_confusion_matrix(
        normalized, class_names,
        os.path.join(results_dir, "normalized_confusion_matrix.png"),
        normalize=True,
    )
    for filename, payload in (
        ("classification_report.json", report),
        ("metrics.json", metrics),
    ):
        with open(os.path.join(results_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    for name, values in per_class.items():
        print(f"  {name:8s} precision={values['precision']:.4f} "
              f"recall={values['recall']:.4f} f1={values['f1']:.4f} "
              f"support={values['support']}")
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    main()
