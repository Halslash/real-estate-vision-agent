"""Evaluate the saved CNN on the untouched test split.

The test split is used ONLY here -- never for early stopping, model selection or
statistics fitting. All metadata (split, columns, label thresholds, city
statistics, preprocessing, architecture) is read from the checkpoint and its
sibling JSON files; the split is never regenerated.

Run from the Kaggle repository root, e.g.
    !python src/evaluate.py \
        --output_dir /kaggle/working/output/improved_mobilenet
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Sequence

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
    LABEL_V1,
    LABEL_V2,
    LABEL_V3,
    HouseImageDataset,
    build_eval_transform,
    compute_labels,
    load_and_validate_records,
)
from src.model import get_architecture, build_model_from_checkpoint, load_checkpoint  # noqa: E402
from src.plots import plot_learning_curves  # noqa: E402
from src.train import resolve_device  # noqa: E402

PROXY_WARNING = (
    "These are market-derived visual proxy labels, not human-annotated physical "
    "condition or aesthetic-quality labels. Price per square foot is confounded "
    "by location, neighbourhood, demand, property characteristics and market "
    "timing. A higher score does NOT prove better physical condition, and this "
    "model does NOT predict the final property price."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Member A's CNN on the held-out test split."
    )
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Informational only during evaluation.")
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
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override the checkpoint path (default: <output_dir>/models/best_model.pth).")
    return parser


def _normalize_confusion(matrix: np.ndarray, axis: int) -> np.ndarray:
    sums = matrix.sum(axis=axis, keepdims=True)
    return np.divide(
        matrix.astype(float), sums, out=np.zeros_like(matrix, dtype=float),
        where=sums != 0,
    )


def plot_confusion_matrix(matrix: np.ndarray, names: Sequence[str], path: str,
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


def plot_prediction_distribution(
    true_counts: np.ndarray, pred_counts: np.ndarray, names: Sequence[str], path: str
) -> None:
    """Compare true vs predicted class counts to expose prior collapse."""
    x_positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x_positions - width / 2, true_counts, width, label="true")
    axis.bar(x_positions + width / 2, pred_counts, width, label="predicted")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(names)
    axis.set_xlabel("Proxy class")
    axis.set_ylabel("Number of test images")
    axis.set_title("True vs predicted class distribution (test split)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def compute_baselines(y_true: Sequence[int], names: Sequence[str], seed: int) -> Dict[str, Any]:
    """Honest, computed reference points for this exact test split."""
    y_true_array = np.asarray(y_true, dtype=int)
    num_classes = len(names)
    counts = np.bincount(y_true_array, minlength=num_classes)

    majority_index = int(np.argmax(counts))
    majority_pred = np.full(len(y_true_array), majority_index, dtype=int)
    majority = {
        "accuracy": float(accuracy_score(y_true_array, majority_pred)),
        "macro_f1": float(f1_score(
            y_true_array, majority_pred, average="macro",
            labels=list(range(num_classes)), zero_division=0,
        )),
        "predicted_class": names[majority_index],
    }

    rng = np.random.default_rng(seed)
    random_pred = rng.integers(0, num_classes, size=len(y_true_array))
    random_baseline = {
        "accuracy": float(accuracy_score(y_true_array, random_pred)),
        "macro_f1": float(f1_score(
            y_true_array, random_pred, average="macro",
            labels=list(range(num_classes)), zero_division=0,
        )),
        "analytical_uniform_accuracy": float(1.0 / num_classes),
        "seed": seed,
    }
    return {"majority_class": majority, "uniform_random": random_baseline}


def build_markdown_report(metrics: Dict[str, Any]) -> str:
    """Human-readable evaluation report with explicit interpretation warnings."""
    lines: List[str] = ["# Evaluation report — Member A visual proxy CNN", ""]
    lines.append(f"- Evaluation date: `{metrics['evaluation_date']}`")
    lines.append(f"- Checkpoint: `{metrics['checkpoint']}`")
    lines.append(f"- Architecture: `{metrics['architecture']}`")
    lines.append(f"- Label configuration version: `{metrics['label_config_version']}`")
    lines.append(f"- Dataset size (valid records loaded): {metrics['dataset_size']}")
    lines.append(f"- Split sizes: {metrics['split_sizes']}")
    lines.append(f"- Best epoch: {metrics['best_epoch']}")
    lines.append(f"- Early stopping used: {metrics['early_stopped']}")
    lines.append("")
    lines.append("> **Interpretation warning.** " + PROXY_WARNING)
    lines.append("")
    lines.append("## Headline metrics (untouched test split)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        lines.append(f"| {key} | {metrics[key]:.4f} |")
    lines.append("")
    lines.append("## Reference baselines (computed on this split)")
    lines.append("")
    lines.append("| Baseline | Accuracy | Macro F1 |")
    lines.append("|---|---:|---:|")
    for name, values in metrics["baselines"].items():
        lines.append(f"| {name} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} |")
    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, values in metrics["per_class"].items():
        lines.append(
            f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |"
        )
    lines.append("")
    lines.append(f"- True class distribution: {metrics['true_class_distribution']}")
    lines.append(f"- Predicted class distribution: {metrics['predicted_class_distribution']}")
    lines.append("")
    lines.append("## Confusion matrices")
    lines.append("")
    lines.append("Raw (rows = true, columns = predicted):")
    lines.append("")
    lines.append("```text")
    lines.append(" ".join(metrics["labels"]))
    for row in metrics["confusion_matrix"]:
        lines.append(" ".join(str(value) for value in row))
    lines.append("```")
    lines.append("")
    lines.append("Row-normalized (recall per class) and column-normalized matrices "
                 "are saved as PNG figures alongside this report.")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `confusion_matrix.png`")
    lines.append("- `normalized_confusion_matrix.png`")
    lines.append("- `column_normalized_confusion_matrix.png`")
    lines.append("- `prediction_distribution.png`")
    lines.append("- `learning_curves.png` (only when training history is available)")
    lines.append("- `metrics.json`, `classification_report.json`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Test metrics must never be used to select hyperparameters; "
                 "selection is based on validation macro F1 only.")
    lines.append("- The uniform/random baseline is a seeded empirical draw; its "
                 "analytical accuracy is 1/number_of_classes.")
    lines.append("- No calibration claim is made for the soft proxy score used by Member B.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

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
    label_version = stats.get("label_config_version", LABEL_V1)
    if label_version not in (LABEL_V1, LABEL_V2, LABEL_V3):
        raise ValueError(
            f"Unsupported label_config_version {label_version!r} in checkpoint. "
            "Use a package version that understands this label methodology rather "
            "than reinterpreting its thresholds."
        )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {get_architecture(checkpoint)} | label config: {label_version}")

    # The split belongs to the checkpoint, so look next to the checkpoint first.
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
    with inference_context:
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
    row_normalized = _normalize_confusion(matrix, axis=1)
    column_normalized = _normalize_confusion(matrix, axis=0)
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))

    per_class = {
        class_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i in labels_order
    }
    history = checkpoint.get("history", []) or []
    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": row_normalized.tolist(),
        "column_normalized_confusion_matrix": column_normalized.tolist(),
        "true_class_distribution": {
            class_names[i]: int(true_counts[i]) for i in labels_order
        },
        "predicted_class_distribution": {
            class_names[i]: int(pred_counts[i]) for i in labels_order
        },
        "baselines": compute_baselines(y_true, class_names, args.seed),
        "labels": class_names,
        "test_samples": len(test_records),
        "split_sizes": {name: len(split[name]) for name in ("train", "val", "test")},
        "dataset_size": len(records),
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
        "architecture": get_architecture(checkpoint),
        "label_config_version": label_version,
        "early_stopped": bool(checkpoint.get("early_stopped", False)),
        "evaluation_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "score_type": "market_derived_visual_proxy",
        "interpretation_warning": PROXY_WARNING,
    }
    report = classification_report(
        y_true, y_pred, labels=labels_order, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report["confusion_matrix"] = metrics["confusion_matrix"]
    report["normalized_confusion_matrix"] = metrics["normalized_confusion_matrix"]
    report["column_normalized_confusion_matrix"] = metrics["column_normalized_confusion_matrix"]
    report["baselines"] = metrics["baselines"]

    # ---- figures ------------------------------------------------------------
    plot_confusion_matrix(matrix, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    plot_confusion_matrix(
        row_normalized, class_names,
        os.path.join(results_dir, "normalized_confusion_matrix.png"),
        normalize=True,
    )
    plot_confusion_matrix(
        column_normalized, class_names,
        os.path.join(results_dir, "column_normalized_confusion_matrix.png"),
        normalize=True,
    )
    plot_prediction_distribution(
        true_counts, pred_counts, class_names,
        os.path.join(results_dir, "prediction_distribution.png"),
    )
    if history:
        plot_learning_curves(history, os.path.join(results_dir, "learning_curves.png"))
    else:
        print("Warning: checkpoint contains no training history; learning_curves.png skipped.")

    # ---- reports ------------------------------------------------------------
    for filename, payload in (
        ("classification_report.json", report),
        ("metrics.json", metrics),
    ):
        with open(os.path.join(results_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "evaluation_report.md"), "w", encoding="utf-8") as handle:
        handle.write(build_markdown_report(metrics))

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    for name, values in per_class.items():
        print(f"  {name:8s} precision={values['precision']:.4f} "
              f"recall={values['recall']:.4f} f1={values['f1']:.4f} "
              f"support={values['support']}")
    print("Baselines: " + ", ".join(
        f"{key}={value['accuracy']:.4f}/{value['macro_f1']:.4f}"
        for key, value in metrics["baselines"].items()
    ))
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    main()
