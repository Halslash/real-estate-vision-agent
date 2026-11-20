"""Optional diagnostic: logistic regression on frozen backbone features.

This is a DECISION TOOL, not part of normal training. It answers one question:
is the modest CNN validation score caused by the training recipe or by the
features/labels themselves?

  * Probe ~= CNN validation macro F1  -> feature/label ceiling; more tuning of
    the classifier head has low expected value.
  * Probe >> CNN validation macro F1  -> training recipe is the bottleneck.

Only the saved TRAIN split is used to fit the probe; the saved VAL split is used
to score it. The TEST split is never touched. Results are written to a separate
JSON file and no model weights are modified.

Example (Kaggle):
    !python src/probe.py \
        --output_dir /kaggle/working/output/improved_mobilenet \
        --checkpoint /kaggle/working/output/improved_mobilenet/models/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

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
from src.model import (  # noqa: E402
    build_model_from_checkpoint,
    create_model,
    get_architecture,
    load_checkpoint,
)
from src.train import resolve_device, set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-feature logistic-regression probe (train -> val only)."
    )
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint supplying the saved split, stats and preprocessing.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--price_col", type=str, default=None)
    parser.add_argument("--sqft_col", type=str, default=None)
    parser.add_argument("--city_col", type=str, default=None)
    parser.add_argument("--feature_source", type=str, default="pretrained",
                        choices=["pretrained", "checkpoint"],
                        help="'pretrained' tests ImageNet features from the "
                             "checkpoint's architecture; 'checkpoint' uses the "
                             "trained backbone.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing probe result file.")
    return parser


def _probe_output_path(args) -> str:
    """Distinct filename per feature source; the default keeps its legacy name."""
    filename = ("probe_results.json" if args.feature_source == "pretrained"
                else f"probe_results_{args.feature_source}.json")
    return os.path.join(args.output_dir, "results", filename)


def _locate_checkpoint(args) -> str:
    path = args.checkpoint or os.path.join(args.output_dir, "models", "best_model.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first or pass --checkpoint."
        )
    return path


def _locate_split(checkpoint_path: str, output_dir: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "split_indices.json"),
        os.path.join(output_dir, "models", "split_indices.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "split_indices.json was not found next to the checkpoint nor in "
        f"{os.path.join(output_dir, 'models')}. Run train.py first."
    )


def extract_features(
    model: torch.nn.Module,
    records: List[Dict[str, Any]],
    preprocessing: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pooled frozen features and labels for the given records."""
    dataset = HouseImageDataset(records, transform=build_eval_transform(preprocessing))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    features: List[np.ndarray] = []
    labels: List[int] = []
    context = (
        torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    )
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            # MobileNetV2/EfficientNet-B0: features -> global average pool -> vector.
            output = model.features(images)
            pooled = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(output, 1), 1)
            features.append(pooled.detach().cpu().numpy())
            labels.extend(targets.tolist())
    if not features:
        raise RuntimeError("No features were extracted; check the split and data paths.")
    return np.vstack(features), np.asarray(labels, dtype=int)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    probe_output = _probe_output_path(args)
    if os.path.exists(probe_output) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing probe result: {probe_output}. "
            "Use a new --output_dir or pass --force."
        )

    checkpoint_path = _locate_checkpoint(args)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    stats = checkpoint["label_stats"]
    preprocessing = checkpoint["preprocessing"]

    split_path = _locate_split(checkpoint_path, args.output_dir)
    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_ids, val_ids = set(split["train"]), set(split["val"])
    if not train_ids or not val_ids:
        raise ValueError("The saved train/validation splits must both be non-empty.")

    detected = stats.get("columns", {})
    config = checkpoint.get("train_config", {})
    records, skipped, _ = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col or detected.get("id") or config.get("id_col"),
        price_col=args.price_col or detected.get("price") or config.get("price_col"),
        sqft_col=args.sqft_col or detected.get("sqft") or config.get("sqft_col"),
        city_col=args.city_col or detected.get("city") or config.get("city_col"),
    )
    train_records = [r for r in records if r["image_id"] in train_ids]
    val_records = [r for r in records if r["image_id"] in val_ids]
    if not train_records or not val_records:
        raise RuntimeError(
            "No records matched the saved split. Ensure --csv_path and --images_dir "
            "match the training run."
        )
    compute_labels(train_records, stats)
    compute_labels(val_records, stats)
    print(f"Skipped records while reloading data: {skipped}")
    print(f"Probe data -> train: {len(train_records)}, val: {len(val_records)} "
          "(test split is NOT used)")

    if args.feature_source == "pretrained":
        architecture = get_architecture(checkpoint)
        print(f"Loading ImageNet-pretrained {architecture} for frozen feature extraction...")
        model = create_model(
            architecture, num_classes=int(checkpoint.get("num_classes", 3)), pretrained=True
        ).to(device)
    else:
        print("Rebuilding the checkpoint backbone for frozen feature extraction...")
        model = build_model_from_checkpoint(checkpoint, device=device)
    model.eval()

    print("Extracting training features...")
    x_train, y_train = extract_features(
        model, train_records, preprocessing, device, args.batch_size, args.num_workers
    )
    print("Extracting validation features...")
    x_val, y_val = extract_features(
        model, val_records, preprocessing, device, args.batch_size, args.num_workers
    )
    print(f"Feature shape: {x_train.shape[1]}-dim; fitting standardized logistic regression...")

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, random_state=args.seed)),
    ])
    classifier.fit(x_train, y_train)
    val_pred = classifier.predict(x_val)

    labels_order = list(range(int(checkpoint.get("num_classes", 3))))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_val, val_pred, average="macro", labels=labels_order, zero_division=0
    )
    results = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(
            y_val, val_pred, average="weighted", labels=labels_order, zero_division=0
        )),
        "feature_source": args.feature_source,
        "architecture": get_architecture(checkpoint),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "feature_dim": int(x_train.shape[1]),
        "checkpoint": checkpoint_path,
        "label_config_version": stats.get("label_config_version", "v1_legacy"),
        "seed": args.seed,
        "used_test_split": False,
        "interpretation": (
            "Compare val macro F1 with the CNN's val macro F1 (see training summary). "
            "Near-equal => likely feature/label ceiling; much higher => likely "
            "training-recipe limitation. The target remains a market-derived proxy."
        ),
    }

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(probe_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Probe validation accuracy: {results['accuracy']:.4f}")
    print(f"Probe validation macro F1: {results['macro_f1']:.4f}")
    print(f"Probe results saved in: {probe_output}")


if __name__ == "__main__":
    main()
