"""Two-phase transfer-learning training for Member A's visual-quality CNN.

Phase 1: frozen MobileNetV2 backbone + trainable classification head (AdamW, LR=--lr).
Phase 2: unfreeze the last blocks, lower LR (--lr_finetune), continue with early stopping.

Run from the Kaggle repository root, e.g.
    !python src/train.py --epochs 10 --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Allow `python src/train.py` (script mode) and `from src.train import ...` both to work.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_DIR,
    IDX_TO_LABEL,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    RESIZE_SIZE,
    HouseImageDataset,
    build_eval_transform,
    build_train_transform,
    compute_labels,
    fit_preprocessing_stats,
    load_and_validate_records,
    make_stratified_split,
    print_summary,
    split_records,
    validate_labeled_split,
)
from src.model import (  # noqa: E402
    count_trainable_parameters,
    create_mobilenet_v2,
    freeze_backbone,
    make_checkpoint,
    save_checkpoint,
    unfreeze_last_blocks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Member A's MobileNetV2 visual-quality proxy model."
    )
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total epochs across both phases (split roughly in half).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Cap on valid images; intended for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Phase 1 learning rate (classification head).")
    parser.add_argument("--lr_finetune", type=float, default=1e-5,
                        help="Phase 2 learning rate (lower, for fine-tuning).")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--price_col", type=str, default=None)
    parser.add_argument("--sqft_col", type=str, default=None)
    parser.add_argument("--city_col", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience in Phase 2 (validation macro F1).")
    parser.add_argument("--finetune_blocks", type=int, default=4,
                        help="Number of final MobileNetV2 blocks unfrozen in Phase 2.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--imbalance_ratio", type=float, default=1.5,
                        help="Use weighted cross-entropy above this max/min weight ratio.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    training: bool,
) -> Dict[str, Any]:
    """Run one epoch and return loss, accuracy and macro F1."""
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- data, split, proxy labels -----------------------------------------
    records, skipped, detected = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col,
        price_col=args.price_col,
        sqft_col=args.sqft_col,
        city_col=args.city_col,
        max_images=args.max_images,
    )
    if not records:
        raise RuntimeError("No valid samples were found; nothing to train on.")

    split = make_stratified_split(records, seed=args.seed)
    train_ids = set(split["train"])
    train_records = [r for r in records if r["image_id"] in train_ids]

    # Statistics are fitted on TRAIN data only, then applied to all splits.
    stats = fit_preprocessing_stats(train_records)
    stats["columns"] = detected
    compute_labels(records, stats)
    by_split = split_records(records, split)
    validate_labeled_split(by_split)
    print_summary(records, skipped, split, by_split)

    preprocessing = {
        "image_size": IMAGE_SIZE,
        "resize_size": RESIZE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "architecture": "mobilenet_v2",
        "pretrained_backbone": True,
    }

    # ---- loaders ------------------------------------------------------------
    train_dataset = HouseImageDataset(
        by_split["train"], transform=build_train_transform(IMAGE_SIZE)
    )
    val_dataset = HouseImageDataset(
        by_split["val"], transform=build_eval_transform(preprocessing)
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    # ---- loss (weighted only when imbalance is significant) -----------------
    counts = np.bincount(
        [r["label"] for r in by_split["train"]], minlength=NUM_CLASSES
    )
    if counts.min() == 0:
        raise ValueError(
            "The training split contains no sample of at least one proxy class. "
            "Increase --max_images or change the dataset; classes: "
            + str(counts.tolist())
        )
    weight_values = counts.sum() / (NUM_CLASSES * counts)
    weighted = bool(weight_values.max() / weight_values.min() > args.imbalance_ratio)
    class_weights = torch.tensor(
        weight_values if weighted else np.ones(NUM_CLASSES, dtype=float),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted CrossEntropyLoss: {weighted} (class counts: {counts.tolist()})")

    # ---- model / phases -----------------------------------------------------
    model = create_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True).to(device)
    freeze_backbone(model)
    print(f"Phase 1 trainable parameters: {count_trainable_parameters(model):,}")

    phase1_epochs = max(1, args.epochs // 2)
    phase2_epochs = max(0, args.epochs - phase1_epochs)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(model_dir, "best_model.pth")

    train_config = dict(vars(args))
    train_config.update({
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "weighted_loss": weighted,
        "class_counts": counts.tolist(),
        "detected_columns": detected,
        "device": str(device),
    })

    def save_best(epoch_number: int, score: float) -> None:
        checkpoint = make_checkpoint(
            model,
            idx_to_label=IDX_TO_LABEL,
            preprocessing=preprocessing,
            label_stats=stats,
            train_config=train_config,
            epoch=epoch_number,
            val_macro_f1=score,
            history=history,
        )
        save_checkpoint(checkpoint_path, checkpoint)

    def run_phase(name: str, epochs: int, optimizer, early_stopping: bool = False) -> bool:
        """Train one phase. Early stopping is enabled only for Phase 2."""
        nonlocal best_f1, best_epoch, patience_counter
        for _ in range(epochs):
            epoch_number = len(history) + 1
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, device, False
            )
            history.append({
                "epoch": epoch_number,
                "phase": name,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["f1_macro"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            print(
                f"Epoch {epoch_number} [{name}] | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
                f"macro F1 {val_metrics['f1_macro']:.4f}"
            )

            if val_metrics["f1_macro"] > best_f1 + 1e-6:
                best_f1 = val_metrics["f1_macro"]
                best_epoch = epoch_number
                patience_counter = 0
                save_best(epoch_number, best_f1)
                print(f"  -> new best model saved (val macro F1 {best_f1:.4f})")
            elif early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  -> early stopping after {args.patience} epochs without improvement")
                    return True
        return False

    optimizer1 = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Phase 1 always runs in full so Phase 2 is never skipped on a normal run.
    run_phase("phase1_head", phase1_epochs, optimizer1, early_stopping=False)

    if phase2_epochs > 0:
        first_unfrozen = unfreeze_last_blocks(model, args.finetune_blocks)
        print(
            f"Phase 2: unfrozen MobileNetV2 blocks {first_unfrozen}..{len(model.features) - 1}"
        )
        print(f"Phase 2 trainable parameters: {count_trainable_parameters(model):,}")
        optimizer2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
        )
        run_phase("phase2_finetune", phase2_epochs, optimizer2)

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was saved; check the validation metrics.")

    # ---- persist all artifacts ---------------------------------------------
    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, "training_history.csv"), index=False
    )

    def dump_json(filename: str, payload) -> None:
        with open(os.path.join(model_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    dump_json("model_config.json", train_config)
    dump_json("preprocessing_stats.json", stats)
    dump_json("class_thresholds.json", {
        "class_threshold_low": stats["class_threshold_low"],
        "class_threshold_high": stats["class_threshold_high"],
        "clip_p1": stats["clip_p1"],
        "clip_p99": stats["clip_p99"],
        "label_names": stats["label_names"],
        "num_classes": stats["num_classes"],
    })
    dump_json("city_medians.json", {
        "city_median_pps": stats["city_median_pps"],
        "global_median_pps": stats["global_median_pps"],
        "note": "Fitted on the training split only; unseen cities use the global median.",
    })
    dump_json("split_indices.json", {name: split[name] for name in ("train", "val", "test")})
    dump_json("preprocessing_config.json", preprocessing)

    print(f"Training complete. Best epoch: {best_epoch} (val macro F1 {best_f1:.4f})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Artifacts: {model_dir} and {results_dir}")


if __name__ == "__main__":
    main()
