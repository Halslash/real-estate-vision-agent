"""Two-phase transfer-learning training for Member A's visual-quality CNN.

Phase 1: frozen backbone + trainable classification head (AdamW, LR=--lr).
Phase 2: unfreeze the last blocks, lower LR (--lr_finetune), cosine decay,
         continue with early stopping on validation macro F1.

Scientific note: the target is a MARKET-DERIVED VISUAL PROXY (see dataset.py),
not a physical-condition label and not a price estimate.

Run from the Kaggle repository root, e.g.
    !python src/train.py --epochs 20 --batch_size 64 \
        --output_dir /kaggle/working/output/improved_mobilenet
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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
    AUGMENTATION_IMPROVED,
    AUGMENTATION_LEGACY,
    DEFAULT_AUGMENTATION,
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_LABEL_CONFIG_VERSION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHRINKAGE_K,
    IDX_TO_LABEL,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_V1,
    LABEL_V2,
    NUM_CLASSES,
    PREPROCESSING_VERSION,
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
    SUPPORTED_ARCHITECTURES,
    ARCH_MOBILENET_V2,
    count_trainable_parameters,
    create_model,
    freeze_backbone,
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
    unfreeze_last_blocks,
)
from src.losses import (  # noqa: E402
    CLASS_WEIGHT_AUTO,
    CLASS_WEIGHT_NONE,
    LOSS_CE,
    SUPPORTED_CLASS_WEIGHTING,
    SUPPORTED_LOSSES,
    compute_class_weights,
    make_criterion,
)
from src.plots import plot_learning_curves  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Member A's MobileNetV2 / EfficientNet-B0 visual proxy model."
    )
    # Preserved Kaggle-facing arguments.
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20,
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
                        help="Number of final backbone blocks unfrozen in Phase 2.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--imbalance_ratio", type=float, default=1.5,
                        help="Use weighted cross-entropy above this max/min weight ratio.")
    parser.add_argument("--seed", type=int, default=42)

    # New configurable training / methodology options.
    parser.add_argument("--architecture", type=str, default=ARCH_MOBILENET_V2,
                        choices=list(SUPPORTED_ARCHITECTURES))
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Model input resolution (Kaggle default: 224).")
    parser.add_argument("--resize_size", type=int, default=RESIZE_SIZE,
                        help="Resize short side before center crop (default: 256).")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing (0 disables it).")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"],
                        help="Per-phase learning-rate schedule.")
    parser.add_argument("--label_config_version", type=str,
                        default=DEFAULT_LABEL_CONFIG_VERSION, choices=[LABEL_V1, LABEL_V2],
                        help="Proxy-label methodology; v2 adds log-ratio + shrunk city medians.")
    parser.add_argument("--shrinkage_k", type=float, default=DEFAULT_SHRINKAGE_K,
                        help="Pseudo-count for shrinking city medians to the global median (v2 only).")
    parser.add_argument("--augmentation", type=str, default=DEFAULT_AUGMENTATION,
                        choices=[AUGMENTATION_IMPROVED, AUGMENTATION_LEGACY],
                        help="Training augmentation recipe (improved removes rotation/crop).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="CPU fallback is supported.")

    # Controlled-experiment options. Defaults reproduce the existing improved run.
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional saved split_indices.json. When given, the split "
                             "is reused verbatim instead of regenerated (needed for OOF folds).")
    parser.add_argument("--loss", type=str, default=LOSS_CE, choices=list(SUPPORTED_LOSSES),
                        help="Training loss; 'ce' keeps the existing cross-entropy recipe.")
    parser.add_argument("--class_weight", type=str, default=CLASS_WEIGHT_AUTO,
                        choices=list(SUPPORTED_CLASS_WEIGHTING),
                        help="'auto' is the original inverse-frequency rule; 'none', "
                             "'balanced' and 'effective_number' are controlled alternatives.")
    parser.add_argument("--class_weight_beta", type=float, default=0.9999,
                        help="Beta for --class_weight effective_number.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--ordinal_epsilon", type=float, default=0.1,
                        help="Adjacent-class mass for --loss ordinal_smooth.")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Select CPU or CUDA with an explicit, actionable error for bad requests."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Use --device auto or --device cpu for CPU fallback."
        )
    return torch.device(requested)


def make_lr_scheduler(optimizer, name: str, total_epochs: int):
    """One scheduler per phase, stepped once per epoch.

    Cosine decay lets the validation metric settle near the end of a longer
    fine-tuning phase instead of stopping at a still-high learning rate.
    """
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(total_epochs)), eta_min=0.0
        )
    raise ValueError(f"Unknown scheduler {name!r}; choose 'cosine' or 'none'.")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    training: bool,
) -> Dict[str, Any]:
    """Run one epoch and return loss, accuracy, macro F1 and raw predictions."""
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
        # Fixed 3-label support: a class missing from one epoch's minibatches
        # must not silently change the meaning of the selection metric.
        "f1_macro": float(f1_score(
            y_true, y_pred, average="macro",
            labels=list(range(NUM_CLASSES)), zero_division=0,
        )),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.image_size < 32 or args.resize_size < args.image_size:
        raise ValueError(
            "--image_size must be >= 32 and --resize_size must be >= --image_size "
            f"(got {args.image_size}, {args.resize_size})."
        )
    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        raise ValueError("--label_smoothing must be in [0, 1).")
    if args.loss != LOSS_CE and args.label_smoothing != 0.0:
        raise ValueError(
            f"--loss {args.loss} defines its own target smoothing; pass "
            "--label_smoothing 0 with it (or keep --loss ce)."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Using device: {device}")
    print(f"Architecture: {args.architecture} | label config: {args.label_config_version}")

    model_dir = os.path.join(args.output_dir, "models")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- data, split, proxy labels -----------------------------------------
    print("Loading CSV and verifying image files (no pixel decoding yet)...")
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

    if args.split_file:
        if not os.path.isfile(args.split_file):
            raise FileNotFoundError(f"--split_file not found: {args.split_file}")
        with open(args.split_file, "r", encoding="utf-8") as handle:
            split = json.load(handle)
        for name in ("train", "val", "test"):
            if not isinstance(split.get(name), list):
                raise ValueError(f"--split_file is missing a '{name}' id list.")
        if not split["train"] or not split["val"]:
            raise ValueError("--split_file must have non-empty train and val lists.")
        known = {r["image_id"] for r in records}
        for name in ("train", "val", "test"):
            unknown = set(split[name]) - known
            if unknown:
                raise ValueError(
                    f"--split_file '{name}' contains ids absent from CSV/images "
                    f"(e.g. {sorted(unknown)[:5]}). Check --csv_path/--images_dir."
                )
        print(f"Reusing split from {args.split_file} "
              f"(train={len(split['train'])}, val={len(split['val'])}, "
              f"test={len(split['test'])})")
    else:
        split = make_stratified_split(records, seed=args.seed)
    train_ids = set(split["train"])
    train_records = [r for r in records if r["image_id"] in train_ids]

    # Statistics are fitted on TRAIN data only, then applied to all splits.
    stats = fit_preprocessing_stats(
        train_records,
        label_config_version=args.label_config_version,
        shrinkage_k=args.shrinkage_k,
    )
    stats["columns"] = detected
    compute_labels(records, stats)
    by_split = split_records(records, split)
    validate_labeled_split(by_split, require_test=bool(split["test"]))
    print_summary(records, skipped, split, by_split)

    preprocessing = {
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "architecture": args.architecture,
        "pretrained_backbone": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "augmentation": args.augmentation,
    }

    # ---- loaders ------------------------------------------------------------
    print("Building data loaders...")
    train_dataset = HouseImageDataset(
        by_split["train"],
        transform=build_train_transform(args.image_size, args.resize_size, args.augmentation),
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

    # ---- loss ---------------------------------------------------------------
    counts = np.bincount(
        [r["label"] for r in by_split["train"]], minlength=NUM_CLASSES
    )
    if counts.min() == 0:
        raise ValueError(
            "The training split contains no sample of at least one proxy class. "
            "Increase --max_images or change the dataset; classes: "
            + str(counts.tolist())
        )
    weight_values, weighted = compute_class_weights(
        counts,
        args.class_weight,
        beta=args.class_weight_beta,
        imbalance_ratio=args.imbalance_ratio,
    )
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    criterion = make_criterion(
        args.loss,
        class_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        ordinal_epsilon=args.ordinal_epsilon,
    )
    print(
        f"Loss: {args.loss} | class weighting: {args.class_weight} "
        f"(active={weighted}, counts={counts.tolist()}, "
        f"weights={np.round(weight_values, 3).tolist()})"
    )
    print(f"Label smoothing: {args.label_smoothing} | LR schedule: {args.scheduler}")

    # ---- model / phases -----------------------------------------------------
    print("Loading ImageNet pretrained backbone (this may take a moment)...")
    model = create_model(
        args.architecture, num_classes=NUM_CLASSES, pretrained=True
    ).to(device)
    freeze_backbone(model)
    print(f"Phase 1 trainable parameters: {count_trainable_parameters(model):,}")

    phase1_epochs = max(1, args.epochs // 2)
    phase2_epochs = max(0, args.epochs - phase1_epochs)
    print(f"Planned epochs -> phase 1: {phase1_epochs}, phase 2: {phase2_epochs}")

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0
    early_stopped = False
    checkpoint_path = os.path.join(model_dir, "best_model.pth")

    train_config: Dict[str, Any] = dict(vars(args))
    train_config.update({
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "weighted_loss": weighted,
        "class_counts": counts.tolist(),
        "split_source": args.split_file or "generated",
        "loss": args.loss,
        "class_weight": args.class_weight,
        "class_weights": weight_values.tolist(),
        "class_weight_beta": args.class_weight_beta,
        "focal_gamma": args.focal_gamma,
        "ordinal_epsilon": args.ordinal_epsilon,
        "detected_columns": detected,
        "resolved_device": str(device),
        "label_config_version": args.label_config_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "augmentation": args.augmentation,
        "early_stopped": False,  # updated in the final checkpoint rewrite
        "stopped_epoch": None,
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
            arch=args.architecture,
            val_metrics=best_val_metrics,
        )
        save_checkpoint(checkpoint_path, checkpoint)

    def run_phase(
        name: str, epochs: int, optimizer, early_stopping: bool = False
    ) -> bool:
        """Train one phase. Early stopping is enabled only for Phase 2."""
        nonlocal best_f1, best_epoch, best_val_metrics, patience_counter, early_stopped
        scheduler = make_lr_scheduler(optimizer, args.scheduler, epochs)
        for _ in range(epochs):
            epoch_number = len(history) + 1
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, device, False
            )
            scheduler.step()
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
                best_val_metrics = {
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "macro_f1": val_metrics["f1_macro"],
                }
                patience_counter = 0
                save_best(epoch_number, best_f1)
                print(f"  -> new best model saved (val macro F1 {best_f1:.4f})")
            elif early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    early_stopped = True
                    print(
                        f"  -> early stopping after {args.patience} epochs "
                        "without improvement"
                    )
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
            f"Phase 2: unfrozen backbone blocks {first_unfrozen}..{len(model.features) - 1}"
        )
        print(f"Phase 2 trainable parameters: {count_trainable_parameters(model):,}")
        optimizer2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
        )
        run_phase("phase2_finetune", phase2_epochs, optimizer2, early_stopping=True)

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was saved; check the validation metrics.")

    # ---- persist all artifacts ---------------------------------------------
    # Update the best checkpoint's metadata in place (weights are untouched) so
    # it carries the FULL history and the final early-stopping decision.
    final_checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    final_checkpoint["history"] = history
    final_checkpoint["early_stopped"] = bool(early_stopped)
    final_checkpoint["train_config"].update({
        "early_stopped": bool(early_stopped),
        "stopped_epoch": len(history),
        "best_epoch": best_epoch,
    })
    save_checkpoint(checkpoint_path, final_checkpoint)

    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, "training_history.csv"), index=False
    )
    plot_learning_curves(history, os.path.join(results_dir, "learning_curves.png"))

    def dump_json(filename: str, payload) -> None:
        with open(os.path.join(model_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    dump_json("model_config.json", final_checkpoint["train_config"])
    dump_json("preprocessing_stats.json", stats)
    dump_json("class_thresholds.json", {
        "label_config_version": stats.get("label_config_version", LABEL_V1),
        "class_threshold_low": stats["class_threshold_low"],
        "class_threshold_high": stats["class_threshold_high"],
        "clip_log_p1": stats.get("clip_log_p1"),
        "clip_log_p99": stats.get("clip_log_p99"),
        "label_names": stats["label_names"],
        "num_classes": stats["num_classes"],
    })
    dump_json("city_medians.json", {
        "label_config_version": stats.get("label_config_version", LABEL_V1),
        "city_raw_log_median": stats.get("city_raw_log_median", {}),
        "city_shrunk_log_median": stats.get("city_shrunk_log_median", {}),
        "city_counts": stats.get("city_counts", {}),
        "shrinkage_k": stats.get("shrinkage_k"),
        "global_median_log": stats.get("global_median_log"),
        # v1 fields are included unchanged when applicable for readability.
        "city_median_pps": stats.get("city_median_pps", {}),
        "global_median_pps": stats.get("global_median_pps"),
        "note": "Fitted on the training split only; unseen cities use the global median.",
    })
    dump_json("split_indices.json", {name: split[name] for name in ("train", "val", "test")})
    dump_json("preprocessing_config.json", preprocessing)
    duration_seconds = time.perf_counter() - train_started
    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
    dump_json("training_summary.json", {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "final_val_metrics": best_val_metrics,
        "training_duration_seconds": round(duration_seconds, 1),
        "peak_gpu_memory_mb": peak_vram_mb,
        "resolved_device": str(device),
        "early_stopped": bool(early_stopped),
        "total_epochs_run": len(history),
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "label_config_version": args.label_config_version,
        "architecture": args.architecture,
    })

    print(f"Training complete. Best epoch: {best_epoch} (val macro F1 {best_f1:.4f})")
    print(f"Early stopped: {early_stopped} | total epochs run: {len(history)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Artifacts: {model_dir} and {results_dir}")


if __name__ == "__main__":
    main()
