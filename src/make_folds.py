"""Build K OOF folds from the *canonical training split only*.

The untouched test split is never used for OOF training. Each fold writes a
``split_indices.json`` with train/val/test lists (test empty) so train.py can
consume it via ``--split_file``. This is a separate protocol from the main
experiments: folds are not comparable to the fixed 70/15/15 runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import KFold

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    load_and_validate_records,
    make_stratified_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output_dir", default="/kaggle/working/output/oof")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--price_col", default=None)
    parser.add_argument("--sqft_col", default=None)
    parser.add_argument("--city_col", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2.")

    records, skipped, detected = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col,
        price_col=args.price_col,
        sqft_col=args.sqft_col,
        city_col=args.city_col,
    )
    base_split = make_stratified_split(records, seed=args.seed, verbose=False)
    train_ids = np.asarray(sorted(base_split["train"]))
    if len(train_ids) <= args.n_folds:
        raise ValueError("Not enough canonical training rows for the requested fold count.")

    splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_root = os.path.join(args.output_dir, "folds")
    os.makedirs(fold_root, exist_ok=True)
    ownership: Dict[str, int] = {}

    for fold, (train_index, val_index) in enumerate(splitter.split(train_ids)):
        val_ids = sorted(train_ids[val_index].tolist())
        fold_train_ids = sorted(train_ids[train_index].tolist())
        for image_id in val_ids:
            if image_id in ownership:
                raise AssertionError(f"image_id {image_id} appears in more than one OOF fold")
            ownership[image_id] = fold
        payload = {
            "train": fold_train_ids,
            "val": val_ids,
            "test": [],
            "protocol": "oof",
            "fold": fold,
            "n_folds": args.n_folds,
            "source_split_seed": args.seed,
            "note": "Folds partition the canonical TRAIN split only; test is untouched.",
        }
        path = os.path.join(fold_root, f"fold_{fold}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"fold {fold}: train={len(fold_train_ids)} val={len(val_ids)} -> {path}")

    index_path = os.path.join(args.output_dir, "oof_index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({
            "n_folds": args.n_folds,
            "seed": args.seed,
            "fold_of_image_id": ownership,
            "canonical_test_ids": base_split["test"],
            "canonical_val_ids": base_split["val"],
            "detected_columns": detected,
            "skipped_records": skipped,
        }, handle, indent=2)
    print(f"OOF index: {index_path} ({len(ownership)} training rows covered)")


if __name__ == "__main__":
    main()
