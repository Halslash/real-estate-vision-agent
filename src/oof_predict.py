"""Generate honest out-of-fold visual scores for Member B's training rows.

Each row is scored ONLY by a fold checkpoint that did not train on it. The
script fails closed if fold ownership is incomplete or overlapping, and never
scores the untouched test split. Output is a CSV keyed by ``image_id``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    load_and_validate_records,
)
from src.vision_agent import VisionAgent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/kaggle/working/output/oof",
                        help="Root produced by make_folds.py, containing folds/ and fold_k/ runs.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--n_folds", type=int, default=None,
                        help="Defaults to the value stored in oof_index.json.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--price_col", default=None)
    parser.add_argument("--sqft_col", default=None)
    parser.add_argument("--city_col", default=None)
    return parser


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    index = _load_json(os.path.join(args.output_dir, "oof_index.json"))
    n_folds = int(args.n_folds or index["n_folds"])
    ownership = {str(k): int(v) for k, v in index["fold_of_image_id"].items()}
    test_ids = set(index.get("canonical_test_ids", []))
    checked_ids: set[str] = set()

    specs: List[Dict[str, Any]] = []
    for fold in range(n_folds):
        spec_path = os.path.join(args.output_dir, "folds", f"fold_{fold}.json")
        spec = _load_json(spec_path)
        if int(spec.get("fold", fold)) != fold:
            raise ValueError(f"Fold spec {spec_path} has an unexpected fold id.")
        val_ids = set(spec["val"])
        train_ids = set(spec["train"])
        if val_ids & train_ids or val_ids & test_ids:
            raise ValueError(f"Fold {fold} leaks validation/test ids into its training set.")
        for image_id in val_ids:
            if ownership.get(image_id) != fold:
                raise ValueError(f"{image_id} is not owned by fold {fold} in oof_index.json.")
        for image_id in train_ids:
            if image_id in ownership and ownership[image_id] == fold:
                raise ValueError(f"Fold {fold} trains on its own OOF row {image_id}.")
        checked_ids.update(val_ids)

        checkpoint = os.path.join(args.output_dir, f"fold_{fold}", "models", "best_model.pth")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")
        specs.append({"fold": fold, "val_ids": sorted(val_ids), "checkpoint": checkpoint})

    if checked_ids != set(ownership):
        missing = sorted(set(ownership) - checked_ids)
        extra = sorted(checked_ids - set(ownership))
        raise ValueError(
            f"OOF coverage mismatch: missing={len(missing)} extra={len(extra)}. "
            "Refusing to emit partial OOF scores."
        )

    records, skipped, detected = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col or detected_index(index, "id"),
        price_col=args.price_col or detected_index(index, "price"),
        sqft_col=args.sqft_col or detected_index(index, "sqft"),
        city_col=args.city_col or detected_index(index, "city"),
    )
    path_by_id = {record["image_id"]: record["image_path"] for record in records}

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        fold = spec["fold"]
        ids = spec["val_ids"]
        paths = []
        for image_id in ids:
            if image_id not in path_by_id:
                raise FileNotFoundError(f"No image file matched for OOF row {image_id}.")
            paths.append(path_by_id[image_id])
        print(f"Scoring fold {fold}: {len(paths)} held-out rows with {spec['checkpoint']}")
        agent = VisionAgent(checkpoint_path=spec["checkpoint"], device=args.device)
        results = agent.score_batch(paths, batch_size=args.batch_size)
        for image_id, result in zip(ids, results):
            score = float(result["condition_score"])
            if not 0.0 <= score <= 1.0:
                raise RuntimeError(f"Invalid OOF score for {image_id}: {score}")
            rows.append({
                "image_id": image_id,
                "fold": fold,
                "condition_score": score,
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "model": result["model"],
                "checkpoint": spec["checkpoint"],
            })

    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if frame["image_id"].duplicated().any() or len(frame) != len(ownership):
        raise AssertionError("OOF output is incomplete or contains duplicate image_id rows.")
    output_csv = os.path.join(args.output_dir, "oof_visual_scores.csv")
    frame.to_csv(output_csv, index=False)
    print(f"Skipped records while reloading data: {skipped}")
    print(f"Wrote {len(frame)} OOF scores -> {output_csv}")
    print("Member B: train on these rows; use the selected final checkpoint for test rows only.")


def detected_index(index: Dict[str, Any], role: str):
    return index.get("detected_columns", {}).get(role)


if __name__ == "__main__":
    main()
