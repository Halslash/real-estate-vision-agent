"""Pure, dependency-free invariants for out-of-fold (OOF) score generation.

Kept separate from ``oof_predict.py`` so fold isolation and the Member B CSV
contract can be unit-tested without torch/torchvision installed. No file I/O and
no model calls happen here.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

OOF_COLUMNS = (
    "image_id",
    "fold",
    "condition_score",
    "p_low",
    "p_average",
    "p_high",
    "label",
    "confidence",
    "model",
    "checkpoint",
)


def check_fold_spec(
    spec: Dict[str, Any],
    fold: int,
    ownership: Dict[str, int],
    test_ids: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Validate one fold specification and return ``(val_ids, train_ids)``.

    Fails closed on a wrong fold id, train/val overlap, any test-id leakage, a
    validation row owned by another fold, or a fold training on its own OOF row.
    """
    if int(spec.get("fold", fold)) != fold:
        raise ValueError(f"Fold spec for fold {fold} has an unexpected fold id.")
    val_ids = set(spec["val"])
    train_ids = set(spec["train"])
    test_set = set(test_ids)
    if val_ids & train_ids or val_ids & test_set or train_ids & test_set:
        raise ValueError(f"Fold {fold} leaks validation/test ids into its training set.")
    for image_id in val_ids:
        if ownership.get(image_id) != fold:
            raise ValueError(f"{image_id} is not owned by fold {fold} in oof_index.json.")
    for image_id in train_ids:
        if ownership.get(image_id) == fold:
            raise ValueError(f"Fold {fold} trains on its own OOF row {image_id}.")
    return sorted(val_ids), sorted(train_ids)


def check_coverage(checked_ids: Iterable[str], ownership: Dict[str, int]) -> None:
    """Require exactly one validation assignment for every owned training row."""
    checked = set(checked_ids)
    expected = set(ownership)
    if checked != expected:
        missing = sorted(expected - checked)
        extra = sorted(checked - expected)
        raise ValueError(
            f"OOF coverage mismatch: missing={len(missing)} extra={len(extra)}. "
            "Refusing to emit partial OOF scores."
        )


def make_oof_row(
    image_id: str,
    fold: int,
    result: Dict[str, Any],
    checkpoint: str,
) -> Dict[str, Any]:
    """Build one Member B CSV row, including the full 3-class probability vector."""
    score = float(result["condition_score"])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Invalid OOF score for {image_id}: {score}")
    try:
        probabilities = result["probabilities"]
        p_low = float(probabilities["low"])
        p_average = float(probabilities["average"])
        p_high = float(probabilities["high"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"OOF result for {image_id} lacks valid low/average/high probabilities."
        ) from exc
    values = (p_low, p_average, p_high)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise RuntimeError(f"Invalid OOF class probabilities for {image_id}: {values}")
    return {
        "image_id": image_id,
        "fold": int(fold),
        "condition_score": score,
        "p_low": p_low,
        "p_average": p_average,
        "p_high": p_high,
        "label": result["label"],
        "confidence": float(result["confidence"]),
        "model": result["model"],
        "checkpoint": checkpoint,
    }
