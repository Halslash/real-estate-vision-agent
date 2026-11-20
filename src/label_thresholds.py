"""Pure, dependency-free tertile rules for the market-derived proxy labels.

Kept separate from ``dataset.py`` so the threshold methodology can be tested on
a CPU without torch. There is no file I/O and no model call here.

``unique_index_tertiles`` mirrors the historical ``v2_log_shrunk`` rule exactly
(it is used for comparison/diagnostics only; ``dataset.py`` keeps its own copy
so that checkpoint behaviour cannot drift). ``sample_quantile_tertiles`` is the
new ``v3_balanced_quantile`` rule: thresholds are weighted by *samples*, so ties
cannot unbalance the classes.

Neither rule sees any validation or test data: callers pass training proxy
values only.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

NUM_CLASSES = 3


def _finite_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size < NUM_CLASSES:
        raise ValueError(
            f"Need at least {NUM_CLASSES} finite proxy values to form tertiles "
            f"(got {array.size})."
        )
    return array


def unique_index_tertiles(values: Sequence[float]) -> Tuple[float, float]:
    """Historical v2 rule: cut the sorted *distinct* values at 1/3 and 2/3.

    A value repeated many times counts once, so heavy rounding of price or area
    can skew the resulting sample class counts. Reproduced for diagnostics; do
    not use it to reinterpret an existing checkpoint.
    """
    array = _finite_values(values)
    unique_values = np.unique(array)
    if unique_values.size < NUM_CLASSES:
        raise ValueError(
            f"Only {unique_values.size} distinct proxy value(s); cannot place "
            "three ordered class thresholds."
        )
    low = float(unique_values[unique_values.size // NUM_CLASSES])
    high = float(unique_values[(2 * unique_values.size) // NUM_CLASSES])
    if not low < high:
        raise ValueError("Historical tertile thresholds collapsed (low >= high).")
    return low, high


def sample_quantile_tertiles(values: Sequence[float]) -> Tuple[float, float]:
    """v3 rule: sample-weighted tertiles, robust to ties in the proxy value."""
    array = _finite_values(values)
    low, high = (float(value) for value in np.quantile(array, [1.0 / 3.0, 2.0 / 3.0]))
    if not low < high:
        raise ValueError(
            "Sample tertiles collapsed (low >= high): the training proxy values "
            "have too few distinct values for three ordered classes."
        )
    return low, high


def class_counts(values: Sequence[float], low: float, high: float) -> np.ndarray:
    """Return training counts for low/average/high under the given thresholds."""
    array = _finite_values(values)
    labels = np.where(array < low, 0, np.where(array < high, 1, 2))
    return np.bincount(labels, minlength=NUM_CLASSES)


def labels_for(values: Sequence[float], low: float, high: float) -> np.ndarray:
    """Deterministic class assignment shared by every diagnostic below."""
    array = _finite_values(values)
    return np.where(array < low, 0, np.where(array < high, 1, 2)).astype(int)


def tie_summary(values: Sequence[float], top_n: int = 10) -> Dict[str, Any]:
    """Quantify tied proxy values. Tied rows MUST share one class (no rank splitting)."""
    array = _finite_values(values)
    uniques, counts = np.unique(array, return_counts=True)
    tied = counts > 1
    rows_tied = int(counts[tied].sum())
    order = np.argsort(counts)[::-1][:top_n]
    return {
        "n_finite_values": int(array.size),
        "n_distinct_values": int(uniques.size),
        "n_tie_groups": int(tied.sum()),
        "rows_in_tie_groups": rows_tied,
        "fraction_rows_in_ties": float(rows_tied / array.size),
        "largest_tie_group": int(counts.max()),
        "largest_tie_value": float(uniques[int(counts.argmax())]),
        "top_tie_groups": [
            {"value": float(uniques[i]), "count": int(counts[i])} for i in order if counts[i] > 1
        ],
    }


def threshold_tie_impact(
    values: Sequence[float], low: float, high: float
) -> Dict[str, Any]:
    """Rows sitting exactly on a cut. v2 cuts are observed values, so this is >0."""
    array = _finite_values(values)
    at_low = int(np.count_nonzero(array == low))
    at_high = int(np.count_nonzero(array == high))
    return {
        "threshold_low": float(low),
        "threshold_high": float(high),
        "rows_equal_to_low": at_low,
        "rows_equal_to_high": at_high,
        "rows_on_a_threshold": at_low + at_high,
        "fraction_on_a_threshold": float((at_low + at_high) / array.size),
    }


def label_change_summary(
    values: Sequence[float],
    low_a: float, high_a: float,
    low_b: float, high_b: float,
) -> Dict[str, Any]:
    """How many rows change class when the cut rule changes (A -> B).

    ``transition_matrix[i][j]`` = rows that were class i under A and class j under B.
    This is the direct measure of v3's effect on label assignment.
    """
    array = _finite_values(values)
    labels_a = labels_for(array, low_a, high_a)
    labels_b = labels_for(array, low_b, high_b)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for old, new in zip(labels_a, labels_b):
        matrix[old, new] += 1
    changed = labels_a != labels_b
    examples = [
        {"value": float(array[i]), "from_class": int(labels_a[i]), "to_class": int(labels_b[i])}
        for i in np.flatnonzero(changed)[:10]
    ]
    return {
        "n_rows": int(array.size),
        "n_changed": int(changed.sum()),
        "fraction_changed": float(changed.sum() / array.size),
        "counts_a": class_counts(array, low_a, high_a).tolist(),
        "counts_b": class_counts(array, low_b, high_b).tolist(),
        "transition_matrix_a_to_b": matrix.tolist(),
        "changed_examples": examples,
    }
