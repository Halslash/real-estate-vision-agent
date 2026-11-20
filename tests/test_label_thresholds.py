"""Local, torch-free tests of the proxy-label tertile rules."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.label_thresholds import (  # noqa: E402
    class_counts,
    label_change_summary,
    sample_quantile_tertiles,
    threshold_tie_impact,
    tie_summary,
    unique_index_tertiles,
)


class ThresholdTests(unittest.TestCase):
    def test_v2_helper_matches_the_historical_formula(self) -> None:
        rng = np.random.default_rng(0)
        values = np.round(rng.normal(size=500), 2)  # includes ties
        unique = np.unique(values)
        expected = (
            float(unique[len(unique) // 3]),
            float(unique[(2 * len(unique)) // 3]),
        )
        self.assertEqual(unique_index_tertiles(values), expected)

    def test_v3_gives_balanced_classes_even_with_heavy_ties(self) -> None:
        rng = np.random.default_rng(1)
        # Heavy rounding creates many identical proxy values.
        values = np.round(rng.lognormal(0.0, 0.4, 3000), 1)

        v2_low, v2_high = unique_index_tertiles(values)
        v3_low, v3_high = sample_quantile_tertiles(values)
        v2_counts = class_counts(values, v2_low, v2_high)
        v3_counts = class_counts(values, v3_low, v3_high)

        # Ties at a boundary cannot be split (identical proxy -> identical label),
        # so v3 reduces but does not erase imbalance. It must be far better than v2.
        self.assertLess(max(v3_counts) - min(v3_counts),
                        (max(v2_counts) - min(v2_counts)) / 2)
        self.assertGreater(min(v3_counts), 0.20 * len(values))

    def test_thresholds_are_deterministic_and_ordered(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = sample_quantile_tertiles(values)
        self.assertAlmostEqual(low, 5.0 / 3.0)
        self.assertAlmostEqual(high, 10.0 / 3.0)
        self.assertEqual(class_counts(values, low, high).tolist(), [2, 2, 2])

    def test_non_finite_values_are_ignored(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, np.nan, np.inf]
        low, high = sample_quantile_tertiles(values)
        self.assertTrue(0.0 < low < high < 3.0)

    def test_degenerate_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            sample_quantile_tertiles([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            unique_index_tertiles([np.nan, np.inf])


class TieAndAssignmentTests(unittest.TestCase):
    def test_tie_summary_counts_rows_not_distinct_values(self) -> None:
        summary = tie_summary([1, 1, 1, 2, 3, 3, np.nan])
        self.assertEqual(summary["n_finite_values"], 6)
        self.assertEqual(summary["n_distinct_values"], 3)
        self.assertEqual(summary["n_tie_groups"], 2)
        self.assertEqual(summary["rows_in_tie_groups"], 5)
        self.assertEqual(summary["largest_tie_group"], 3)
        self.assertAlmostEqual(summary["fraction_rows_in_ties"], 5 / 6)

    def test_threshold_impact_counts_rows_exactly_on_each_cut(self) -> None:
        # v2 cuts are observed values, so at least the cut rows are counted.
        values = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0]
        low, high = unique_index_tertiles(values)
        impact = threshold_tie_impact(values, low, high)
        self.assertGreater(impact["rows_equal_to_low"], 0)
        self.assertGreater(impact["rows_equal_to_high"], 0)
        self.assertEqual(impact["rows_on_a_threshold"],
                         impact["rows_equal_to_low"] + impact["rows_equal_to_high"])

    def test_label_change_summary_tracks_the_moved_rows(self) -> None:
        # A cut at 2.0 -> 3.0 moves the single value 2.0 from average to low.
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        report = label_change_summary(values, 2.0, 4.0, 3.0, 5.0)
        self.assertEqual(report["n_changed"], 2)
        self.assertEqual(report["counts_a"], [1, 2, 3])
        self.assertEqual(report["counts_b"], [2, 2, 2])
        matrix = report["transition_matrix_a_to_b"]
        self.assertEqual(sum(sum(row) for row in matrix), len(values))
        self.assertEqual(len(report["changed_examples"]), 2)

    def test_identical_thresholds_produce_no_changes(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = sample_quantile_tertiles(values)
        report = label_change_summary(values, low, high, low, high)
        self.assertEqual(report["n_changed"], 0)
        matrix = report["transition_matrix_a_to_b"]
        self.assertTrue(all(matrix[i][j] == 0 for i in range(3) for j in range(3) if i != j))


if __name__ == "__main__":
    unittest.main()
