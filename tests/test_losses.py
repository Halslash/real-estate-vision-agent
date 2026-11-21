"""Regression test for compute_class_weights input validation."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.losses import CLASS_WEIGHT_BALANCED, compute_class_weights  # noqa: E402


class ClassWeightValidationTests(unittest.TestCase):
    def test_zero_count_reports_value_error_for_sequence_input(self) -> None:
        """The validation branch must work for lists/tuples, not only ndarrays."""
        for counts in ([10, 0, 5], (10, 0, 5)):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(ValueError, "at least one training sample"):
                    compute_class_weights(counts, mode=CLASS_WEIGHT_BALANCED)

    def test_balanced_weights_are_unchanged(self) -> None:
        weights, active = compute_class_weights([10, 5, 5], mode=CLASS_WEIGHT_BALANCED)
        self.assertTrue(active)
        for actual, expected in zip(weights.tolist(), [2 / 3, 4 / 3, 4 / 3]):
            self.assertAlmostEqual(float(actual), expected, places=5)


if __name__ == "__main__":
    unittest.main()
