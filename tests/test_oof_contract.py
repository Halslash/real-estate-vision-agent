"""OOF isolation and Member B CSV contract tests.

These are pure invariants and run locally without torch; the Kaggle runs that
produce the actual scores are separate (see docs/KAGGLE_GUIDE.md).
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.oof_contract import (  # noqa: E402
    OOF_COLUMNS,
    check_coverage,
    check_fold_spec,
    make_oof_row,
)


def base_spec() -> dict:
    return {"fold": 0, "train": ["c", "d"], "val": ["a", "b"]}


def ownership() -> dict:
    return {"a": 0, "b": 0, "c": 1, "d": 1}


def sample_result() -> dict:
    return {
        "condition_score": 0.65,
        "label": "high",
        "probabilities": {"low": 0.10, "average": 0.30, "high": 0.60},
        "confidence": 0.60,
        "model": "mobilenet_v2",
        "score_type": "market_derived_visual_proxy",
    }


class FoldIsolationTests(unittest.TestCase):
    def test_valid_fold(self) -> None:
        val, train = check_fold_spec(base_spec(), 0, ownership(), ["t1"])
        self.assertEqual((val, train), (["a", "b"], ["c", "d"]))

    def test_wrong_fold_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_fold_spec(base_spec(), 1, ownership(), [])

    def test_train_val_overlap_rejected(self) -> None:
        spec = base_spec()
        spec["train"] = ["b", "d"]
        with self.assertRaises(ValueError):
            check_fold_spec(spec, 0, ownership(), [])

    def test_test_id_leakage_rejected(self) -> None:
        spec = base_spec()
        spec["train"] = ["c", "t1"]
        with self.assertRaises(ValueError):
            check_fold_spec(spec, 0, ownership(), ["t1"])

    def test_wrong_ownership_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_fold_spec(base_spec(), 0, {"a": 1, "b": 0, "c": 1, "d": 1}, [])

    def test_fold_training_on_own_row_rejected(self) -> None:
        spec = base_spec()
        spec["train"] = ["a", "d"]
        with self.assertRaises(ValueError):
            check_fold_spec(spec, 0, ownership(), [])


class CoverageTests(unittest.TestCase):
    def test_complete_coverage_passes(self) -> None:
        check_coverage(["a", "b", "c", "d"], ownership())

    def test_missing_row_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_coverage(["a", "b", "c"], ownership())

    def test_extra_row_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_coverage(["a", "b", "c", "d", "x"], ownership())


class MemberBRowTests(unittest.TestCase):
    def test_row_has_all_required_columns(self) -> None:
        row = make_oof_row("a", 0, sample_result(), "/ckpt/fold_0.pth")
        self.assertEqual(tuple(row), OOF_COLUMNS)
        self.assertEqual(row["image_id"], "a")
        self.assertEqual(row["fold"], 0)
        self.assertEqual(row["p_low"], 0.10)
        self.assertEqual(row["p_average"], 0.30)
        self.assertEqual(row["p_high"], 0.60)
        self.assertEqual(row["condition_score"], 0.65)
        self.assertEqual(row["confidence"], 0.60)
        self.assertEqual(row["checkpoint"], "/ckpt/fold_0.pth")

    def test_missing_probabilities_rejected(self) -> None:
        result = sample_result()
        del result["probabilities"]
        with self.assertRaises(ValueError):
            make_oof_row("a", 0, result, "/ckpt")

    def test_out_of_range_score_rejected(self) -> None:
        result = sample_result()
        result["condition_score"] = 1.4
        with self.assertRaises(RuntimeError):
            make_oof_row("a", 0, result, "/ckpt")


if __name__ == "__main__":
    unittest.main()
