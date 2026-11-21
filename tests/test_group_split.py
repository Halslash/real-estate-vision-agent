"""Tests for the opt-in leakage-resistant group split (no training)."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.group_split import (  # noqa: E402
    MISSING_ADDRESS_TOKENS,
    build_groups,
    make_group_split,
    missing_address_ids,
    normalize_address,
)

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


def _records(count: int):
    """Synthetic records with a pps value spread over three tertiles."""
    return [
        {
            "image_id": str(i),
            "price_per_sqft": 100.0 + (i % 30) * 10.0,
            "price": (100.0 + (i % 30) * 10.0) * 1000,
            "sqft": 1000,
            "city": "Test City, CA",
        }
        for i in range(count)
    ]


def _groups(count: int):
    """Pairs of consecutive ids form one property group."""
    return {f"g{i // 2:03d}": [str(i), str(i + 1)] for i in range(0, count, 2)}


def _image_fixture(tmp: str):
    """1/2 share an address, 1/3 share exact bytes, 1/4 are near-duplicates."""
    def save(name: str, pattern: str, color: int = 255) -> None:
        image = Image.new("L", (64, 64), color)
        for x in range(64):
            for y in range(64):
                if pattern == "vertical" and x % 8 < 4:
                    image.putpixel((x, y), 0)
                elif pattern == "horizontal" and y % 8 < 4:
                    image.putpixel((x, y), 0)
        image.convert("RGB").save(os.path.join(tmp, f"{name}.jpg"))

    save("1", "vertical")
    save("2", "horizontal")            # same address as 1, different image
    save("3", "vertical")              # different address, identical bytes to 1
    save("4", "vertical", color=254)   # different address, near-duplicate of 1
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8))
    noise.convert("RGB").save(os.path.join(tmp, "5.jpg"))

    records = [{"image_id": str(i)} for i in range(1, 6)]
    streets = {"1": "1317 Van Buren Ave", "2": "1317 VAN BUREN AVENUE",
               "3": "9 Other Street", "4": "42 Different Road",
               "5": "99 Unrelated Lane"}
    return records, streets


class AddressTests(unittest.TestCase):
    def test_normalizes_case_punctuation_and_abbreviations(self) -> None:
        expected = "1317 van buren avenue"
        for raw in ("1317 Van Buren Avenue", "1317 van buren ave.", " 1317  VAN BUREN AVE "):
            self.assertEqual(normalize_address(raw), expected)

    def test_sentinels_are_not_treated_as_one_shared_property(self) -> None:
        for raw in MISSING_ADDRESS_TOKENS:
            self.assertIsNone(normalize_address(raw))
        self.assertIsNone(normalize_address(None))
        self.assertIsNone(normalize_address(float("nan")))

    def test_missing_addresses_are_reported_explicitly(self) -> None:
        streets = {"1": "1317 Van Buren Ave", "2": "Address not provided",
                   "3": "", "4": float("nan")}
        self.assertEqual(
            missing_address_ids(streets, ["1", "2", "3", "4"]), ["2", "3", "4"]
        )


class BuildGroupsTests(unittest.TestCase):
    def test_links_same_address_and_exact_bytes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, streets = _image_fixture(tmp)
            groups = build_groups(records, streets, tmp, verbose=False)
            owner = {image_id: group for group, members in groups.items() for image_id in members}
            self.assertEqual(owner["1"], owner["2"], "same address must stay together")
            self.assertEqual(owner["1"], owner["3"], "exact image bytes must stay together")
            self.assertNotEqual(owner["1"], owner["4"], "dHash is off by default")
            self.assertNotEqual(owner["1"], owner["5"], "unrelated images must stay separate")

    def test_near_duplicates_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, streets = _image_fixture(tmp)
            groups = build_groups(records, streets, tmp, near_duplicates=True, verbose=False)
            owner = {image_id: group for group, members in groups.items() for image_id in members}
            self.assertEqual(owner["1"], owner["4"], "opt-in dHash must link near-duplicates")

    def test_default_workflow_does_not_invoke_dhash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, streets = _image_fixture(tmp)
            with patch("src.group_split._dhash", side_effect=AssertionError("dHash must stay off")):
                build_groups(records, streets, tmp, verbose=False)

    def test_missing_image_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_groups([{"image_id": "1"}], {"1": "1 Main Street"}, tmp, verbose=False)

    def test_empty_records_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_groups([], {}, tmp, verbose=False)


class SplitTests(unittest.TestCase):
    def test_groups_are_disjoint_and_all_records_covered(self) -> None:
        records = _records(60)
        groups = _groups(60)
        split = make_group_split(records, groups, seed=42, verbose=False)
        sets = [set(split[name]) for name in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set().union(*sets), {r["image_id"] for r in records})
        owner = {image_id: name for name, ids in split.items() for image_id in ids}
        for members in groups.values():
            self.assertEqual(len({owner[image_id] for image_id in members}), 1)

    def test_addresses_and_exact_images_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 40
            records = _records(n)
            streets = {str(i): f"{1000 + i} Test Street" for i in range(n)}
            streets["1"] = streets["0"]  # same address, different bytes
            for i in range(n):
                color = (i % 20) * 10   # ids i and i+20 share exact image bytes
                Image.new("RGB", (16, 16), (color, color, color)).save(
                    os.path.join(tmp, f"{i}.jpg")
                )
            groups = build_groups(records, streets, tmp, verbose=False)
            split = make_group_split(records, groups, seed=42, verbose=False)

            owner = {image_id: name for name, ids in split.items() for image_id in ids}
            self.assertEqual(len(owner), n, "every valid record appears exactly once")
            self.assertEqual(set(owner), {str(i) for i in range(n)})
            self.assertEqual(owner["0"], owner["1"], "same address must not cross splits")
            for i in range(20):
                self.assertEqual(owner[str(i)], owner[str(i + 20)],
                                 "exact duplicate image must not cross splits")

    def test_reproducible_and_seed_sensitive(self) -> None:
        records, groups = _records(60), _groups(60)
        first = make_group_split(records, groups, seed=42, verbose=False)
        again = make_group_split(records, groups, seed=42, verbose=False)
        self.assertEqual(first, again)
        other = make_group_split(records, groups, seed=7, verbose=False)
        self.assertNotEqual(first, other)

    def test_split_sizes_and_strata_are_reasonable(self) -> None:
        records, groups = _records(600), _groups(600)
        split = make_group_split(records, groups, seed=42, verbose=False)
        total = len(records)
        for name, expected in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            self.assertLess(abs(len(split[name]) / total - expected), 0.05, name)
        pps = {r["image_id"]: r["price_per_sqft"] for r in records}
        for name, ids in split.items():
            lows = [pps[i] for i in ids]
            self.assertLess(min(lows), 150.0)   # low tertile present
            self.assertGreater(max(lows), 250.0)  # high tertile present

    def test_too_few_groups_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            make_group_split(_records(4), _groups(4), verbose=False)

    @unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed (run on Kaggle)")
    def test_proxy_class_distribution_preserved(self) -> None:
        from src.dataset import compute_labels, fit_preprocessing_stats, split_records

        records = _records(600)
        for record in records:
            record["price"] = record["price_per_sqft"] * 1000
        split = make_group_split(records, _groups(600), seed=42, verbose=False)
        train_ids = set(split["train"])
        stats = fit_preprocessing_stats([r for r in records if r["image_id"] in train_ids])
        labeled = split_records(compute_labels(records, stats, inplace=False), split)
        for name in ("train", "val", "test"):
            counts = np.bincount([r["label"] for r in labeled[name]], minlength=3)
            self.assertTrue((counts > 0).all(), f"{name}: {counts.tolist()}")
            for proportion in counts / counts.sum():
                self.assertGreater(proportion, 0.15)
                self.assertLess(proportion, 0.50)


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed (run on Kaggle)")
class SuppliedSplitValidationTests(unittest.TestCase):
    """train.py must refuse a --split_file that leaks, duplicates or drops rows."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.train import validate_supplied_split

        cls.validate = staticmethod(validate_supplied_split)
        cls.records = _records(10)

    def test_complete_disjoint_split_is_accepted(self) -> None:
        split = {
            "train": [str(i) for i in range(6)],
            "val": ["6", "7"],
            "test": ["8", "9"],
        }
        self.validate(split, self.records)

    def test_overlapping_splits_are_rejected(self) -> None:
        split = {
            "train": [str(i) for i in range(7)],
            "val": ["6", "7"],
            "test": ["8", "9"],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.validate(split, self.records)

    def test_dropped_record_is_rejected(self) -> None:
        split = {
            "train": [str(i) for i in range(5)],
            "val": ["5", "6"],
            "test": ["7", "8"],
        }
        with self.assertRaisesRegex(ValueError, "omits"):
            self.validate(split, self.records)

    def test_duplicate_id_is_rejected(self) -> None:
        split = {
            "train": ["0", "1", "2", "3", "4", "4"],
            "val": ["5", "6"],
            "test": ["7", "8", "9"],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.validate(split, self.records)

    def test_unknown_id_is_rejected(self) -> None:
        split = {
            "train": [str(i) for i in range(6)],
            "val": ["6", "7"],
            "test": ["8", "999"],
        }
        with self.assertRaisesRegex(ValueError, "absent"):
            self.validate(split, self.records)

    def test_oof_style_empty_test_is_accepted(self) -> None:
        split = {"train": [str(i) for i in range(8)], "val": ["8", "9"], "test": []}
        self.validate(split, self.records)


if __name__ == "__main__":
    unittest.main()
