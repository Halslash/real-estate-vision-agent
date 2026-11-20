"""Local tests for the qualitative demonstration builder (no torch needed)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.demonstrate import build_html, build_rows, collect_images  # noqa: E402


RESULT = {
    "condition_score": 0.55,
    "label": "average",
    "probabilities": {"low": 0.20, "average": 0.50, "high": 0.30},
    "confidence": 0.50,
    "model": "mobilenet_v2",
    "score_type": "market_derived_visual_proxy",
}


class DemonstrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images_dir = os.path.join(self.temp.name, "raw")
        os.makedirs(self.images_dir)
        for name in ("b.jpg", "a.jpg", "c.png", "ignore.txt"):
            with open(os.path.join(self.images_dir, name), "wb") as handle:
                handle.write(b"fake-image-bytes")

    def test_collect_images_is_sorted_and_bounded(self) -> None:
        args = Namespace(image_paths=[], images_dir=self.images_dir, num_examples=2)
        paths = collect_images(args)
        self.assertEqual([os.path.basename(p) for p in paths], ["a.jpg", "b.jpg"])

    def test_collect_images_rejects_missing_path(self) -> None:
        args = Namespace(image_paths=[os.path.join(self.images_dir, "nope.jpg")],
                         images_dir=None, num_examples=1)
        with self.assertRaises(FileNotFoundError):
            collect_images(args)

    def test_build_rows_copies_originals_and_keeps_probabilities(self) -> None:
        out_images = os.path.join(self.temp.name, "out", "images")
        os.makedirs(out_images)
        paths = [os.path.join(self.images_dir, "a.jpg")]
        rows = build_rows(paths, [RESULT], out_images)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["predicted_class"], "average")
        self.assertEqual((row["p_low"], row["p_average"], row["p_high"]), (0.20, 0.50, 0.30))
        self.assertEqual(row["condition_score"], 0.55)
        self.assertEqual(row["confidence"], 0.50)
        self.assertTrue(os.path.isfile(os.path.join(out_images, "000_a.jpg")))

    def test_html_labels_confidence_as_not_calibrated(self) -> None:
        out_images = os.path.join(self.temp.name, "out2", "images")
        os.makedirs(out_images)
        rows = build_rows([os.path.join(self.images_dir, "a.jpg")], [RESULT], out_images)
        page = build_html(rows, "/ckpt.pth")
        self.assertIn("max predicted class probability (NOT calibrated)", page)
        self.assertIn("<img src='images/000_a.jpg'", page)
        self.assertIn("P(low)</td><td>0.2000", page)
        self.assertIn("Interpretation warning", page)


if __name__ == "__main__":
    unittest.main()
