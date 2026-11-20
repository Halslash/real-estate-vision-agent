"""VisionAgent inference tests. Skipped locally when torch/torchvision are absent;
run them on Kaggle where the real dependencies and checkpoints exist.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))

if HAS_TORCH:
    import numpy as np
    from PIL import Image

    from src.dataset import IMAGENET_MEAN, IMAGENET_STD
    from src.losses import (
        CLASS_WEIGHT_AUTO,
        CLASS_WEIGHT_BALANCED,
        CLASS_WEIGHT_EFFECTIVE_NUMBER,
        CLASS_WEIGHT_NONE,
        compute_class_weights,
    )
    from src.model import create_model, make_checkpoint, save_checkpoint
    from src.vision_agent import VisionAgent


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed (run on Kaggle)")
class LossTests(unittest.TestCase):
    def test_weight_modes(self) -> None:
        counts = [100, 10, 40]
        none, active = compute_class_weights(counts, CLASS_WEIGHT_NONE)
        self.assertFalse(active)
        self.assertTrue((none == 1.0).all())

        auto, auto_active = compute_class_weights(counts, CLASS_WEIGHT_AUTO)
        self.assertTrue(auto_active)  # ratio 10 > default 1.5

        balanced, _ = compute_class_weights(counts, CLASS_WEIGHT_BALANCED)
        self.assertAlmostEqual(float(balanced[0] / balanced[1]), 0.1, places=5)

        effective, _ = compute_class_weights(counts, CLASS_WEIGHT_EFFECTIVE_NUMBER)
        self.assertAlmostEqual(float(effective.mean()), 1.0, places=5)

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_class_weights([10, 0, 5], CLASS_WEIGHT_BALANCED)


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed (run on Kaggle)")
class VisionAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.checkpoint_path = os.path.join(cls.temp_dir.name, "ckpt.pth")
        model = create_model("mobilenet_v2", num_classes=3, pretrained=False, progress=False)
        checkpoint = make_checkpoint(
            model,
            idx_to_label={0: "low", 1: "average", 2: "high"},
            preprocessing={
                "image_size": 224, "resize_size": 256,
                "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "architecture": "mobilenet_v2",
            },
            label_stats={"label_config_version": "v2_log_shrunk"},
            train_config={"test_fixture": True},
            epoch=1,
            val_macro_f1=0.0,
            history=[],
        )
        save_checkpoint(cls.checkpoint_path, checkpoint)

        cls.rgb_path = os.path.join(cls.temp_dir.name, "rgb.png")
        base = Image.new("RGB", (260, 300), (120, 90, 70))
        base.save(cls.rgb_path)
        cls.gray_path = os.path.join(cls.temp_dir.name, "gray.png")
        base.convert("L").save(cls.gray_path)
        cls.rgba_path = os.path.join(cls.temp_dir.name, "rgba.png")
        Image.new("RGBA", (260, 300), (120, 90, 70, 255)).save(cls.rgba_path)
        cls.bad_path = os.path.join(cls.temp_dir.name, "not_an_image.png")
        with open(cls.bad_path, "wb") as handle:
            handle.write(b"not a png")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.agent = VisionAgent(self.checkpoint_path, device="cpu")

    def assert_valid(self, result: dict) -> None:
        for key in ("condition_score", "label", "probabilities", "confidence", "model", "score_type"):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["condition_score"], 0.0)
        self.assertLessEqual(result["condition_score"], 1.0)
        self.assertIn(result["label"], ("low", "average", "high"))
        self.assertEqual(set(result["probabilities"]), {"low", "average", "high"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(result["model"], "mobilenet_v2")
        self.assertEqual(result["score_type"], "market_derived_visual_proxy")

    def test_checkpoint_loading(self) -> None:
        self.assert_valid(self.agent.score(self.rgb_path))

    def test_single_and_batch_agree(self) -> None:
        paths = [self.rgb_path, self.gray_path, self.rgba_path]
        singles = [self.agent.score(path) for path in paths]
        batches = self.agent.score_batch(paths, batch_size=2)
        for single, batch in zip(singles, batches):
            self.assert_valid(batch)
            self.assertAlmostEqual(single["condition_score"], batch["condition_score"], places=6)
            self.assertEqual(single["label"], batch["label"])

    def test_batch_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.score_batch([self.rgb_path], batch_size=0)

    def test_color_mode_equivalence(self) -> None:
        rgb = self.agent.score(self.rgb_path)["condition_score"]
        gray = self.agent.score(self.gray_path)["condition_score"]
        rgba = self.agent.score(self.rgba_path)["condition_score"]
        self.assertAlmostEqual(rgb, rgba, places=6)
        # Grayscale is a different pixel mapping, so only check it is valid.
        self.assertGreaterEqual(gray, 0.0)
        self.assertLessEqual(gray, 1.0)

    def test_missing_image_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.agent.score(os.path.join(self.temp_dir.name, "missing.png"))

    def test_corrupt_image_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.score(self.bad_path)

    def test_using_agent_before_load_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            VisionAgent(device="cpu").score(self.rgb_path)

    def test_legacy_load_api(self) -> None:
        agent = VisionAgent(device="cpu").load(self.checkpoint_path)
        self.assert_valid(agent.score(self.rgb_path))


if __name__ == "__main__":
    unittest.main()
