"""Config/CLI contract tests. These run without torch installed."""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.run_experiment import load_config, resolve, to_flags  # noqa: E402


def cli_options(script: str) -> set[str]:
    """Read add_argument('--name', ...) literals from a script without importing it."""
    path = os.path.join(ROOT, "src", script)
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                options.add(arg.value[2:])
    return options


class ExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(os.path.join(ROOT, "configs", "experiments.json"))

    def test_every_configured_flag_is_accepted_by_train(self) -> None:
        accepted = cli_options("train.py")
        for name in self.config["experiments"]:
            with self.subTest(experiment=name):
                spec = resolve(self.config, name)
                tokens = to_flags(spec["args"])
                flags = {token[2:] for token in tokens[::2]}  # values are the odd tokens
                unknown = flags - accepted
                self.assertFalse(unknown, f"{name}: unsupported train flags {sorted(unknown)}")

    def test_probe_and_evaluate_accept_the_flags_the_runner_builds(self) -> None:
        probe = cli_options("probe.py")
        evaluate = cli_options("evaluate.py")
        for required in ("checkpoint", "feature_source", "output_dir"):
            self.assertIn(required, probe)
        for required in ("checkpoint", "output_dir", "id_col", "price_col", "sqft_col", "city_col"):
            self.assertIn(required, evaluate)

    def test_seed_and_split_settings_are_shared(self) -> None:
        defaults = self.config["defaults"]
        self.assertEqual(defaults["seed"], 42)
        self.assertEqual(defaults["batch_size"], 64)
        self.assertEqual(defaults["label_config_version"], "v2_log_shrunk")
        for name, spec in self.config["experiments"].items():
            with self.subTest(experiment=name):
                merged = dict(defaults)
                merged.update(spec.get("args", {}))
                self.assertEqual(merged["seed"], 42)

    def test_output_dirs_are_distinct_and_not_historical(self) -> None:
        paths = [resolve(self.config, name)["output_dir"] for name in self.config["experiments"]]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertNotIn("improved_mobilenet", path)
            self.assertNotEqual(os.path.abspath(path), "/kaggle/working/output")

    def test_protected_names_are_rejected(self) -> None:
        # A protected name must fail even if someone adds it to the config.
        for name in ("improved_mobilenet", "baseline", "models", "results"):
            with self.subTest(name=name):
                poisoned = dict(self.config)
                poisoned["experiments"] = dict(self.config["experiments"])
                poisoned["experiments"][name] = {"description": "should never run"}
                with self.assertRaises(ValueError):
                    resolve(poisoned, name)

    def test_existing_run_artifact_fails_closed(self) -> None:
        # Simulate a prior run without touching any real/Kaggle output path.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "results", "metrics.json"), "w") as handle:
                handle.write("{}")
            poisoned = dict(self.config)
            poisoned["experiments"] = {
                "ghost": {"description": "x", "output_dir": tmp}
            }
            with self.assertRaises(FileExistsError):
                resolve(poisoned, "ghost")

    def test_to_flags_skips_none_and_false_and_keeps_underscores(self) -> None:
        flags = to_flags({"seed": 42, "id_col": None, "verbose": False, "do_it": True})
        self.assertEqual(flags, ["--seed", "42", "--do_it"])
        self.assertTrue(all("-" not in flag[2:] for flag in flags))


if __name__ == "__main__":
    unittest.main()
