"""Thin config-driven runner so Kaggle commands stay short and reproducible.

One JSON config holds shared data/settings; each named experiment overrides a
few flags and always gets its own output directory. The runner refuses to
overwrite an existing checkpoint or the protected historical output folders.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTECTED_NAMES = {"baseline", "improved_mobilenet", "models", "results", "output"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.json")
    parser.add_argument("--name", help="Experiment key in the config; use --list to see keys.")
    parser.add_argument("--list", action="store_true", help="List experiment names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--eval", action="store_true",
                        help="Also run evaluate.py (use ONLY for the final selected model).")
    parser.add_argument("--probe", action="store_true", help="Also run probe.py (train->val only).")
    parser.add_argument("--force", action="store_true",
                        help="Allow reusing a checkpoint directory (never for protected names).")
    return parser


def load_config(path: str) -> Dict[str, Any]:
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Experiment config not found: {full}")
    with open(full, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("defaults", "experiments"):
        if key not in config:
            raise KeyError(f"Config {full} must define '{key}'.")
    return config


def resolve(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in config["experiments"]:
        raise KeyError(f"Unknown experiment {name!r}. Available: {sorted(config['experiments'])}")
    if name in PROTECTED_NAMES:
        raise ValueError(
            f"Refusing to write experiment {name!r}: that name is reserved for historical "
            "results and must never be overwritten."
        )
    merged: Dict[str, Any] = dict(config["defaults"])
    merged.update(config["experiments"][name].get("args", {}))
    root = config.get("root", "/kaggle/working/output/experiments")
    output_dir = config["experiments"][name].get("output_dir", os.path.join(root, name))
    if os.path.abspath(output_dir) == os.path.abspath("/kaggle/working/output"):
        raise ValueError("Refusing to write directly to /kaggle/working/output.")
    checkpoint = os.path.join(output_dir, "models", "best_model.pth")
    if os.path.isfile(checkpoint) and not config["experiments"][name].get("allow_resume", False) and not FORCE[0]:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}. Use a new --name or --force."
        )
    return {"args": merged, "output_dir": output_dir, "checkpoint": checkpoint,
            "description": config["experiments"][name].get("description", "")}


# Set from main() so resolve() keeps a simple signature.
FORCE: List[bool] = [False]


def to_flags(values: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        # All project CLIs use underscore option names (--id_col, --csv_path...);
        # argparse does not treat hyphens as equivalent.
        flag = "--" + key
        if value is True:
            flags.append(flag)
        else:
            flags.append(flag)
            flags.append(str(value))
    return flags


def command_for(script: str, values: Dict[str, Any], output_dir: str) -> List[str]:
    return [sys.executable, os.path.join("src", script)] + to_flags(values) + [
        "--output_dir", output_dir,
    ]


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    FORCE[0] = args.force
    config = load_config(args.config)
    if args.list:
        for name, spec in config["experiments"].items():
            print(f"{name}: {spec.get('description', '')}")
        return
    if not args.name:
        raise ValueError("Pass --name <experiment> (or --list).")

    spec = resolve(config, args.name)
    values = dict(spec["args"])
    train_cmd = command_for("train.py", values, spec["output_dir"])
    commands = [("train", train_cmd)]
    if args.probe:
        probe_values = {
            "csv_path": values.get("csv_path"),
            "images_dir": values.get("images_dir"),
            "checkpoint": spec["checkpoint"],
            "batch_size": values.get("batch_size"),
            "num_workers": values.get("num_workers"),
            "seed": values.get("seed"),
            "device": values.get("device"),
            "feature_source": "pretrained",
        }
        commands.append(("probe", ["python", "src/probe.py"] + to_flags(probe_values)
                         + ["--output_dir", spec["output_dir"]]))
    if args.eval:
        eval_values = {
            "csv_path": values.get("csv_path"),
            "images_dir": values.get("images_dir"),
            "checkpoint": spec["checkpoint"],
            "batch_size": values.get("batch_size"),
            "num_workers": values.get("num_workers"),
            "seed": values.get("seed"),
            "device": values.get("device"),
            "id_col": values.get("id_col"),
            "price_col": values.get("price_col"),
            "sqft_col": values.get("sqft_col"),
            "city_col": values.get("city_col"),
        }
        commands.append(("evaluate", ["python", "src/evaluate.py"] + to_flags(eval_values)
                         + ["--output_dir", spec["output_dir"]]))

    log: Dict[str, Any] = {
        "experiment": args.name,
        "description": spec["description"],
        "config": os.path.abspath(args.config if os.path.isabs(args.config)
                                  else os.path.join(REPO_ROOT, args.config)),
        "resolved_args": values,
        "output_dir": spec["output_dir"],
        "checkpoint": spec["checkpoint"],
        "commands": [" ".join(command) for _, command in commands],
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    print(f"Experiment: {args.name} -> {spec['output_dir']}")
    for step, command in commands:
        print(f"[{step}] " + " ".join(command))
    for step, command in commands:
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        print("Dry run: no commands executed and no files written.")
        return
    log["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(spec["output_dir"], exist_ok=True)
    log_path = os.path.join(spec["output_dir"], "experiment_run.json")
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
