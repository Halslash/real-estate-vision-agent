"""Read-only v2 vs v3 label-distribution and tie audit on the REAL dataset.

Answers, without training any model:
  1. What are the actual v2 and v3 class distributions on train/val/test?
  2. How many proxy values are tied, how large are the tie groups, and how many
     rows change class when the tertile rule changes?

Both threshold sets are fitted on the TRAIN partition only, using the exact same
``fit_preprocessing_stats`` / ``compute_labels`` code as training. The test split
is reported for documentation only and is NEVER used to choose a model, a
threshold or a label version. No checkpoint or experiment output is modified.

Example (Kaggle):
    !python src/label_audit.py --output /kaggle/working/output/label_audit/label_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGES_DIR,
    DEFAULT_LABEL_CONFIG_VERSION,
    DEFAULT_SHRINKAGE_K,
    LABEL_V2,
    LABEL_V3,
    compute_labels,
    fit_preprocessing_stats,
    load_and_validate_records,
    make_stratified_split,
    split_records,
)
from src.label_thresholds import (  # noqa: E402
    label_change_summary,
    threshold_tie_impact,
    tie_summary,
)

SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", default="/kaggle/working/output/label_audit/label_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrinkage_k", type=float, default=DEFAULT_SHRINKAGE_K)
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--price_col", type=str, default=None)
    parser.add_argument("--sqft_col", type=str, default=None)
    parser.add_argument("--city_col", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Smoke test only; NEVER use its output as a result.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing audit JSON.")
    return parser


def _distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    counts = [sum(1 for label in labels if label == i) for i in range(3)]
    total = len(labels)
    return {
        "n": total,
        "counts_low_average_high": counts,
        "fractions_low_average_high": [round(value / total, 6) if total else 0.0
                                       for value in counts],
        "balance_spread": max(counts) - min(counts),
    }


def _version_report(
    stats: Dict[str, Any], grouped: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    train_values = [float(row["proxy_value"]) for row in grouped["train"]]
    report: Dict[str, Any] = {
        "label_config_version": stats["label_config_version"],
        "threshold_low": float(stats["class_threshold_low"]),
        "threshold_high": float(stats["class_threshold_high"]),
        "threshold_rule": stats.get("threshold_rule", "distinct_value_index_tertiles"),
        "train_tie_summary": tie_summary(train_values),
        "train_threshold_impact": threshold_tie_impact(
            train_values, stats["class_threshold_low"], stats["class_threshold_high"]
        ),
        "distributions": {},
    }
    for name in SPLITS:
        rows = grouped[name]
        values = [float(row["proxy_value"]) for row in rows]
        report["distributions"][name] = {
            **_distribution(rows),
            "tie_summary": tie_summary(values),
        }
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing audit output: {args.output}. "
            "Choose a new path or pass --force."
        )
    records, skipped, detected = load_and_validate_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        id_col=args.id_col,
        price_col=args.price_col,
        sqft_col=args.sqft_col,
        city_col=args.city_col,
        max_images=args.max_images,
    )
    split = make_stratified_split(records, seed=args.seed)
    grouped = split_records(records, split)

    stats_v2 = fit_preprocessing_stats(
        grouped["train"], label_config_version=LABEL_V2, shrinkage_k=args.shrinkage_k
    )
    stats_v3 = fit_preprocessing_stats(
        grouped["train"], label_config_version=LABEL_V3, shrinkage_k=args.shrinkage_k
    )
    labeled = {
        version: {
            name: compute_labels(grouped[name], stats, inplace=False)
            for name in SPLITS
        }
        for version, stats in ((LABEL_V2, stats_v2), (LABEL_V3, stats_v3))
    }

    comparison: Dict[str, Any] = {}
    for name in SPLITS:
        # Proxy values are identical between v2 and v3 by construction; only the
        # cut positions differ, so v2 values are passed once with both thresholds.
        values = [float(row["proxy_value"]) for row in labeled[LABEL_V2][name]]
        comparison[name] = label_change_summary(
            values,
            stats_v2["class_threshold_low"], stats_v2["class_threshold_high"],
            stats_v3["class_threshold_low"], stats_v3["class_threshold_high"],
        )

    return {
        "csv_path": args.csv_path,
        "images_dir": args.images_dir,
        "seed": args.seed,
        "shrinkage_k": args.shrinkage_k,
        "default_label_config_version": DEFAULT_LABEL_CONFIG_VERSION,
        "detected_columns": detected,
        "rows_loaded": len(records),
        "rows_skipped": skipped,
        "split_sizes": {name: len(grouped[name]) for name in SPLITS},
        "smoke_test_max_images": args.max_images,
        "method": (
            "v2 and v3 thresholds fitted on TRAIN records only via the production "
            "fit_preprocessing_stats(); same split, same proxy values and city "
            "shrinkage. Val/test distributions are DESCRIPTIVE ONLY: no model, "
            "threshold or label-version decision may use them."
        ),
        "v2": _version_report(stats_v2, labeled[LABEL_V2]),
        "v3": _version_report(stats_v3, labeled[LABEL_V3]),
        "label_assignment_v2_to_v3": comparison,
        "guardrails": {
            "test_split_used_for_selection": False,
            "any_checkpoint_modified": False,
            "any_label_silently_redefined": False,
        },
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for version in (LABEL_V2, LABEL_V3):
        block = report[version]
        print(f"{version}: thresholds ({block['threshold_low']:.6f}, "
              f"{block['threshold_high']:.6f}) | train counts "
              f"{block['distributions']['train']['counts_low_average_high']} | "
              f"rows in ties {block['train_tie_summary']['rows_in_tie_groups']} "
              f"({block['train_tie_summary']['fraction_rows_in_ties']:.1%})")
    changed = report["label_assignment_v2_to_v3"]["train"]
    print(f"Train rows changing class v2->v3: {changed['n_changed']} "
          f"({changed['fraction_changed']:.1%})")
    if args.max_images:
        print("WARNING: --max_images smoke test; these numbers are NOT results.")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
