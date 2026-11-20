"""Standalone CSV inspection. Run this on Kaggle before training.

    !python src/inspect_csv.py
    !python src/inspect_csv.py --csv_path /path/to/other.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Support both `python src/inspect_csv.py` and `from src.inspect_csv import ...`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dataset import DEFAULT_CSV_PATH  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the real-estate CSV.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.csv_path)

    print(f"CSV path: {args.csv_path}")
    print(f"CSV shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for name in df.columns:
        print(f"  - {name}")
    print("\nFirst five rows:")
    print(df.head(5).to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nData types:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
