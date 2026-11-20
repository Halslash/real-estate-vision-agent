"""Read-only data/split leakage audit. Does NOT change records or splits.

Checks (evidence for docs/CNN_PERFORMANCE_AUDIT.md):
  * CSV -> image matching failures and file-extension distribution;
  * exact duplicate image bytes and duplicate perceptual hashes;
  * near-duplicate hashes inside the same street group;
  * street / city groups spanning more than one split.

All heavy work is local file hashing; no model, CSV rewrite or GPU is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

DEFAULT_CSV = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal2.csv"
)
DEFAULT_IMAGES = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal_pics/"
)
DEFAULT_SPLIT = "/kaggle/working/output/experiments/e0_reference/models/split_indices.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT,
                        help="Saved split_indices.json (from a training run).")
    parser.add_argument("--output", default="/kaggle/working/output/data_audit/data_audit.json")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--price_col", default="price")
    parser.add_argument("--sqft_col", default="sqft")
    parser.add_argument("--city_col", default="citi")
    parser.add_argument("--street_col", default="street")
    parser.add_argument("--hash_bits", type=int, default=5,
                        help="Near-duplicate Hamming-distance threshold on the 64-bit dHash.")
    parser.add_argument("--max_images", type=int, default=None)
    return parser


def _norm_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("\\", "/")
    stem = os.path.splitext(os.path.basename(text))[0].strip()
    if not stem or stem.lower() in {"nan", "none", "null", "na"}:
        return None
    try:
        number = float(stem)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return stem


def _find_image(images_dir: str, image_id: str) -> Optional[str]:
    """Try the canonical .jpg first, then common extensions (diagnostic only)."""
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(images_dir, image_id + extension)
        if os.path.isfile(path):
            return path
    return None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dhash(path: str) -> Optional[int]:
    """64-bit difference hash of the 8x8 grayscale thumbnail."""
    try:
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    except Exception:
        return None
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


def _group_overlap(groups: Dict[str, Dict[str, int]], split_names: List[str]) -> Dict[str, Any]:
    spanning = {
        key: counts for key, counts in groups.items() if len(counts) > 1
    }
    per_pair: Counter = Counter()
    for counts in spanning.values():
        names = sorted(counts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                per_pair[f"{names[i]}|{names[j]}"] += 1
    return {
        "groups_in_multiple_splits": len(spanning),
        "pair_counts": dict(per_pair),
        "examples": [
            {"group": key, "counts": counts}
            for key, counts in list(spanning.items())[:20]
        ],
        "split_names": split_names,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    frame = pd.read_csv(args.csv_path)
    required = [args.id_col, args.price_col, args.sqft_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing CSV column(s) {missing}; available: {list(frame.columns)}")

    split: Optional[Dict[str, List[str]]] = None
    if args.split_file and os.path.isfile(args.split_file):
        with open(args.split_file, "r", encoding="utf-8") as handle:
            split = json.load(handle)
    elif args.split_file:
        print(f"WARNING: split file not found ({args.split_file}); split checks skipped.")

    split_of: Dict[str, str] = {}
    if split:
        for name in ("train", "val", "test"):
            for image_id in split.get(name, []):
                split_of[str(image_id)] = name

    total_rows = len(frame)
    if args.max_images and args.max_images > 0:
        frame = frame.head(args.max_images)

    extensions: Counter = Counter()
    missing_files: List[str] = []
    exact_hashes: Dict[str, List[str]] = defaultdict(list)
    dhash_values: Dict[str, int] = {}
    streets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cities: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    street_by_id: Dict[str, str] = {}

    for _, row in frame.iterrows():
        image_id = _norm_id(row[args.id_col])
        if image_id is None:
            continue
        path = _find_image(args.images_dir, image_id)
        if path is None:
            missing_files.append(image_id)
            continue
        extensions[os.path.splitext(path)[1].lower()] += 1
        exact_hashes[_sha256(path)].append(image_id)
        dhash_values[image_id] = _dhash(path)

        group = split_of.get(image_id, "unsplit")
        street = str(row[args.street_col]).strip() if args.street_col in frame.columns else ""
        city = str(row[args.city_col]).strip() if args.city_col in frame.columns else ""
        if street and street.lower() != "nan":
            streets[street][group] += 1
            street_by_id[image_id] = street
        if city and city.lower() != "nan":
            cities[city][group] += 1

    duplicate_exact = {
        digest: ids for digest, ids in exact_hashes.items() if len(ids) > 1
    }
    splitting_duplicates = {
        digest: ids for digest, ids in duplicate_exact.items()
        if len({split_of.get(i, "unsplit") for i in ids}) > 1
    }

    # Near-duplicates: restrict comparison to same-street groups to keep this O(N * group).
    near_pairs: List[Dict[str, Any]] = []
    by_street: Dict[str, List[str]] = defaultdict(list)
    for image_id, street in street_by_id.items():
        by_street[street].append(image_id)
    threshold = int(args.hash_bits)
    for street, ids in by_street.items():
        for i in range(len(ids)):
            a, ha = ids[i], dhash_values.get(ids[i])
            if ha is None:
                continue
            for j in range(i + 1, len(ids)):
                b, hb = ids[j], dhash_values.get(ids[j])
                if hb is None or split_of.get(a) == split_of.get(b):
                    continue
                distance = (ha ^ hb).bit_count()
                if distance <= threshold:
                    near_pairs.append({
                        "street": street, "image_id_a": a, "image_id_b": b,
                        "split_a": split_of.get(a, "unsplit"),
                        "split_b": split_of.get(b, "unsplit"),
                        "hamming_distance": distance,
                    })

    report: Dict[str, Any] = {
        "csv_path": args.csv_path,
        "images_dir": args.images_dir,
        "split_file": args.split_file if split else None,
        "rows_in_csv": total_rows,
        "rows_scanned": int(min(total_rows, args.max_images or total_rows)),
        "images_matched": sum(extensions.values()),
        "missing_image_files": len(missing_files),
        "missing_image_examples": missing_files[:20],
        "extension_counts": dict(extensions),
        "exact_duplicate_groups": len(duplicate_exact),
        "exact_duplicate_groups_across_splits": len(splitting_duplicates),
        "exact_duplicate_examples": list(splitting_duplicates.values())[:20],
        "near_duplicate_pairs_across_splits": len(near_pairs),
        "near_duplicate_hamming_threshold": threshold,
        "near_duplicate_examples": near_pairs[:20],
        "street_overlap": _group_overlap(streets, ["train", "val", "test"]),
        "city_overlap": _group_overlap(cities, ["train", "val", "test"]),
        "note": (
            "Read-only diagnostic. It does not change the split or any label. "
            "Cross-split duplicates/near-duplicates would make held-out metrics optimistic."
        ),
    }
    return report


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Rows scanned: {report['rows_scanned']} | matched images: {report['images_matched']}")
    print(f"Missing image files: {report['missing_image_files']}")
    print(f"Exact duplicate groups across splits: {report['exact_duplicate_groups_across_splits']}")
    print(f"Near-duplicate pairs across splits (same street): "
          f"{report['near_duplicate_pairs_across_splits']}")
    print(f"Street groups spanning splits: {report['street_overlap']['groups_in_multiple_splits']}")
    print(f"City groups spanning splits: {report['city_overlap']['groups_in_multiple_splits']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
