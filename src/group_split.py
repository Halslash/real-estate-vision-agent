"""Opt-in, leakage-resistant 70/15/15 group split for Member A.

This module NEVER trains and NEVER edits records, labels or the default split.
It only emits a ``split_indices.json`` that ``train.py --split_file`` can reuse.

By default, groups are connected components over two cheap evidence types:
  1. the same normalized full street ADDRESS (the CSV ``street`` field is a full
     address, e.g. ``1317 Van Buren Avenue``, not just a street name);
  2. byte-identical image files (SHA-256).

``--near_duplicates`` additionally links visually similar images with 64-bit
dHash (Hamming <= --hash_bits) using a global all-pairs comparison. It is OFF by
default because that comparison is the Kaggle bottleneck; it is an optional
audit experiment, not a prerequisite for a defensible split.

Splitting happens at GROUP level with the same temporary price-per-sqft tertile
stratification and the same two-step 70/30 -> 50/50 recipe as
``dataset.make_stratified_split``. No proxy label is fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_audit import _dhash  # noqa: E402  (PIL/pandas only, no torch)

DEFAULT_CSV = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal2.csv"
)
DEFAULT_IMAGES = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal_pics/"
)
DEFAULT_OUTPUT_DIR = "/kaggle/working/output/leakage_resistant_split"

# These denote an unusable address, not a shared property. Such rows start as
# singletons; byte-identical images still get grouped by the exact-hash edge.
MISSING_ADDRESS_TOKENS = {"", "address not provided", "not provided", "unknown", "nan", "none", "null", "na"}
ABBREVIATIONS = {
    "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "rd": "road", "ln": "lane", "ct": "court", "cir": "circle",
    "pl": "place", "ter": "terrace", "hwy": "highway", "pkwy": "parkway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ste": "suite", "apt": "apartment", "unit": "unit",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--images_dir", default=DEFAULT_IMAGES)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--street_col", default="street",
                        help="Column holding the full street address.")
    parser.add_argument("--near_duplicates", action="store_true",
                        help="Optional global dHash linking (slow). Off by default.")
    parser.add_argument("--hash_bits", type=int, default=5,
                        help="dHash threshold; used only with --near_duplicates.")
    parser.add_argument("--allow_existing", action="store_true",
                        help="Explicitly allow replacing an existing generated split.")
    return parser


def normalize_address(value: Any) -> Optional[str]:
    """Canonicalize a full address; None means 'unusable as a property key'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower().strip()).strip()
    if text in MISSING_ADDRESS_TOKENS:
        return None
    return " ".join(ABBREVIATIONS.get(token, token) for token in text.split()) or None


def missing_address_ids(streets: Mapping[str, Any], ids: Sequence[str]) -> List[str]:
    """Ids with an absent/placeholder address; reported, never merged as one property."""
    return [image_id for image_id in ids if normalize_address(streets.get(image_id)) is None]


def _id_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _popcount(values: np.ndarray) -> np.ndarray:
    """Popcount for uint64 arrays; numpy < 2.0 has no np.bitwise_count."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.uint8)


def _near_duplicate_pairs(hashes: Mapping[str, Optional[int]], threshold: int) -> List[Tuple[str, str]]:
    """All global pairs with dHash Hamming distance <= threshold (chunked)."""
    ids = [key for key, value in hashes.items() if value is not None]
    values = np.array([hashes[key] for key in ids], dtype=np.uint64)
    pairs: List[Tuple[str, str]] = []
    chunk = 512
    n = len(values)
    indices = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xor = np.bitwise_xor(values[start:stop, None], values[None, :])
        distance = _popcount(xor)
        upper = indices[None, :] > indices[start:stop, None]
        rows, cols = np.where((distance <= threshold) & upper)
        pairs.extend((ids[start + row], ids[col]) for row, col in zip(rows, cols))
    return pairs


def build_groups(
    records: Sequence[Mapping[str, Any]],
    streets: Mapping[str, Any],
    images_dir: str,
    hash_bits: int = 5,
    *,
    near_duplicates: bool = False,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Return group_id -> sorted member image ids (one group never crosses splits).

    Address and exact-SHA-256 edges are always used. Global dHash edges are
    computed only when ``near_duplicates`` is explicitly enabled.
    """
    ids = [str(record["image_id"]) for record in records]
    if not ids:
        raise ValueError("No valid records; cannot build groups.")
    union = _UnionFind(ids)

    by_address: Dict[str, List[str]] = defaultdict(list)
    for image_id in ids:
        address = normalize_address(streets.get(image_id))
        if address is not None:
            by_address[address].append(image_id)
    address_edges = 0
    for members in by_address.values():
        for other in members[1:]:
            address_edges += union.union(members[0], other)

    exact_hashes: Dict[str, List[str]] = defaultdict(list)
    for image_id in ids:
        path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image for grouped id {image_id} not found: {path}")
        exact_hashes[_sha256_file(path)].append(image_id)
    exact_edges = sum(
        union.union(members[0], other)
        for members in exact_hashes.values() for other in members[1:]
    )

    near_edges = 0
    if near_duplicates:
        dhashes: Dict[str, Optional[int]] = {
            image_id: _dhash(os.path.join(images_dir, f"{image_id}.jpg")) for image_id in ids
        }
        near_pairs = _near_duplicate_pairs(dhashes, int(hash_bits))
        near_edges = sum(union.union(a, b) for a, b in near_pairs)

    components: Dict[str, List[str]] = defaultdict(list)
    for image_id in ids:
        components[union.find(image_id)].append(image_id)
    groups = {
        f"g{index:05d}": sorted(members, key=_id_sort_key)
        for index, (_, members) in enumerate(sorted(components.items(), key=lambda item: _id_sort_key(item[1][0])))
    }
    if verbose:
        sizes = Counter(len(members) for members in groups.values())
        print(f"Groups: {len(groups)} (address edges={address_edges}, "
              f"exact-hash edges={exact_edges}, "
              f"near-dup edges={near_edges} [{'on' if near_duplicates else 'off'}])")
        print(f"Group-size distribution: {dict(sorted(sizes.items()))} | "
              f"largest={max(sizes) if sizes else 0}")
    return groups


def _majority_bin(bins: Sequence[int]) -> int:
    counts = Counter(bins)
    return min(bin_id for bin_id, count in counts.items() if count == max(counts.values()))


def make_group_split(
    records: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Group-stratified 70/15/15 split using the project's temporary pps bins."""
    if len(groups) < 6:
        raise ValueError(f"At least 6 groups are needed for a 70/15/15 split (got {len(groups)}).")

    pps = np.array([float(record["price_per_sqft"]) for record in records], dtype=float)
    record_bins = pd.qcut(pps, q=3, labels=False, duplicates="drop")
    if pd.isna(record_bins).any():
        raise ValueError("Stratification failed: non-finite price-per-sqft values.")
    bins = {str(record["image_id"]): int(value) for record, value in zip(records, record_bins)}
    if len(set(bins.values())) < 2:
        raise ValueError("Stratification failed: fewer than two price-per-sqft bins.")

    ordered = sorted(groups, key=lambda group: min(_id_sort_key(i) for i in groups[group]))
    group_bins = np.array([_majority_bin([bins[i] for i in groups[g]]) for g in ordered])
    for bin_id in np.unique(group_bins):
        if int((group_bins == bin_id).sum()) < 2:
            raise ValueError(f"Stratification bin {bin_id} has too few groups for a split.")

    train_groups, temp_groups = train_test_split(
        np.array(ordered), test_size=0.30, random_state=seed, stratify=group_bins,
    )
    lookup = dict(zip(ordered, group_bins.tolist()))
    temp_bins = np.array([lookup[group] for group in temp_groups])
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.50, random_state=seed + 1, stratify=temp_bins,
    )

    owner = {}
    for group in train_groups:
        owner[group] = "train"
    for group in val_groups:
        owner[group] = "val"
    for group in test_groups:
        owner[group] = "test"

    split: Dict[str, List[str]] = {name: [] for name in ("train", "val", "test")}
    for group, members in groups.items():
        split[owner[group]].extend(members)
    for name in split:
        split[name] = sorted(split[name], key=_id_sort_key)

    all_ids = [str(record["image_id"]) for record in records]
    train_set, val_set, test_set = (set(split[name]) for name in ("train", "val", "test"))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Internal error: group split overlaps.")
    if train_set | val_set | test_set != set(all_ids):
        raise AssertionError("Internal error: group split does not cover every record.")
    owner_of_id = {image_id: name for name, ids in split.items() for image_id in ids}
    for group, members in groups.items():
        if len({owner_of_id[image_id] for image_id in members}) != 1:
            raise AssertionError(f"Internal error: group {group} was divided.")
    if verbose:
        total = len(all_ids)
        print("Group split -> " + " | ".join(
            f"{name}: {len(split[name])} ({len(split[name]) / total:.1%})" for name in ("train", "val", "test")
        ))
    return split


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    split_path = os.path.join(args.output_dir, "split_indices.json")
    manifest_path = os.path.join(args.output_dir, "group_manifest.json")
    if not args.allow_existing:
        for path in (split_path, manifest_path):
            if os.path.exists(path):
                raise FileExistsError(
                    f"Refusing to overwrite {path}. Use a new --output_dir or pass --allow_existing."
                )
    if args.hash_bits < 0:
        raise ValueError("--hash_bits must be non-negative.")

    # Lazy import: keeps the pure split functions importable without torch.
    from src.dataset import (  # noqa: PLC0415
        ID_ALIASES, _normalize_image_id, detect_column, load_and_validate_records,
    )

    records, skipped, detected = load_and_validate_records(
        csv_path=args.csv_path, images_dir=args.images_dir,
    )
    frame = pd.read_csv(args.csv_path)
    id_col = detect_column(frame, ID_ALIASES, None, "image id")
    if args.street_col not in frame.columns:
        raise KeyError(f"Street column {args.street_col!r} not found; available: {list(frame.columns)}")
    streets = {
        _normalize_image_id(row[id_col]): row[args.street_col]
        for _, row in frame.iterrows()
    }

    groups = build_groups(
        records, streets, args.images_dir, args.hash_bits,
        near_duplicates=args.near_duplicates,
    )
    split = make_group_split(records, groups, args.seed)
    missing = missing_address_ids(streets, [str(r["image_id"]) for r in records])

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "strategy": ("property_address_plus_exact_hash_plus_global_dhash"
                     if args.near_duplicates else "property_address_plus_exact_hash"),
        "seed": args.seed,
        "near_duplicates": bool(args.near_duplicates),
        "hash_bits": args.hash_bits if args.near_duplicates else None,
        "missing_address_ids": missing,
        "skipped_records": skipped,
        "detected_columns": detected,
        "n_groups": len(groups),
        "group_sizes": {group: len(members) for group, members in groups.items()},
        "group_of_image_id": {image_id: group for group, members in groups.items() for image_id in members},
        "split_counts": {name: len(ids) for name, ids in split.items()},
    }
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump({name: split[name] for name in ("train", "val", "test")}, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Skipped records: {skipped}")
    print(f"Missing/placeholder addresses: {len(missing)} (examples: {missing[:5]})")
    print(f"Wrote {split_path} and {manifest_path}")


if __name__ == "__main__":
    main()
