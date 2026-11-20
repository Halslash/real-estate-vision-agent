"""Data preparation for Member A's CNN vision agent.

Single source of truth for:
  * Kaggle paths and preprocessing constants
  * CSV column detection and row validation
  * proxy-label construction (ALL statistics fitted on TRAIN data only)
  * stratified 70/15/15 splitting
  * image transforms and the PyTorch Dataset

No file in this package is executed here; everything is meant to run on Kaggle.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Shared constants (imported by train.py, evaluate.py and vision_agent.py)
# ---------------------------------------------------------------------------
DEFAULT_IMAGES_DIR = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal_pics/"
)
DEFAULT_CSV_PATH = (
    "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/"
    "Real_estatewith_image/archive-2/socal2/socal2.csv"
)
DEFAULT_OUTPUT_DIR = "/kaggle/working/output"

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_NAMES = ["low", "average", "high"]
NUM_CLASSES = 3
IDX_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}

ID_ALIASES = (
    "image_id", "imageid", "id", "img_id", "imgid", "image", "image_name",
    "imagename", "filename", "file_name", "picture_id", "pic_id",
)
PRICE_ALIASES = (
    "price", "listing_price", "listingprice", "sale_price", "saleprice",
    "sold_price", "market_price", "price_usd", "cost",
)
SQFT_ALIASES = (
    "sqft", "sq_ft", "sqft_living", "sqft_total", "square_feet",
    "squarefeet", "square_feet_living", "living_area", "property_sqft",
)
CITY_ALIASES = (
    "city", "citi", "n_citi", "nciti", "city_name", "town", "location_city",
)

# Substring matching is restricted to distinctive tokens. Generic words such as
# "area", "size", "value" or "name" are deliberately excluded because they would
# happily match "lot_area", "city_name" or "assessed_value" instead.
ID_FUZZY_ALIASES = ("image_id", "imageid", "img_id", "imgid", "filename", "file_name")
PRICE_FUZZY_ALIASES = (
    "listing_price", "listingprice", "sale_price", "saleprice", "sold_price", "price",
)
SQFT_FUZZY_ALIASES = ("sqft", "square_feet", "squarefeet", "living_area")
CITY_FUZZY_ALIASES = ("city", "n_citi", "nciti")

MISSING_TOKENS = {"", "nan", "none", "null", "na"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _normalize_name(name: Any) -> str:
    """Turn a CSV header into a comparable key: 'Sale Price' -> 'sale_price'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    explicit: Optional[str] = None,
    role: str = "column",
    fuzzy_aliases: Sequence[str] = (),
) -> Optional[str]:
    """Find a column case-insensitively, preferring an explicit override.

    Exact (normalized) alias matches win first. Only the distinctive
    ``fuzzy_aliases`` may match substrings; if more than one column matches we
    raise instead of guessing. Returns None when no optional column (city)
    could be detected.
    """
    columns = list(df.columns)
    normalized = {_normalize_name(c): c for c in columns}

    if explicit is not None:
        if explicit in columns:
            return explicit
        if _normalize_name(explicit) in normalized:
            return normalized[_normalize_name(explicit)]
        raise KeyError(
            f"Explicit {role} column {explicit!r} was not found in the CSV.\n"
            f"Available columns: {columns}"
        )

    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    candidates = []
    for key, original in normalized.items():
        if "per_sqft" in key or "per_square" in key:
            continue  # e.g. price_per_sqft is derived, not the raw price
        if any(token in key for token in fuzzy_aliases):
            candidates.append(original)
    candidates = list(dict.fromkeys(candidates))

    if len(candidates) > 1:
        raise KeyError(
            f"Ambiguous {role} column candidates: {candidates}.\n"
            "Pass the exact column with --id_col / --price_col / --sqft_col / "
            "--city_col instead of letting detection guess."
        )
    return candidates[0] if candidates else None


def _normalize_image_id(value: Any) -> Optional[str]:
    """Map '42', 42.0, '42.jpg' or 'a/b/42.jpg' to the stem '42'."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip().replace("\\", "/")
    if _normalize_name(text) in MISSING_TOKENS:
        return None
    stem = os.path.splitext(os.path.basename(text))[0].strip()
    if _normalize_name(stem) in MISSING_TOKENS:
        return None
    try:
        number = float(stem)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return stem


def _city_key(value: Any) -> Optional[str]:
    """Return a clean string city key, or None when city data is unusable."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if _normalize_name(text) in MISSING_TOKENS:
        return None
    return text


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _is_valid_image(path: str) -> bool:
    """Detect truncated / unreadable image files without raising."""
    try:
        with Image.open(path) as image:
            image.convert("RGB")  # forces pixel decoding, catches corrupted files
        return True
    except Exception:
        return False


def _id_sort_key(image_id: str) -> Tuple[int, Any]:
    return (0, int(image_id)) if image_id.isdigit() else (1, image_id)


# ---------------------------------------------------------------------------
# CSV -> validated records
# ---------------------------------------------------------------------------
def load_and_validate_records(
    csv_path: str,
    images_dir: str,
    id_col: Optional[str] = None,
    price_col: Optional[str] = None,
    sqft_col: Optional[str] = None,
    city_col: Optional[str] = None,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Optional[str]]]:
    """Load the CSV and keep only rows backed by a readable JPEG file.

    Returns (records, skipped_count, detected_columns).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Images directory not found: {images_dir}")

    df = pd.read_csv(csv_path)
    detected = {
        "id": detect_column(df, ID_ALIASES, id_col, "image id", ID_FUZZY_ALIASES),
        "price": detect_column(df, PRICE_ALIASES, price_col, "price", PRICE_FUZZY_ALIASES),
        "sqft": detect_column(df, SQFT_ALIASES, sqft_col, "square footage", SQFT_FUZZY_ALIASES),
        "city": detect_column(df, CITY_ALIASES, city_col, "city (optional)", CITY_FUZZY_ALIASES),
    }

    missing = [
        role for role in ("id", "price", "sqft") if detected[role] is None
    ]
    if missing:
        raise KeyError(
            f"Could not detect required column(s): {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Pass --id_col, --price_col and/or --sqft_col explicitly."
        )

    records: List[Dict[str, Any]] = []
    skipped = 0
    seen_ids = set()

    for _, row in df.iterrows():
        image_id = _normalize_image_id(row[detected["id"]])
        price = _positive_float(row[detected["price"]])
        sqft = _positive_float(row[detected["sqft"]])

        if image_id is None or price is None or sqft is None:
            skipped += 1
            continue
        if image_id in seen_ids:  # guarantees an image cannot land in two splits
            skipped += 1
            continue

        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.isfile(image_path) or not _is_valid_image(image_path):
            skipped += 1
            continue

        seen_ids.add(image_id)
        city = _city_key(row[detected["city"]]) if detected["city"] else None
        records.append({
            "image_id": image_id,
            "image_path": image_path,
            "price": price,
            "sqft": sqft,
            "price_per_sqft": price / sqft,
            "city": city,
        })

    records.sort(key=lambda record: _id_sort_key(record["image_id"]))
    if max_images is not None and int(max_images) > 0:
        records = records[: int(max_images)]

    if verbose:
        print(f"Loaded CSV: {csv_path}")
        print(f"Detected columns: {detected}")
        print(f"Total valid samples: {len(records)}")
        print(f"Skipped samples: {skipped}")
    return records, skipped, detected


# ---------------------------------------------------------------------------
# Splitting (stratification uses only a temporary, non-label binning)
# ---------------------------------------------------------------------------
def make_stratified_split(
    records: Sequence[Dict[str, Any]], seed: int = 42, verbose: bool = True
) -> Dict[str, List[str]]:
    """Return 70/15/15 train/val/test image-id lists with no overlap.

    The stratification bins are temporary quantiles of raw price-per-sqft; they
    are used ONLY to balance the split. The actual proxy labels are fitted
    afterwards on training data by ``fit_preprocessing_stats``.
    """
    if len(records) < 6:
        raise ValueError(
            "At least 6 valid samples are required for a stratified 70/15/15 "
            f"split (got {len(records)}). Increase --max_images or supply more data."
        )

    keys = np.array([r["image_id"] for r in records])
    pps = np.array([r["price_per_sqft"] for r in records], dtype=float)
    strata = np.asarray(pd.qcut(pps, q=3, labels=False, duplicates="drop"), dtype=int)

    if len(np.unique(strata)) < 2:
        raise ValueError(
            "Stratification failed: price-per-square-foot values are too uniform "
            "to form at least two bins. Check the price and square-footage columns."
        )
    for stratum in np.unique(strata):
        count = int((strata == stratum).sum())
        if count < 2:
            raise ValueError(
                f"Stratification bin {stratum} has only {count} sample(s); at least "
                "2 are needed. Increase --max_images or provide a larger dataset."
            )

    train_keys, temp_keys = train_test_split(
        keys, test_size=0.30, random_state=seed, stratify=strata
    )
    key_to_stratum = dict(zip(keys.tolist(), strata.tolist()))
    temp_strata = np.array([key_to_stratum[k] for k in temp_keys])
    for stratum in np.unique(temp_strata):
        count = int((temp_strata == stratum).sum())
        if count < 2:
            raise ValueError(
                "Not enough samples in a stratification bin to build both a "
                f"validation and a test split (bin {stratum} has {count}). "
                "Increase --max_images or provide a larger dataset."
            )

    val_keys, test_keys = train_test_split(
        temp_keys, test_size=0.50, random_state=seed + 1, stratify=temp_strata
    )

    split = {
        "train": sorted(train_keys.tolist()),
        "val": sorted(val_keys.tolist()),
        "test": sorted(test_keys.tolist()),
    }
    train_set, val_set, test_set = (set(split[name]) for name in ("train", "val", "test"))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Internal error: train/val/test splits overlap.")
    if verbose:
        print(f"Split sizes -> train: {len(split['train'])}, "
              f"val: {len(split['val'])}, test: {len(split['test'])}")
    return split


def split_records(
    records: Sequence[Dict[str, Any]], split: Dict[str, List[str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by split using their image ids."""
    grouped = {}
    for name in ("train", "val", "test"):
        wanted = set(split[name])
        grouped[name] = [r for r in records if r["image_id"] in wanted]
    return grouped


# ---------------------------------------------------------------------------
# Proxy labels -- fit on TRAIN only, apply everywhere
# ---------------------------------------------------------------------------
def _ratio_for_record(record: Dict[str, Any], stats: Dict[str, Any]) -> float:
    pps = float(record["price_per_sqft"])
    pps = min(max(pps, stats["clip_p1"]), stats["clip_p99"])

    city = _city_key(record.get("city"))
    median = stats["city_median_pps"].get(city, stats["global_median_pps"]) if city else stats["global_median_pps"]
    if not np.isfinite(median) or median <= 0:
        median = stats["global_median_pps"]
    if not np.isfinite(median) or median <= 0:
        return pps
    return pps / median


def fit_preprocessing_stats(
    train_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit clipping bounds, medians and tertile thresholds on TRAIN records."""
    if len(train_records) < 3:
        raise ValueError("Need at least 3 training records to fit proxy-label statistics.")

    pps = np.array([r["price_per_sqft"] for r in train_records], dtype=float)
    clip_p1, clip_p99 = (float(v) for v in np.percentile(pps, [1, 99]))
    if clip_p1 >= clip_p99:
        clip_p1, clip_p99 = float(pps.min()), float(pps.max())

    clipped = np.clip(pps, clip_p1, clip_p99)
    global_median = float(np.median(clipped))

    by_city: Dict[str, List[float]] = {}
    for record, value in zip(train_records, clipped):
        city = _city_key(record.get("city"))
        if city is not None:
            by_city.setdefault(city, []).append(float(value))
    city_medians = {city: float(np.median(values)) for city, values in by_city.items()}

    ratios = np.array([_ratio_for_record(r, {
        "clip_p1": clip_p1,
        "clip_p99": clip_p99,
        "city_median_pps": city_medians,
        "global_median_pps": global_median,
    }) for r in train_records], dtype=float)

    unique_ratios = np.unique(ratios)
    if len(unique_ratios) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(unique_ratios)} distinct price-per-sqft ratio value(s) in "
            "the training split; 3 proxy classes cannot be formed. Increase "
            "--max_images or check the price / square-footage columns."
        )

    # Thresholds come from observed distinct values, not interpolated
    # percentiles. Percentiles can equal the minimum/maximum when values are
    # tied, which would leave the 'low' or 'high' class empty. With continuous
    # data these are effectively the same as ordinary tertiles.
    low = float(unique_ratios[len(unique_ratios) // NUM_CLASSES])
    high = float(unique_ratios[(2 * len(unique_ratios)) // NUM_CLASSES])

    return {
        "clip_p1": clip_p1,
        "clip_p99": clip_p99,
        "global_median_pps": global_median,
        "city_median_pps": city_medians,
        "class_threshold_low": low,
        "class_threshold_high": high,
        "label_names": list(LABEL_NAMES),
        "num_classes": NUM_CLASSES,
        "ratio_definition": "clipped_price_per_sqft / city_median (global median fallback)",
        "fitted_on": "train split only",
        "n_train_records": len(train_records),
    }


def compute_labels(
    records: Sequence[Dict[str, Any]], stats: Dict[str, Any], inplace: bool = True
) -> List[Dict[str, Any]]:
    """Attach ``label`` and ``ratio`` to each record using saved statistics."""
    labeled = list(records) if inplace else [dict(r) for r in records]
    for record in labeled:
        ratio = _ratio_for_record(record, stats)
        record["ratio"] = float(ratio)
        record["label"] = int(
            0 if ratio < stats["class_threshold_low"]
            else (1 if ratio < stats["class_threshold_high"] else 2)
        )
    return labeled


def validate_labeled_split(
    labeled: Dict[str, List[Dict[str, Any]]]
) -> None:
    """Fail clearly on small datasets instead of training an invalid split."""
    for name in ("train", "val", "test"):
        if not labeled[name]:
            raise ValueError(
                f"The {name} split is empty. Increase --max_images or use a "
                "larger dataset."
            )
    counts = np.bincount(
        [r["label"] for r in labeled["train"]], minlength=NUM_CLASSES
    )
    if (counts == 0).any():
        raise ValueError(
            "The training split does not contain all 3 proxy classes "
            f"(low/average/high = {counts.tolist()}). Increase --max_images or "
            "use a larger dataset; refusing to train an invalid split."
        )


def print_summary(
    records: Sequence[Dict[str, Any]],
    skipped: int,
    split: Dict[str, List[str]],
    labeled: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Print the sample accounting and class distributions required by the task."""
    print(f"Total valid samples: {len(records)}")
    print(f"Skipped samples: {skipped}")
    all_counts = np.bincount([r["label"] for r in records], minlength=NUM_CLASSES)
    print("Overall class distribution: " + ", ".join(
        f"{LABEL_NAMES[i]}={int(all_counts[i])}" for i in range(NUM_CLASSES)
    ))
    for name in ("train", "val", "test"):
        subset = labeled[name]
        counts = np.bincount([r["label"] for r in subset], minlength=NUM_CLASSES)
        print(f"{name} samples: {len(subset)} | class distribution " + ", ".join(
            f"{LABEL_NAMES[i]}={int(counts[i])}" for i in range(NUM_CLASSES)
        ))


# ---------------------------------------------------------------------------
# Transforms and Dataset
# ---------------------------------------------------------------------------
def build_train_transform(image_size: int = IMAGE_SIZE):
    """Augmentations used only during training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(preprocessing: Optional[Dict[str, Any]] = None):
    """Deterministic transform shared by validation, testing and inference."""
    preprocessing = preprocessing or {}
    image_size = int(preprocessing.get("image_size", IMAGE_SIZE))
    resize_size = int(preprocessing.get("resize_size", RESIZE_SIZE))
    mean = preprocessing.get("mean", IMAGENET_MEAN)
    std = preprocessing.get("std", IMAGENET_STD)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HouseImageDataset(Dataset):
    """Maps validated records to (image tensor, integer proxy label)."""

    def __init__(self, records: Sequence[Dict[str, Any]], transform=None):
        self.samples = [(r["image_path"], int(r["label"])) for r in records]
        self.transform = transform
        if not self.samples:
            raise ValueError("Cannot build an empty HouseImageDataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # Images are pre-validated in load_and_validate_records; this bounded
        # retry is a last-resort guard against a file that was corrupted later.
        # Each returned pair still uses the label of the image actually read.
        for offset in range(len(self.samples)):
            path, label = self.samples[(index + offset) % len(self.samples)]
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("No readable images remain in HouseImageDataset.")

        if self.transform is not None:
            image = self.transform(image)
        return image, label
