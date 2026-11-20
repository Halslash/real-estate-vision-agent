"""Data preparation for Member A's CNN vision agent.

Single source of truth for:
  * Kaggle paths and preprocessing constants
  * CSV column detection and row validation
  * proxy-label construction (ALL statistics fitted on TRAIN data only)
  * stratified 70/15/15 splitting
  * image transforms and the PyTorch Dataset

Scientific note: labels are MARKET-DERIVED PROXIES. They are tertiles of
log(price / sqft) relative to a shrunken city median, not human annotations of
physical condition or aesthetic quality. Versioned label configurations keep
this methodology from silently changing under an existing checkpoint.

No file in this package is executed here; everything is meant to run on Kaggle.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.label_thresholds import sample_quantile_tertiles

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

# Versioned label methodology. The original baseline used raw price-per-sqft
# ratios and unshrunk city medians; that behaviour is preserved exactly for
# checkpoints that carry no version field. The improved default is v2; v3 is an
# opt-in experiment that only changes the tertile cut rule.
LABEL_V1 = "v1_legacy"
LABEL_V2 = "v2_log_shrunk"
LABEL_V3 = "v3_balanced_quantile"
# v3 is an OPT-IN experiment, not a silent default change: historical v2
# checkpoints must keep being evaluated with their own recorded thresholds.
DEFAULT_LABEL_CONFIG_VERSION = LABEL_V2
DEFAULT_SHRINKAGE_K = 10.0  # pseudo-count pulling small cities to the global median

# Versioned preprocessing metadata. Evaluation geometry (resize + center crop
# + ImageNet normalization) is unchanged; v2 records augmentation explicitly.
PREPROCESSING_VERSION = "v2"
AUGMENTATION_LEGACY = "legacy"
AUGMENTATION_IMPROVED = "improved"
DEFAULT_AUGMENTATION = AUGMENTATION_IMPROVED

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
    """Load the CSV and keep rows with valid metadata and an existing image file.

    Image contents are decoded lazily in ``HouseImageDataset.__getitem__`` so
    the initial scan stays fast. Returns (records, skipped_count, detected_columns).
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
        if not os.path.isfile(image_path):
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
def _ratio_for_record_legacy(record: Dict[str, Any], stats: Dict[str, Any]) -> float:
    """Baseline v1 proxy value: clipped price-per-sqft / raw city median.

    Kept byte-for-byte so the original checkpoint remains reproducible. Never
    modify this function to 'improve' labels -- change the v2 path instead.
    """
    pps = float(record["price_per_sqft"])
    pps = min(max(pps, stats["clip_p1"]), stats["clip_p99"])

    city = _city_key(record.get("city"))
    median = stats["city_median_pps"].get(city, stats["global_median_pps"]) if city else stats["global_median_pps"]
    if not np.isfinite(median) or median <= 0:
        median = stats["global_median_pps"]
    if not np.isfinite(median) or median <= 0:
        return pps
    return pps / median


def _log_value_for_record(record: Dict[str, Any], stats: Dict[str, Any]) -> float:
    """v2 proxy value: clipped log(price/sqft) minus shrunken city log median.

    Working in log space makes the quantity a log-ratio (a difference), which
    tames the right skew of price per square foot. The city denominator is
    shrunk toward the global training median, so a one-property city cannot
    define its own reference. All constants come from the saved stats.
    """
    pps = float(record["price_per_sqft"])
    if np.isfinite(pps) and pps > 0:
        log_pps = math.log(pps)
    else:
        # Loader normally rejects these; keep a deterministic fallback rather
        # than raising during inference on an externally supplied record.
        log_pps = float(stats["clip_log_p1"])
    log_pps = min(max(log_pps, stats["clip_log_p1"]), stats["clip_log_p99"])

    city = _city_key(record.get("city"))
    median = (
        stats["city_shrunk_log_median"].get(city, stats["global_median_log"])
        if city else stats["global_median_log"]
    )
    if not np.isfinite(median):
        median = stats["global_median_log"]
    if not np.isfinite(median):
        return log_pps  # last-resort fallback; global median is always finite
    return float(log_pps - median)


def _value_for_record(record: Dict[str, Any], stats: Dict[str, Any]) -> float:
    """Dispatch to the methodology recorded in the checkpoint itself."""
    version = stats.get("label_config_version", LABEL_V1)
    if version in (LABEL_V2, LABEL_V3):
        # v3 changes only WHERE the tertile cuts fall, not the proxy value itself.
        return _log_value_for_record(record, stats)
    if version == LABEL_V1:
        return _ratio_for_record_legacy(record, stats)
    raise ValueError(
        f"Unsupported label_config_version {version!r}. This checkpoint was "
        "produced by a different label methodology; load it with a matching "
        "version of this package instead of reinterpreting its thresholds."
    )


def _fit_legacy_stats(train_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproduce the original baseline proxy statistics (no shrinkage, raw pps)."""
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

    ratios = np.array([_ratio_for_record_legacy(r, {
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

    low = float(unique_ratios[len(unique_ratios) // NUM_CLASSES])
    high = float(unique_ratios[(2 * len(unique_ratios)) // NUM_CLASSES])

    return {
        "label_config_version": LABEL_V1,
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


def _fit_v2_stats(
    train_records: Sequence[Dict[str, Any]], shrinkage_k: float
) -> Dict[str, Any]:
    """Fit log-ratio tertiles with city medians shrunk toward the global median.

    Everything here is fitted on TRAIN records only. No validation or test row
    influences clipping bounds, the global median, city medians or thresholds.
    """
    if len(train_records) < 3:
        raise ValueError("Need at least 3 training records to fit proxy-label statistics.")
    k = float(shrinkage_k)
    if not np.isfinite(k) or k < 0:
        raise ValueError(
            f"--shrinkage_k must be a finite, non-negative number (got {shrinkage_k!r})."
        )

    pps = np.array([r["price_per_sqft"] for r in train_records], dtype=float)
    log_pps = np.full(pps.shape, np.nan, dtype=float)
    valid = np.isfinite(pps) & (pps > 0)
    log_pps[valid] = np.log(pps[valid])
    if int(valid.sum()) < 3:
        raise ValueError(
            "Fewer than 3 finite, positive price-per-square-foot values were found "
            "in the training split. Check the price and square-footage columns."
        )

    values = log_pps[valid]
    clip_log_p1, clip_log_p99 = (float(v) for v in np.percentile(values, [1, 99]))
    if clip_log_p1 >= clip_log_p99:
        clip_log_p1, clip_log_p99 = float(values.min()), float(values.max())

    clipped = np.clip(values, clip_log_p1, clip_log_p99)
    global_median = float(np.median(clipped))

    by_city: Dict[str, List[float]] = {}
    for record, value in zip(train_records, clipped):
        city = _city_key(record.get("city"))
        if city is not None and np.isfinite(value):
            by_city.setdefault(city, []).append(float(value))
    raw_medians = {city: float(np.median(vals)) for city, vals in by_city.items()}
    city_counts = {city: int(len(vals)) for city, vals in by_city.items()}
    # Shrinkage: a city with n samples contributes n/n+k of its own median and
    # k/n+k of the global training median. Tiny cities therefore cannot produce
    # a degenerate one-sample reference; large cities keep their own statistics.
    shrunk_medians = {
        city: (city_counts[city] * median + k * global_median)
        / (city_counts[city] + k)
        for city, median in raw_medians.items()
    }

    stats: Dict[str, Any] = {
        "label_config_version": LABEL_V2,
        "value_transform": "log_price_per_sqft",
        "shrinkage_k": k,
        "clip_log_p1": clip_log_p1,
        "clip_log_p99": clip_log_p99,
        "global_median_log": global_median,
        "city_raw_log_median": raw_medians,
        "city_counts": city_counts,
        "city_shrunk_log_median": shrunk_medians,
        "label_names": list(LABEL_NAMES),
        "num_classes": NUM_CLASSES,
        "ratio_definition": (
            "clipped_log(price/sqft) - shrunken_city_log_median "
            "(log-ratio; global training median for missing/unseen cities)"
        ),
        "fitted_on": "train split only",
        "n_train_records": len(train_records),
    }

    proxy_values = np.array(
        [_log_value_for_record(r, stats) for r in train_records], dtype=float
    )
    proxy_values = proxy_values[np.isfinite(proxy_values)]
    unique_values = np.unique(proxy_values)
    if len(unique_values) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(unique_values)} distinct log-ratio value(s) in the training "
            "split; 3 ordered proxy classes cannot be formed. Increase --max_images "
            "or check the price / square-footage columns; refusing to fit thresholds "
            "that would produce an empty class."
        )

    # Observed distinct values (not interpolated percentiles) avoid duplicate
    # thresholds when many values are tied at the extremes.
    low = float(unique_values[len(unique_values) // NUM_CLASSES])
    high = float(unique_values[(2 * len(unique_values)) // NUM_CLASSES])
    if not (low < high):
        raise ValueError(
            "Tertile thresholds collapsed (low >= high); the training log-ratio "
            "values contain too few distinct values for three ordered classes. "
            "Increase --max_images or use a larger dataset."
        )
    stats["class_threshold_low"] = low
    stats["class_threshold_high"] = high
    return stats


def _fit_v3_stats(
    train_records: Sequence[Dict[str, Any]], shrinkage_k: float
) -> Dict[str, Any]:
    """v3 keeps v2's proxy values and medians but cuts at SAMPLE tertiles.

    v2 cuts at the 1/3 and 2/3 positions of the *distinct* sorted values, so a
    heavily repeated proxy value counts once and can starve the outer classes.
    v3 uses sample-weighted quantiles; tied values still share one label, so
    residual imbalance is possible and is reported in the experiment docs.

    Built on top of the untouched v2 fitter so that v2 statistics stay frozen.
    """
    stats = _fit_v2_stats(train_records, shrinkage_k)
    proxy_values = np.array(
        [_log_value_for_record(r, stats) for r in train_records], dtype=float
    )
    proxy_values = proxy_values[np.isfinite(proxy_values)]
    low, high = sample_quantile_tertiles(proxy_values)
    stats["label_config_version"] = LABEL_V3
    stats["class_threshold_low"] = low
    stats["class_threshold_high"] = high
    stats["threshold_rule"] = "sample_weighted_quantiles"
    return stats


def fit_preprocessing_stats(
    train_records: Sequence[Dict[str, Any]],
    *,
    label_config_version: str = DEFAULT_LABEL_CONFIG_VERSION,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
) -> Dict[str, Any]:
    """Fit clipping bounds, medians and tertile thresholds on TRAIN records.

    ``label_config_version=LABEL_V1`` reproduces the original baseline exactly;
    v2 adds log-ratio values and shrunken city medians; v3 is v2 with balanced
    sample tertiles (opt-in experiment). The default stays v2 so existing
    checkpoints are never reinterpreted. The chosen version is written into the
    returned stats and therefore into the checkpoint.
    """
    if label_config_version == LABEL_V1:
        return _fit_legacy_stats(train_records)
    if label_config_version == LABEL_V2:
        return _fit_v2_stats(train_records, shrinkage_k)
    if label_config_version == LABEL_V3:
        return _fit_v3_stats(train_records, shrinkage_k)
    raise ValueError(
        f"Unsupported label_config_version {label_config_version!r}; "
        f"choose one of {LABEL_V1!r}, {LABEL_V2!r}, {LABEL_V3!r}."
    )


def compute_labels(
    records: Sequence[Dict[str, Any]], stats: Dict[str, Any], inplace: bool = True
) -> List[Dict[str, Any]]:
    """Attach ``label`` and ``proxy_value`` using the checkpoint's own methodology.

    ``ratio`` is kept as an alias for backwards compatibility. Under v2 it holds
    a log-ratio (a difference in log space), not a division -- see
    ``stats['ratio_definition']``.
    """
    labeled = list(records) if inplace else [dict(r) for r in records]
    for record in labeled:
        value = _value_for_record(record, stats)
        record["proxy_value"] = float(value)
        record["ratio"] = float(value)
        record["label"] = int(
            0 if value < stats["class_threshold_low"]
            else (1 if value < stats["class_threshold_high"] else 2)
        )
    return labeled


def validate_labeled_split(
    labeled: Dict[str, List[Dict[str, Any]]], require_test: bool = True
) -> None:
    """Fail clearly on small datasets instead of training an invalid split.

    ``require_test=False`` is only for OOF folds, whose test list is empty by
    design because the untouched test split is reserved for final evaluation.
    """
    names = ("train", "val", "test") if require_test else ("train", "val")
    for name in names:
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
def build_train_transform(
    image_size: int = IMAGE_SIZE,
    resize_size: int = RESIZE_SIZE,
    augmentation: str = DEFAULT_AUGMENTATION,
):
    """Augmentations used only during training.

    v1 (legacy) is preserved for controlled experiments with the baseline
    recipe. The improved default deliberately keeps geometry aligned with
    evaluation: no rotation (houses are upright), no aggressive random crop
    (rooflines and facades carry signal), and restrained colour jitter so
    material cues are not destroyed. Horizontal flip is retained because it is
    geometry-preserving for the market-derived target.
    """
    if augmentation == AUGMENTATION_LEGACY:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if augmentation != AUGMENTATION_IMPROVED:
        raise ValueError(
            f"Unknown augmentation {augmentation!r}; choose "
            f"{AUGMENTATION_IMPROVED!r} or {AUGMENTATION_LEGACY!r}."
        )
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),
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
        # Images are only checked for existence during record loading, so decode
        # here. The bounded retry skips corrupt files; each returned pair keeps
        # the label of the image that was actually read.
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
