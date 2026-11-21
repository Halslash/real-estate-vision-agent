# Member A — one-cell Kaggle preflight (and later training command)

**The cell below is PREFLIGHT ONLY.** It clones and verifies the pinned commit,
loads the mounted dataset, generates the simplified leakage-resistant split,
runs the essential read-only audit, verifies every guarantee, and writes a
machine-readable `verification.json`. It does **not** train the CNN, load model
weights, run test inference, or evaluate test metrics. Do not bypass any assert.

---

## The one cell

```python
import os, sys, json, subprocess
from collections import defaultdict
import numpy as np
import pandas as pd

PIN = "bc39d102d2e9cde83144966d6e6c910926109429"
REPO = "/kaggle/working/real-estate-vision-agent"
URL = "https://github.com/Halslash/real-estate-vision-agent.git"
CSV = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv"
IMG = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics"
OUT = "/kaggle/working/output/leakage_resistant_split"
EXPECTED_RECORDS = 15474

# 0. Clone if needed, then check out the exact pinned commit in detached HEAD.
if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", URL, REPO], check=True)
assert os.path.isdir(os.path.join(REPO, ".git")), f"Not a git checkout: {REPO}"
subprocess.run(["git", "fetch", "--all", "--tags"], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", PIN], cwd=REPO, check=True)
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert HEAD == PIN, f"HEAD {HEAD} != pinned {PIN}; stop."
print("PIN verified:", HEAD)
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.dataset import (ID_ALIASES, _normalize_image_id, detect_column,
                         load_and_validate_records, fit_preprocessing_stats,
                         compute_labels, split_records)
from src.group_split import normalize_address, _sha256_file
from src.train import validate_supplied_split

# 1. Refuse to reuse/regenerate an existing split artifact.
assert not os.path.exists(os.path.join(OUT, "split_indices.json")), (
    "Split already exists; do not silently regenerate. Delete OUT explicitly to retry."
)
assert os.path.isfile(CSV), CSV
assert os.path.isdir(IMG), IMG

# 2. Load the real dataset; report skipped rows and (below) missing images.
records, skipped, detected = load_and_validate_records(CSV, IMG, verbose=False)
print("detected columns:", detected)
print("valid records:", len(records), "| skipped rows:", skipped)
assert len(records) == EXPECTED_RECORDS, (
    f"Expected {EXPECTED_RECORDS} valid records, got {len(records)}; stop."
)

# 3. Generate the simplified split: address + exact SHA-256, seed 42, dHash OFF.
subprocess.run([sys.executable, "src/group_split.py",
                "--csv_path", CSV, "--images_dir", IMG,
                "--output_dir", OUT, "--seed", "42"], cwd=REPO, check=True)
split = json.load(open(os.path.join(OUT, "split_indices.json"), encoding="utf-8"))
manifest = json.load(open(os.path.join(OUT, "group_manifest.json"), encoding="utf-8"))
assert manifest["seed"] == 42 and manifest["near_duplicates"] is False, manifest
assert manifest["strategy"] == "property_address_plus_exact_hash", manifest["strategy"]
print("strategy:", manifest["strategy"], "| groups:", manifest["n_groups"])

# 4. Essential read-only audit of the generated split (all images).
AUDIT = os.path.join(OUT, "data_audit.json")
subprocess.run([sys.executable, "src/data_audit.py",
                "--csv_path", CSV, "--images_dir", IMG,
                "--split_file", os.path.join(OUT, "split_indices.json"),
                "--output", AUDIT], cwd=REPO, check=True)
a = json.load(open(AUDIT, encoding="utf-8"))
print("rows in CSV:", a["rows_in_csv"], "| rows scanned:", a["rows_scanned"],
      "| matched images:", a["images_matched"], "| missing image files:", a["missing_image_files"])
assert a["rows_in_csv"] == EXPECTED_RECORDS
assert a["rows_scanned"] == EXPECTED_RECORDS
assert a["images_matched"] == EXPECTED_RECORDS
assert a["missing_image_files"] == 0, a["missing_image_files"]
assert a["exact_duplicate_groups_across_splits"] == 0, a["exact_duplicate_groups_across_splits"]
print("same-street near-dup pairs across splits (diagnostic only):",
      a["near_duplicate_pairs_across_splits"])

# 5. Fail-closed split validation, then every-valid-ID-exactly-once.
validate_supplied_split(split, records)
all_ids = [i for name in ("train", "val", "test") for i in split[name]]
known = {r["image_id"] for r in records}
assert len(all_ids) == len(records) and set(all_ids) == known, "IDs are not exactly once."
owner = {i: name for name in ("train", "val", "test") for i in split[name]}

# 6. Independent checks: normalized address and exact SHA-256 never cross splits.
frame = pd.read_csv(CSV)
id_col = detect_column(frame, ID_ALIASES, None, "image id")
assert id_col is not None and "street" in frame.columns
streets = {_normalize_image_id(row[id_col]): row["street"] for _, row in frame.iterrows()}
addr_splits, sha_splits = defaultdict(set), defaultdict(set)
for iid, name in owner.items():
    address = normalize_address(streets.get(iid))
    if address is not None:
        addr_splits[address].add(name)
    sha_splits[_sha256_file(os.path.join(IMG, iid + ".jpg"))].add(name)
assert all(len(v) == 1 for v in addr_splits.values()), "A normalized address crosses splits."
assert all(len(v) == 1 for v in sha_splits.values()), "An exact SHA-256 image group crosses splits."

# The manifest's own group invariant (address + exact-SHA connected components).
gmembers = defaultdict(list)
for iid, gid in manifest["group_of_image_id"].items():
    gmembers[gid].append(iid)
assert all(len({owner[i] for i in m}) == 1 for m in gmembers.values())

fractions = {name: len(split[name]) / len(records) for name in ("train", "val", "test")}
for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
    assert abs(fractions[name] - target) < 0.02, (name, fractions[name])
print("split counts:", {n: len(split[n]) for n in split},
      "| proportions:", {n: round(f, 4) for n, f in fractions.items()})

# 7. Class counts; thresholds fitted on TRAIN ONLY. No model and no test metrics.
train_recs = [r for r in records if r["image_id"] in set(split["train"])]
stats = fit_preprocessing_stats(train_recs, label_config_version="v2_log_shrunk", shrinkage_k=10.0)
assert stats["fitted_on"] == "train split only" and stats["n_train_records"] == len(train_recs)
labeled = split_records(compute_labels(records, stats, inplace=False), split)
counts = {n: np.bincount([r["label"] for r in labeled[n]], minlength=3).tolist()
          for n in ("train", "val", "test")}
assert all(all(c > 0 for c in v) for v in counts.values()), counts
print("class counts (low/average/high):", counts)
print("train-fitted thresholds:", stats["class_threshold_low"], stats["class_threshold_high"])

# 8. Machine-readable verification, written only after every check passed.
verification = {
    "status": "PASS",
    "pinned_sha": PIN,
    "head_sha": HEAD,
    "dataset": {
        "csv": CSV, "images": IMG,
        "rows_in_csv": a["rows_in_csv"], "valid_records": len(records),
        "expected_records": EXPECTED_RECORDS, "skipped_records": skipped,
        "missing_image_files": a["missing_image_files"],
    },
    "split": {
        "strategy": manifest["strategy"], "seed": manifest["seed"],
        "near_duplicates": manifest["near_duplicates"], "n_groups": manifest["n_groups"],
        "counts": {n: len(split[n]) for n in ("train", "val", "test")},
        "proportions": {n: round(fractions[n], 6) for n in fractions},
        "class_counts": counts,
        "missing_address_ids": len(manifest["missing_address_ids"]),
    },
    "audit": {
        "rows_scanned": a["rows_scanned"], "images_matched": a["images_matched"],
        "exact_duplicate_groups_across_splits": a["exact_duplicate_groups_across_splits"],
        "same_street_near_duplicate_pairs_across_splits": a["near_duplicate_pairs_across_splits"],
        "street_groups_spanning_splits": a["street_overlap"]["groups_in_multiple_splits"],
    },
    "checks": {
        "head_matches_pin": True, "every_valid_id_exactly_once": True,
        "no_address_crosses_splits": True, "no_exact_sha_group_crosses_splits": True,
        "no_group_crosses_splits": True, "split_proportions_ok": True,
        "train_only_preprocessing": True,
    },
    "preprocessing": {
        "fitted_on": stats["fitted_on"], "n_train_records": stats["n_train_records"],
        "class_threshold_low": stats["class_threshold_low"],
        "class_threshold_high": stats["class_threshold_high"],
        "label_config_version": stats["label_config_version"],
    },
    "note": "Preflight only: no training, no weights, no test inference/evaluation; global dHash OFF.",
}
with open(os.path.join(OUT, "verification.json"), "w", encoding="utf-8") as handle:
    json.dump(verification, handle, indent=2)
print("PREFLIGHT PASS ->", os.path.join(OUT, "verification.json"))
```

All artifacts stay in `/kaggle/working/output/leakage_resistant_split/`, never in
`/kaggle/working/output/experiments/`. The split guarantees and its limits are
documented in `docs/MEMBER_A_LEAKAGE_RESISTANT_SPLIT.md`; notably, visually
similar photos at different addresses are **not** exhaustively excluded because
global dHash is off by default. `EXPECTED_RECORDS = 15474` is the verified public
dataset size; if the mounted dataset differs, the cell fails rather than quietly
building a different split.

---

## Final CNN training cell — run once, only after the preflight cell passes

Selected configuration: `e0_reference` from `configs/experiments.json`
(MobileNetV2, `v2_log_shrunk`, CE + original `auto` weighting, seed 42, 30
epochs). Every flag is written out explicitly and verified against
`src/train.py`'s parser at the pinned commit. `train.py` is called directly with
`--split_file`; the random-split branch of `train.py` and `run_experiment.py`
are never involved. **This cell does not touch the test split for training,
preprocessing, threshold fitting, early stopping or selection.**

```python
import os, sys, json, hashlib, subprocess
import torch

PIN = "bc39d102d2e9cde83144966d6e6c910926109429"
REPO = "/kaggle/working/real-estate-vision-agent"
URL = "https://github.com/Halslash/real-estate-vision-agent.git"
CSV = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv"
IMG = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics"
SPLIT = "/kaggle/working/output/leakage_resistant_split/split_indices.json"
REPORT = "/kaggle/working/output/leakage_resistant_split/verification.json"
OUT = "/kaggle/working/output/experiments/e0_reference_group_split"
EXPECTED = {"train": 10829, "val": 2337, "test": 2308}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# 0. Verify the pinned code (clone if the notebook was reset).
if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", URL, REPO], check=True)
subprocess.run(["git", "fetch", "--all", "--tags"], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", PIN], cwd=REPO, check=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert head == PIN, f"HEAD {head} != pinned {PIN}; stop."
print("PIN verified:", head)

# 1. Gate on the verified preflight artifact: never regenerate a split here.
assert os.path.isfile(REPORT) and os.path.isfile(SPLIT), "Run the preflight cell first."
report = json.load(open(REPORT, encoding="utf-8"))
assert report["status"] == "PASS" and report["pinned_sha"] == PIN and report["head_sha"] == PIN
assert report["checks"]["head_matches_pin"] is True
assert report["checks"]["every_valid_id_exactly_once"] is True
assert report["checks"]["no_address_crosses_splits"] is True
assert report["checks"]["no_exact_sha_group_crosses_splits"] is True
assert report["dataset"]["valid_records"] == 15474 and report["dataset"]["missing_image_files"] == 0
assert report["audit"]["exact_duplicate_groups_across_splits"] == 0
assert report["audit"]["same_street_near_duplicate_pairs_across_splits"] == 0
assert report["split"]["strategy"] == "property_address_plus_exact_hash"
assert report["split"]["near_duplicates"] is False and report["split"]["seed"] == 42
assert report["split"]["counts"] == EXPECTED
assert report["preprocessing"]["fitted_on"] == "train split only"
assert report["preprocessing"]["label_config_version"] == "v2_log_shrunk"

split = json.load(open(SPLIT, encoding="utf-8"))
assert {name: len(split[name]) for name in ("train", "val", "test")} == EXPECTED
ids = [i for name in ("train", "val", "test") for i in split[name]]
assert len(ids) == len(set(ids)) == 15474, "Verified split is not exactly-once over 15474 ids."
split_sha, report_sha = sha256(SPLIT), sha256(REPORT)
print("Verified split:", EXPECTED, "| sha256:", split_sha[:12])

# 2. Fresh output directory (train.py also fails closed on an existing checkpoint).
assert not os.path.exists(os.path.join(OUT, "models", "best_model.pth")), (
    "Checkpoint already exists; use a new --output_dir. Do not silently resume."
)
os.makedirs(OUT, exist_ok=True)

# 3. GPU availability check: fail closed, never fall back to a multi-hour CPU run.
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Kaggle GPU is not enabled; refusing CPU training."
gpu_name = torch.cuda.get_device_name(0)
print("GPU:", gpu_name)

# 4. Exact selected configuration, with the verified split passed explicitly.
cmd = [
    sys.executable, "src/train.py",
    "--csv_path", CSV, "--images_dir", IMG, "--split_file", SPLIT,
    "--output_dir", OUT,
    "--architecture", "mobilenet_v2", "--image_size", "224", "--resize_size", "256",
    "--label_config_version", "v2_log_shrunk", "--shrinkage_k", "10.0",
    "--augmentation", "improved",
    "--epochs", "30", "--batch_size", "64",
    "--lr", "0.0001", "--lr_finetune", "0.00001", "--weight_decay", "0.0001",
    "--finetune_blocks", "4", "--patience", "5", "--scheduler", "cosine",
    "--loss", "ce", "--label_smoothing", "0.05",
    "--class_weight", "auto", "--class_weight_beta", "0.9999",
    "--focal_gamma", "2.0", "--ordinal_epsilon", "0.1",
    "--imbalance_ratio", "1.5", "--seed", "42", "--num_workers", "2",
    "--device", "auto",
]
# Bind this run to the exact preflight artifacts before training starts.
gate = {
    "pinned_sha": PIN, "head_sha": head,
    "canonical_split": SPLIT, "canonical_split_sha256": split_sha,
    "preflight_report": REPORT, "preflight_report_sha256": report_sha,
    "preflight_status": "PASS", "split_counts": EXPECTED,
    "selection_metric": "validation_macro_f1_only",
    "test_data_used_for": [], "command": cmd,
    "gpu": gpu_name, "torch_version": torch.__version__,
}
with open(os.path.join(OUT, "training_gate.json"), "w", encoding="utf-8") as handle:
    json.dump(gate, handle, indent=2)
subprocess.run(cmd, cwd=REPO, check=True)

# 5. Post-run checks. NO test inference or metric is computed in this cell.
for rel in ("models/best_model.pth", "models/model_config.json",
            "models/preprocessing_config.json", "models/preprocessing_stats.json",
            "models/class_thresholds.json", "models/city_medians.json",
            "models/split_indices.json", "models/training_summary.json",
            "results/training_history.csv", "results/learning_curves.png"):
    assert os.path.isfile(os.path.join(OUT, rel)), f"Missing artifact: {rel}"

checkpoint = torch.load(os.path.join(OUT, "models", "best_model.pth"),
                        map_location="cpu", weights_only=False)
assert checkpoint["arch"] == "mobilenet_v2"
assert checkpoint["label_config_version"] == "v2_log_shrunk"
assert checkpoint["label_stats"]["fitted_on"] == "train split only"
assert checkpoint["label_stats"]["n_train_records"] == 10829
assert checkpoint["train_config"]["split_source"] == SPLIT
assert checkpoint["train_config"]["seed"] == 42 and checkpoint["train_config"]["epochs"] == 30
assert json.load(open(os.path.join(OUT, "models", "split_indices.json"),
                      encoding="utf-8")) == split, "Saved split differs from verified split."
history = checkpoint["history"]
best = max(history, key=lambda row: row["val_macro_f1"])
assert checkpoint["best_epoch"] == best["epoch"]
assert abs(checkpoint["val_macro_f1"] - best["val_macro_f1"]) < 1e-6
assert all("test" not in key for row in history for key in row), "A test metric leaked into selection."
summary = json.load(open(os.path.join(OUT, "models", "training_summary.json"), encoding="utf-8"))
print("Best epoch:", checkpoint["best_epoch"], "| val macro F1:", checkpoint["val_macro_f1"])
print("Epochs run:", summary["total_epochs_run"], "| early stopped:", summary["early_stopped"])
print("Actual duration (s):", summary["training_duration_seconds"],
      "| peak VRAM (MB):", summary["peak_gpu_memory_mb"], "| device:", summary["resolved_device"])
print("Artifacts:", OUT)
```

Exact output directory: `/kaggle/working/output/experiments/e0_reference_group_split/`
(`models/best_model.pth`, `models/*.json`, `results/training_history.csv`,
`results/learning_curves.png`, plus `training_gate.json`). Runtime is **not**
predicted here: the cell prints the actual `training_duration_seconds` and
`peak_gpu_memory_mb` that `train.py` records. Internet must be ON in the Kaggle
notebook settings for the one-time ImageNet weight download; `src/model.py`
fails loudly instead of using random weights if that download fails.

Checkpoint selection: after every epoch, `train.py` computes validation macro F1
with all three labels fixed and overwrites `best_model.pth` only on a strict
improvement (`> best + 1e-6`). Phase 1 (15 epochs) trains the head with the
backbone frozen and never early-stops; phase 2 (15 epochs) unfreezes the last 4
backbone blocks and stops after 5 epochs without improvement. History holds only
train/val loss, accuracy and `val_macro_f1` — no test metric exists in it.

### One street group spans splits in the audit — investigated

The single `street_overlap` example is the literal `"Address not provided"`
sentinel (train 18, val 4, test 1). A check with `group_split.normalize_address`
over all 15,474 rows finds **0 usable normalized addresses crossing splits**;
the 23 placeholder ids exactly match `group_manifest.json`'s
`missing_address_ids`. One placeholder id is joined by exact SHA-256 to another
image, but both are in the same split, and exact-duplicate groups across splits
are 0. There is no other address overlap to ignore.

## One-time final test evaluation — separate cell, only after training succeeds

Do not run this during intermediate experiments. It refuses to run twice and
verifies the selection/split/no-test-statistics gates before touching
`evaluate.py`. Test metrics must not be used for any further selection.

```python
import os, sys, json, hashlib, subprocess, datetime
import torch

PIN = "bc39d102d2e9cde83144966d6e6c910926109429"
REPO = "/kaggle/working/real-estate-vision-agent"
CSV = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv"
IMG = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics"
SPLIT = "/kaggle/working/output/leakage_resistant_split/split_indices.json"
REPORT = "/kaggle/working/output/leakage_resistant_split/verification.json"
OUT = "/kaggle/working/output/experiments/e0_reference_group_split"
EXPECTED_COUNTS = {"train": 10829, "val": 2337, "test": 2308}
EXPECTED_BEST_EPOCH = 27
EXPECTED_BEST_VAL_F1 = 0.4314855288867976

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# 0. Pinned code only.
assert os.path.isdir(os.path.join(REPO, ".git")), "Repository checkout missing; rerun preflight."
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert head == PIN, f"HEAD {head} != pinned {PIN}; stop."
print("PIN verified:", head)

# 1. One-time guard: refuse to re-evaluate the test split.
METRICS = os.path.join(OUT, "results", "metrics.json")
MARK = os.path.join(OUT, "results", "evaluation_gate.json")
assert not os.path.exists(METRICS) and not os.path.exists(MARK), (
    "Test split was already evaluated; do not evaluate it a second time."
)

# 2. Training-run provenance: same pin, same artifacts, same split.
gate_path = os.path.join(OUT, "training_gate.json")
assert os.path.isfile(gate_path), "training_gate.json missing; training provenance unverifiable."
gate = json.load(open(gate_path, encoding="utf-8"))
assert gate["pinned_sha"] == PIN and gate["head_sha"] == PIN
assert gate["preflight_status"] == "PASS"
assert gate["selection_metric"] == "validation_macro_f1_only"
assert gate["test_data_used_for"] == []
assert gate["canonical_split"] == SPLIT and gate["preflight_report"] == REPORT
assert sha256(gate["canonical_split"]) == gate["canonical_split_sha256"]
assert sha256(gate["preflight_report"]) == gate["preflight_report_sha256"]
assert gate["split_counts"] == EXPECTED_COUNTS
report = json.load(open(REPORT, encoding="utf-8"))
assert report["status"] == "PASS" and report["pinned_sha"] == PIN
assert report["split"]["counts"] == EXPECTED_COUNTS
split = json.load(open(SPLIT, encoding="utf-8"))
assert {n: len(split[n]) for n in ("train", "val", "test")} == EXPECTED_COUNTS

# 3. Checkpoint genuinely is the best-validation-F1 epoch from this training run.
ckpt_path = os.path.join(OUT, "models", "best_model.pth")
assert os.path.isfile(ckpt_path), "Checkpoint missing."
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
assert checkpoint["arch"] == "mobilenet_v2"
assert checkpoint["label_config_version"] == "v2_log_shrunk"
assert checkpoint["label_stats"]["fitted_on"] == "train split only"
assert checkpoint["label_stats"]["n_train_records"] == 10829
assert checkpoint["train_config"]["split_source"] == SPLIT
assert checkpoint["train_config"]["seed"] == 42
assert checkpoint["best_epoch"] == EXPECTED_BEST_EPOCH
assert checkpoint["train_config"]["best_epoch"] == EXPECTED_BEST_EPOCH
assert abs(checkpoint["val_macro_f1"] - EXPECTED_BEST_VAL_F1) < 1e-6
best = max(checkpoint["history"], key=lambda row: row["val_macro_f1"])
assert best["epoch"] == EXPECTED_BEST_EPOCH
assert checkpoint["best_epoch"] == best["epoch"]
assert abs(checkpoint["val_macro_f1"] - best["val_macro_f1"]) < 1e-6
assert all("test" not in key for row in checkpoint["history"] for key in row)
assert json.load(open(os.path.join(OUT, "models", "split_indices.json"),
                      encoding="utf-8")) == split
summary = json.load(open(os.path.join(OUT, "models", "training_summary.json"), encoding="utf-8"))
assert summary["best_epoch"] == EXPECTED_BEST_EPOCH
assert abs(summary["best_val_macro_f1"] - EXPECTED_BEST_VAL_F1) < 1e-6
assert summary["total_epochs_run"] == 30 and summary["early_stopped"] is False
assert summary["resolved_device"].startswith("cuda")
print("Gate OK: epoch", EXPECTED_BEST_EPOCH, "val macro F1", checkpoint["val_macro_f1"],
      "| train-only stats; split identical to preflight artifact.")

# 4. Trigger the single test evaluation (evaluate.py writes all test reports).
subprocess.run([
    sys.executable, "src/evaluate.py",
    "--csv_path", CSV, "--images_dir", IMG, "--output_dir", OUT,
    "--checkpoint", ckpt_path, "--batch_size", "64", "--num_workers", "2",
    "--seed", "42", "--device", "auto",
], cwd=REPO, check=True)

# 5. Verify the supported metrics were written, then seal the one-time gate.
metrics = json.load(open(METRICS, encoding="utf-8"))
REQUIRED = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "per_class", "confusion_matrix", "normalized_confusion_matrix",
            "column_normalized_confusion_matrix", "baselines", "test_samples",
            "split_sizes", "score_type", "interpretation_warning")
assert all(k in metrics for k in REQUIRED), [k for k in REQUIRED if k not in metrics]
assert metrics["test_samples"] == EXPECTED_COUNTS["test"]
assert metrics["split_sizes"] == EXPECTED_COUNTS
assert metrics["best_epoch"] == EXPECTED_BEST_EPOCH
assert metrics["score_type"] == "market_derived_visual_proxy"
for rel in ("metrics.json", "classification_report.json", "evaluation_report.md",
            "confusion_matrix.png", "normalized_confusion_matrix.png",
            "column_normalized_confusion_matrix.png", "prediction_distribution.png",
            "learning_curves.png"):
    assert os.path.isfile(os.path.join(OUT, "results", rel)), f"Missing: {rel}"

evaluation = {
    "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "pinned_sha": PIN, "head_sha": head,
    "checkpoint": ckpt_path, "checkpoint_sha256": sha256(ckpt_path),
    "canonical_split": SPLIT, "split_sha256": sha256(SPLIT),
    "preflight_report": REPORT, "preflight_sha256": sha256(REPORT),
    "metrics": METRICS, "metrics_sha256": sha256(METRICS),
    "best_epoch": EXPECTED_BEST_EPOCH,
    "validation_macro_f1": checkpoint["val_macro_f1"],
    "test_accuracy": metrics["accuracy"],
    "test_macro_f1": metrics["macro_f1"],
    "test_macro_precision": metrics["macro_precision"],
    "test_macro_recall": metrics["macro_recall"],
    "test_weighted_f1": metrics["weighted_f1"],
    "test_per_class": metrics["per_class"],
    "test_confusion_matrix": metrics["confusion_matrix"],
    "test_normalized_confusion_matrix": metrics["normalized_confusion_matrix"],
    "test_baselines": metrics["baselines"],
    "test_samples": metrics["test_samples"], "split_sizes": metrics["split_sizes"],
    "all_metrics": metrics,
    "score_definition": "0.5*P(average) + 1.0*P(high); confidence=max class probability (not calibrated)",
    "interpretation_warning": metrics["interpretation_warning"],
    "note": "One-time final test evaluation; test metrics are never used for selection.",
}
with open(MARK, "w", encoding="utf-8") as handle:
    json.dump(evaluation, handle, indent=2)
print("ONE-TIME TEST EVALUATION COMPLETE ->", METRICS, "| gate:", MARK)
```

`src/evaluate.py` already records test accuracy, macro/weighted F1, macro
precision/recall, per-class precision/recall/F1/support, raw and normalized
confusion matrices, computed baselines and the proxy-label warning in
`results/metrics.json`, `results/classification_report.json` and
`results/evaluation_report.md`.

**Scientific framing (unchanged):** the CNN predicts market-derived visual
quality proxy classes (price-per-sqft tertiles relative to a shrunken city
median), not independently verified physical-condition grades. The continuous
score stays `0.5 * P(average) + 1.0 * P(high)`, and confidence is the maximum
class probability (`VisionAgent._format_result`), **not** a calibrated
confidence. Macro F1 uses all three fixed labels.
