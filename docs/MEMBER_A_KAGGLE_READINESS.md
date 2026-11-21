# Member A — one-cell Kaggle split workflow (and later training command)

**The cell below performs no training and does not evaluate the test split.** It
loads the mounted Kaggle data, generates the leakage-resistant split, runs the
essential read-only audit, verifies integrity and saves the split files. Set
`PIN` to the SHA of the commit containing `src/group_split.py`,
`tests/test_group_split.py` and the `src/train.py` validation guard; the cell
asserts that file is present and fails loudly otherwise. Do not bypass it.

---

## The one cell

```python
import os, sys, json, subprocess
from collections import defaultdict
import numpy as np

# 0. Get the code (pinned). Runs once; safe to re-run after a reset.
REPO = "/kaggle/working/real-estate-vision-agent"
URL  = "https://github.com/Halslash/real-estate-vision-agent.git"
PIN  = "<PASTE-THE-PUSHED-COMMIT-SHA>"   # commit containing src/group_split.py
if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", URL, REPO], check=True)
subprocess.run(["git", "fetch", "--all", "--tags"], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", PIN], cwd=REPO, check=True)
assert os.path.isfile(os.path.join(REPO, "src", "group_split.py")), "Push the commit first."
os.chdir(REPO); sys.path.insert(0, REPO)

from src.dataset import load_and_validate_records, fit_preprocessing_stats, compute_labels, split_records
from src.train import validate_supplied_split

CSV = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv"
IMG = "/kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/"
OUT = "/kaggle/working/output/leakage_resistant_split"
assert not os.path.exists(os.path.join(OUT, "split_indices.json")), "Delete OUT to regenerate."

# 1. Load data.
records, skipped, detected = load_and_validate_records(CSV, IMG)
assert skipped == 0, "CSV rows were skipped; stop."
print("records:", len(records), "| columns:", detected)

# 2. Generate split: normalized address + exact SHA-256, seed 42; global dHash OFF.
subprocess.run([sys.executable, "src/group_split.py", "--csv_path", CSV, "--images_dir", IMG,
                "--output_dir", OUT, "--seed", "42"], cwd=REPO, check=True)
split = json.load(open(OUT + "/split_indices.json"))
manifest = json.load(open(OUT + "/group_manifest.json"))

# 3. Essential read-only audit (all images).
subprocess.run([sys.executable, "src/data_audit.py", "--csv_path", CSV, "--images_dir", IMG,
                "--split_file", OUT + "/split_indices.json", "--output", OUT + "/data_audit.json"],
               cwd=REPO, check=True)
a = json.load(open(OUT + "/data_audit.json"))
assert a["rows_scanned"] == len(records)
assert a["missing_image_files"] == 0
assert a["exact_duplicate_groups_across_splits"] == 0
print("same-street near-dup pairs across splits (diagnostic only):",
      a["near_duplicate_pairs_across_splits"])

# 4. Verify integrity. No group / address / exact image crosses a split.
validate_supplied_split(split, records)
assert set(i for v in split.values() for i in v) == {r["image_id"] for r in records}
owner = {i: s for s in ("train", "val", "test") for i in split[s]}
gmembers = defaultdict(list)
for iid, gid in manifest["group_of_image_id"].items():
    gmembers[gid].append(iid)
assert all(len({owner[i] for i in m}) == 1 for m in gmembers.values())
for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
    assert abs(len(split[name]) / len(records) - target) < 0.02, name

# Class counts only; thresholds fitted on TRAIN only. No model, no test evaluation.
train_recs = [r for r in records if r["image_id"] in set(split["train"])]
stats = fit_preprocessing_stats(train_recs, label_config_version="v2_log_shrunk", shrinkage_k=10.0)
labeled = split_records(compute_labels(records, stats, inplace=False), split)
counts = {n: np.bincount([r["label"] for r in labeled[n]], minlength=3).tolist() for n in split}
assert all(all(c > 0 for c in v) for v in counts.values())
print("split counts:", {n: len(split[n]) for n in split}, "| groups:", len(gmembers))
print("class counts:", counts)
print("missing/placeholder addresses:", len(manifest.get("missing_address_ids", [])))

# 5. Save verification summary alongside the split files.
summary = {
    "split": OUT + "/split_indices.json",
    "manifest": OUT + "/group_manifest.json",
    "audit": OUT + "/data_audit.json",
    "n_records": len(records),
    "split_counts": {n: len(split[n]) for n in split},
    "n_groups": len(gmembers),
    "class_counts": counts,
    "exact_duplicate_groups_across_splits": a["exact_duplicate_groups_across_splits"],
    "same_street_near_dup_pairs_across_splits": a["near_duplicate_pairs_across_splits"],
    "missing_image_files": a["missing_image_files"],
    "note": "Generated and verified on Kaggle; global dHash OFF; no model trained; test not evaluated.",
}
json.dump(summary, open(OUT + "/verification.json", "w"), indent=2)
print("Saved", OUT + "/verification.json")
```

All artifacts stay in `/kaggle/working/output/leakage_resistant_split/`, never in
`/kaggle/working/output/experiments/`. The split guarantees and its limits are
documented in `docs/MEMBER_A_LEAKAGE_RESISTANT_SPLIT.md`; notably, visually
similar photos at different addresses are **not** exhaustively excluded because
global dHash is off by default.

---

## Later training command — do not run until the cell above passes

`configs/experiments.json` `e0_reference` inherits every value below from
`defaults`; the flags are written out explicitly so the protocol is auditable.
`--split_file` is passed directly to `train.py` instead of editing the shared
config, so no existing experiment definition changes. No `--eval` / `--probe`.
Keep selection on **validation** macro F1 only; do not evaluate the test split.

```python
!python src/train.py \
  --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv \
  --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ \
  --split_file /kaggle/working/output/leakage_resistant_split/split_indices.json \
  --output_dir /kaggle/working/output/experiments/e0_reference_group_split \
  --architecture mobilenet_v2 --image_size 224 --resize_size 256 \
  --label_config_version v2_log_shrunk --shrinkage_k 10.0 --augmentation improved \
  --epochs 30 --batch_size 64 \
  --lr 0.0001 --lr_finetune 0.00001 --weight_decay 0.0001 \
  --finetune_blocks 4 --patience 5 --scheduler cosine \
  --loss ce --label_smoothing 0.05 \
  --class_weight auto --class_weight_beta 0.9999 --focal_gamma 2.0 --ordinal_epsilon 0.1 \
  --imbalance_ratio 1.5 --seed 42 --num_workers 2 --device auto
```
