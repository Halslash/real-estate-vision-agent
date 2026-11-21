# Member A — Kaggle verification cells and first group-split training run

**No training is performed by these cells. The test split is not evaluated.**
Nothing here assumes a local split file: the split is generated from the mounted
Kaggle dataset inside the notebook.

> Set `PIN` below to the SHA of the commit containing `src/group_split.py`,
> `tests/test_group_split.py` and the `src/train.py` validation guard. Cell 1
> asserts that `src/group_split.py` is present in the checked-out commit and
> fails loudly otherwise; do not bypass that assertion.

---

## Cell 1 — clone / checkout the intended commit

```python
import os, subprocess

REPO = "/kaggle/working/real-estate-vision-agent"
URL  = "https://github.com/Halslash/real-estate-vision-agent.git"
PIN  = "<PASTE-THE-PUSHED-COMMIT-SHA>"   # must contain src/group_split.py

if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", URL, REPO], check=True)
subprocess.run(["git", "fetch", "--all", "--tags"], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", PIN], cwd=REPO, check=True)
print("HEAD:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
assert os.path.isfile(os.path.join(REPO, "src", "group_split.py")), (
    "src/group_split.py absent from this commit: commit and push it first."
)
os.chdir(REPO)
print("cwd:", os.getcwd())
```

## Cell 2 — load the actual Kaggle CSV and images

```python
import os
from src.dataset import DEFAULT_CSV_PATH, DEFAULT_IMAGES_DIR, load_and_validate_records

assert os.path.isfile(DEFAULT_CSV_PATH), DEFAULT_CSV_PATH
assert os.path.isdir(DEFAULT_IMAGES_DIR), DEFAULT_IMAGES_DIR
records, skipped, detected = load_and_validate_records(DEFAULT_CSV_PATH, DEFAULT_IMAGES_DIR)
print("records:", len(records), "| skipped:", skipped, "| columns:", detected)
assert skipped == 0, "Refusing to proceed: CSV rows were skipped."
```

## Cell 3 — generate the leakage-resistant split with the project's code

```python
import os
OUT = "/kaggle/working/output/leakage_resistant_split"
assert not os.path.exists(os.path.join(OUT, "split_indices.json")), (
    "Split already exists; delete this audit dir to regenerate."
)
!python src/group_split.py --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ --output_dir /kaggle/working/output/leakage_resistant_split --seed 42
```

## Cell 4 — verify sizes, group disjointness, class counts, coverage

```python
import json
from collections import defaultdict
import numpy as np
from src.dataset import fit_preprocessing_stats, compute_labels, split_records
from src.train import validate_supplied_split

SPLIT = OUT + "/split_indices.json"
MANIFEST = OUT + "/group_manifest.json"
split = json.load(open(SPLIT))
manifest = json.load(open(MANIFEST))

# train.py's own guard: no unknown / overlapping / duplicate / dropped ids.
validate_supplied_split(split, records)
assert set(i for v in split.values() for i in v) == {r["image_id"] for r in records}

owner = {i: s for s in ("train", "val", "test") for i in split[s]}
gmembers = defaultdict(list)
for iid, gid in manifest["group_of_image_id"].items():
    gmembers[gid].append(iid)
assert all(len({owner[i] for i in m}) == 1 for m in gmembers.values())
print("groups:", len(gmembers), "| no group crosses a split")

for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
    frac = len(split[name]) / len(records)
    assert abs(frac - target) < 0.02, (name, frac)
    print(name, len(split[name]), round(frac, 4))

# Class counts only (no model, no test evaluation). Thresholds fitted on TRAIN only.
train_recs = [r for r in records if r["image_id"] in set(split["train"])]
stats = fit_preprocessing_stats(train_recs, label_config_version="v2_log_shrunk", shrinkage_k=10.0)
labeled = split_records(compute_labels(records, stats, inplace=False), split)
for name in ("train", "val", "test"):
    counts = np.bincount([r["label"] for r in labeled[name]], minlength=3)
    assert (counts > 0).all()
    print(name, counts.tolist(), (counts / counts.sum()).round(3).tolist())
```

## Cell 5 — full leakage audit (all images, no --max_images)

```python
AUDIT = OUT + "/data_audit.json"
!python src/data_audit.py --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2.csv --images_dir /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal_pics/ --split_file /kaggle/working/output/leakage_resistant_split/split_indices.json --output /kaggle/working/output/leakage_resistant_split/data_audit.json
a = json.load(open(AUDIT))
print("rows:", a["rows_scanned"], "| missing images:", a["missing_image_files"],
      "| exact cross-split groups:", a["exact_duplicate_groups_across_splits"],
      "| near-dup cross-split pairs:", a["near_duplicate_pairs_across_splits"])
assert a["rows_scanned"] == len(records)
assert a["missing_image_files"] == 0
assert a["exact_duplicate_groups_across_splits"] == 0
assert a["near_duplicate_pairs_across_splits"] == 0
# The one spanning "street" group is the literal "Address not provided" sentinel.
```

## Cell 6 — persist verification outside any experiment directory

```python
summary = {
    "split": SPLIT,
    "manifest": MANIFEST,
    "audit": AUDIT,
    "n_records": len(records),
    "split_counts": {n: len(split[n]) for n in ("train", "val", "test")},
    "n_groups": len(gmembers),
    "class_counts": {
        n: np.bincount([r["label"] for r in labeled[n]], minlength=3).tolist()
        for n in ("train", "val", "test")
    },
    "exact_duplicate_groups_across_splits": a["exact_duplicate_groups_across_splits"],
    "near_duplicate_pairs_across_splits": a["near_duplicate_pairs_across_splits"],
    "missing_image_files": a["missing_image_files"],
    "note": "Generated and verified on Kaggle; no model trained; test not evaluated.",
}
with open(OUT + "/verification.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print("Wrote", OUT + "/verification.json")
```

All artifacts stay in `/kaggle/working/output/leakage_resistant_split/`, never in
`/kaggle/working/output/experiments/`.

---

## First training command — fresh v2 reference (do not run until cells 1-6 pass)

`configs/experiments.json` `e0_reference` inherits every value below from
`defaults`; the flags are written out explicitly so the protocol is auditable.
`--split_file` is passed directly to `train.py` instead of editing the shared
config, so no existing experiment definition changes. No `--eval` / `--probe`.

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

Do not evaluate the test split after this run. Selection stays on validation
macro F1 only.
