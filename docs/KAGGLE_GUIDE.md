# Kaggle guide — Member A

All training/evaluation runs on **Kaggle GPU**. Paths below are the mounted
dataset defaults. Never train locally, never push to GitHub, never overwrite the
historical `output/models` or `output/improved_mobilenet` folders.

## A. Clone / update

```python
!git clone <YOUR_REPOSITORY_URL> /kaggle/working/vision_agent
%cd /kaggle/working/vision_agent
# on an existing clone:
!git -C /kaggle/working/vision_agent pull --ff-only
```

## B. Dataset prep and sanity check

```python
!python src/inspect_csv.py --csv_path /kaggle/input/datasets/nikhilkushwaha2529/real-estate-with-image/Real_estatewith_image/archive-2/socal2/socal2.csv
!python -m unittest discover -s tests -v
```

Defaults (already in `configs/experiments.json`): CSV `.../socal2/socal2.csv`,
images `.../socal2/socal_pics/`, batch 64, seed 42, 2 workers. The loader expects
`<images_dir>/<image_id>.jpg`. Check for other extensions/missing files with:

```python
!python src/data_audit.py
```

## C. Run experiments

```python
!python src/run_experiment.py --list
!python src/run_experiment.py --name e0_reference --dry-run
!python src/run_experiment.py --name e0_reference
!python src/run_experiment.py --name e1_no_weighting
!python src/run_experiment.py --name e2_balanced_ce
!python src/run_experiment.py --name e3_effective_number
!python src/run_experiment.py --name e4_focal
!python src/run_experiment.py --name e5_ordinal_smooth
!python src/run_experiment.py --name e6_efficientnet_b0
!python src/run_experiment.py --name e7_higher_resolution
!python src/run_experiment.py --name e8_legacy_augmentation
!python src/run_experiment.py --name e9_shrinkage_k1
!python src/run_experiment.py --name e10_longer_finetune
!python src/run_experiment.py --name e0_reference --probe   # diagnostic probe only
```

Each run gets its own `output/experiments/<name>/` with checkpoint, config,
history, duration/VRAM and `experiment_run.json`. The runner refuses to reuse an
existing checkpoint unless `--force` is passed.

## D. Leakage audit (before trusting held-out scores)

```python
!python src/data_audit.py --split_file /kaggle/working/output/experiments/e0_reference/models/split_indices.json
```

## E. Final test evaluation — ONCE, winner only

```python
!python src/evaluate.py --output_dir /kaggle/working/output/experiments/e0_reference
# or, equivalently, after locking the decision:
!python src/run_experiment.py --name e0_reference --eval --force
```

Do not do this for every experiment. Do not tune anything after seeing test
metrics. If no new run beats the incumbent's 0.4521 validation macro F1, keep
`/kaggle/working/output/improved_mobilenet/models/best_model.pth`.

## F. VisionAgent smoke test

```python
from src.vision_agent import VisionAgent
agent = VisionAgent(checkpoint_path="/kaggle/working/output/improved_mobilenet/models/best_model.pth", device="cuda")
print(agent.score("/path/to/house.jpg"))
print(len(agent.score_batch(["/path/a.jpg", "/path/b.jpg"], batch_size=64)))
```

Full tests (including synthetic checkpoint loading, batch/single agreement,
invalid paths and output validity):

```python
!python -m unittest discover -s tests -v
```

## G. OOF scores for Member B

```python
!python src/make_folds.py --n_folds 5 --seed 42 --output_dir /kaggle/working/output/oof
# train each fold with --split_file .../folds/fold_k.json --output_dir .../fold_k
!python src/oof_predict.py --output_dir /kaggle/working/output/oof
```

See `docs/MEMBER_B_INTEGRATION.md` for the full contract.

## H. Checkpoint export

```python
!cp /kaggle/working/output/experiments/<winner>/models/best_model.pth /kaggle/working/final_visual_model.pth
!python - <<'PY'
from src.model import load_checkpoint
c = load_checkpoint("/kaggle/working/final_visual_model.pth", map_location="cpu")
print(c["arch"], c["label_config_version"], c["best_epoch"], c["val_macro_f1"])
PY
```

Export only the selected checkpoint. Include `docs/EXPERIMENT_RESULTS.md` and
`docs/FINAL_MEMBER_A_REPORT.md` with the submission; leave historical folders
untouched.
