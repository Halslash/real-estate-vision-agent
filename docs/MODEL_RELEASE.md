# Model distribution — Member A final checkpoint

**Primary distribution: the checkpoint is included directly in this Git
repository.** Cloning is sufficient; no GitHub Release, Kaggle access, or extra
download is required. The same checkpoint is **also published** as a GitHub
Release asset — an **additional optional** distribution route, not the only way
to obtain the model. The two copies were verified byte-identical.

## Tracked checkpoint

| Field | Value |
|---|---|
| Repository path | `checkpoints/member_a_cnn_e0_reference_group_split_best.pth` |
| Size | **9,171,467 bytes** (verified locally with `stat`) |
| SHA-256 | `9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b` (verified locally with `sha256sum`) |
| Training source | `e0_reference_group_split`, source commit `bc39d102d2e9cde83144966d6e6c910926109429`, originally at `/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth` |

The global `.gitignore` rule `*.pth` is unchanged: only this one selected
checkpoint was explicitly added and is tracked by Git. Do not weaken the ignore
rule or commit other weights.

Verify the file after cloning and confirm Git tracks it:

```bash
sha256sum checkpoints/member_a_cnn_e0_reference_group_split_best.pth
# expected: 9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b
git ls-files --error-unmatch checkpoints/member_a_cnn_e0_reference_group_split_best.pth
```

Verified checkpoint identity (local readback): `arch=mobilenet_v2`,
`label_config_version=v2_log_shrunk`, `epochs=30`, `seed=42`, `best_epoch=27`,
`val_macro_f1=0.4314855288867976`; `VisionAgent` loads it and scores images.

## Optional GitHub Release (published and verified)

| Field | Value |
|---|---|
| Tag | `v1.0.0-member-a-cnn` |
| Asset filename | `member_a_cnn_e0_reference_group_split_best.pth` |
| Download URL (verified, HTTP 200) | `https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth` |
| Asset size | **9,171,467 bytes** (matches the tracked checkpoint) |
| SHA-256 of published asset | `9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b` (downloaded and compared — identical to the repository copy) |

### Release notes

The verbatim notes are in `docs/MODEL_RELEASE_NOTES.md`. Summary: MobileNetV2,
CE + auto effective-number weights β=0.9999, label smoothing 0.05, 30 epochs,
seed 42, property-disjoint split (10,829/2,337/2,308), best epoch 27,
validation Macro F1 `0.4314855`, one-time test Macro F1 `0.4240`. The model is a
market-derived visual proxy, not verified condition and not a price; confidence
is not calibrated; do not use this checkpoint to score Member B outer-test rows
that overlap its training data.

### Verify the published asset

The release and its asset already exist; download and compare against the
repository copy (the asset is *not* needed to use the model, since cloning
already provides the weights):

```bash
curl -fL --retry 3 -o /tmp/opencode/member_a_checkpoint.pth \
  https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth
sha256sum /tmp/opencode/member_a_checkpoint.pth
sha256sum checkpoints/member_a_cnn_e0_reference_group_split_best.pth
# both: 9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b
```

## Inference from the repository copy

```bash
python - <<'PY'
from src.vision_agent import VisionAgent
agent = VisionAgent(checkpoint_path="checkpoints/member_a_cnn_e0_reference_group_split_best.pth")
print(agent.score("example_house.jpg"))
PY
```
