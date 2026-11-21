# Model release — Member A final checkpoint (PREPARATION ONLY)

**Status: NOT PUBLISHED.** No GitHub Release exists yet. The final checkpoint is
currently available only at the Kaggle path
`/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth`;
it is absent from this working copy, so its size and SHA-256 cannot be verified
here. Nothing in this file claims that a release asset, URL, or checksum has been
created or validated.

## Planned release

| Field | Value |
|---|---|
| Tag | `v1.0.0-member-a-cnn` |
| Asset filename | `member_a_cnn_e0_reference_group_split_best.pth` |
| Intended download URL (will 404 until published) | `https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth` |
| Source commit | `bc39d102d2e9cde83144966d6e6c910926109429` |
| Size | **unknown locally** — run `stat -c %s` on the Kaggle file |
| SHA-256 | **unknown locally** — run `sha256sum` on the Kaggle file |

## Release notes (copy verbatim when publishing)

```text
Member A final visual-proxy CNN — e0_reference_group_split

Architecture: MobileNetV2 (ImageNet pretrained), 224px centre crop from 256px,
3-class softmax head; condition_score = 0.5*P(average) + 1.0*P(high).
Experiment: e0_reference_group_split, source commit bc39d102d2e9cde83144966d6e6c910926109429.
Training: 30 epochs total, batch size 64, AdamW 1e-4 (head) / 1e-5 (fine-tune),
cosine schedule, CE + auto effective-number class weights beta=0.9999,
label smoothing 0.05, improved augmentation, seed 42, property-disjoint split
(10829/2337/2308 train/val/test). Best epoch 27.
Validation Macro F1: 0.4314855. One-time test Macro F1: 0.4240 (2,308 samples).
Test metrics were evaluated once and are sealed; do not reuse the test split.

Interpretation: market-derived visual proxy (shrunken log price-per-sqft tertile),
NOT verified physical condition and NOT a price estimate. Confidence is max class
probability, not calibrated. Do not use this checkpoint to score Member B outer-test
rows that overlap its training data; use fold-specific CNNs instead.
```

## Publish steps (require the repository owner's GitHub credentials)

1. On Kaggle, verify the selected checkpoint, then copy it locally:

```bash
python - <<'PY'
import torch
p = "/kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth"
c = torch.load(p, map_location="cpu", weights_only=False)
assert c["arch"] == "mobilenet_v2"
assert c["label_config_version"] == "v2_log_shrunk"
assert c["train_config"]["epochs"] == 30
assert c["best_epoch"] == 27
print("checkpoint identity OK")
PY
cp /kaggle/working/output/experiments/e0_reference_group_split/models/best_model.pth \
   member_a_cnn_e0_reference_group_split_best.pth
stat -c %s member_a_cnn_e0_reference_group_split_best.pth
sha256sum member_a_cnn_e0_reference_group_split_best.pth
```

2. From the repository root, create the release (never commit the `.pth` file):

```bash
gh auth login
gh release create v1.0.0-member-a-cnn \
  member_a_cnn_e0_reference_group_split_best.pth \
  --title "Member A final CNN — e0_reference_group_split" \
  --notes-file docs/MODEL_RELEASE_NOTES.md
```

3. Replace the `SHA placeholder` and `size placeholder` entries in `README.md`
   and `docs/MODEL_RELEASE.md` with the measured values **after** step 1.
4. Verify the published asset:

```bash
curl -fL -o /tmp/opencode/member_a_checkpoint.pth \
  https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth
sha256sum /tmp/opencode/member_a_checkpoint.pth
```

The binary stays excluded by `.gitignore` (`*.pth`); do not weaken that rule.

## Download and inference after publication

```bash
curl -fL -o member_a_cnn_e0_reference_group_split_best.pth \
  https://github.com/Halslash/real-estate-vision-agent/releases/download/v1.0.0-member-a-cnn/member_a_cnn_e0_reference_group_split_best.pth
sha256sum member_a_cnn_e0_reference_group_split_best.pth   # compare with published SHA-256
python - <<'PY'
from src.vision_agent import VisionAgent
agent = VisionAgent(checkpoint_path="member_a_cnn_e0_reference_group_split_best.pth")
print(agent.score("example_house.jpg"))
PY
```
