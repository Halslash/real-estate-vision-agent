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
Checkpoint: 9,171,467 bytes; SHA-256
9ffd1952fca3f18627d5070a3f3d7eeb8781f101a245d9675efa285d98d7cd4b.
The weights are also tracked in the repository at
checkpoints/member_a_cnn_e0_reference_group_split_best.pth.

Interpretation: market-derived visual proxy (shrunken log price-per-sqft tertile),
NOT verified physical condition and NOT a price estimate. Confidence is max class
probability, not calibrated. Do not use this checkpoint to score Member B outer-test
rows that overlap its training data; use fold-specific CNNs instead.
