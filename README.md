# ConvNeXt DeLR – 26-Landmark Cephalometric Regression

ConvNeXtV2-based Dual-encoder Landmark Regression (D-CeLR) for the Aariz cephalometric dataset. The repo includes training, inference, and metric utilities, tuned for 26 landmarks (junior/senior annotations averaged).

## Contents
- `delr/model.py` – D-CeLR model with ConvNeXtV2/ResNet backbones.
- `delr/datasets.py` – Aariz dataloader with optional strong augmentations and 19/26/all landmark selection.
- `delr/metrics.py` – MRE (mm/px) and SDR utilities.
- `train.py` – training loop (OneCycleLR, mixed losses).
- `infer.py` – inference + metrics, JSON export.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Dataset layout (Aariz):
```
<dataset_root>/
  train/valid/test/
    Cephalograms/*.png|jpg|bmp
    Annotations/Cephalometric Landmarks/Junior Orthodontists/*.json
    Annotations/Cephalometric Landmarks/Senior Orthodontists/*.json
  cephalogram_machine_mappings.csv
```

## Training (defaults: 26 landmarks, augmentations on)
Example run with ConvNeXtV2 Tiny:
```bash
python train.py \
  --dataset-root /teamspace/studios/this_studio/Aariz/Aariz \
  --backbone convnextv2_tiny \
  --landmarks 26 \
  --epochs 200 \
  --batch-size 16 \
  --image-size 1024 \
  --output-dir outputs/convnextv2_tiny_26_e200_b16
```
Notable flags:
- `--no-augment` to disable geometric/color/noise aug.
- `--normalize` to apply mean=0.5, std=0.5 after `ToTensor`.
- `--image-size` to trade memory vs. accuracy (default 1024).
- `--backbone` supports `convnextv2_base`, `convnextv2_tiny`, or `resnet34`.

Checkpoints are written to `best_model.pt` (lowest val MRE) and `last_model.pt` in the chosen `--output-dir`.

## Inference
```bash
python infer.py \
  --dataset-root /teamspace/studios/this_studio/Aariz/Aariz \
  --split test \
  --checkpoint outputs/convnextv2_tiny_26_e200_b4_img1024/best_model.pt \
  --backbone convnextv2_tiny \
  --batch-size 2 \
  --image-size 1024 \
  --output outputs/convnextv2_tiny_26_e200_b4_img1024/test_predictions.json
```
Use `--skip-metrics` to skip MRE/SDR computation.

## Results (Aariz, 26 landmarks, ConvNeXtV2 Tiny)
Test metrics for `outputs/convnextv2_tiny_26_e200_b4_img1024/best_model.pt` (image size 1024, batch 4):
- MRE: **1.105 mm** (9.719 px)
- SDR: **86.7% / 91.0% / 93.9% / 96.7%** at 2 / 2.5 / 3 / 4 mm

Worst-5 landmarks by MRE (mm): Ramus 2.094; Condylion 2.000; Articulare 1.577; Porion 1.552; Gonion 1.473.

## Paths of interest
- Base run: `outputs/convnextv2_tiny_26_e200_b4_img1024/`
- Test predictions: `outputs/convnextv2_tiny_26_e200_b4_img1024/test_predictions.json`
- Overlays (pred-only): `outputs/convnextv2_tiny_26_e200_b4_img1024/overlays/`

## Requirements
See `requirements.txt` (PyTorch ≥ 2.1 with torchvision, timm). GPU recommended for 1024px training.
