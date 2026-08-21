# Uncertainty-Aware Domain Generalization for Thyroid Nodule Segmentation

Final-year project — Sadia Afrin Kumu (2018056), ICT, Islamic University Bangladesh.

Train on **TN3K** only. Test zero-shot on **DDTI** (different hospital). Measure the accuracy
drop, measure whether uncertainty estimates survive the domain shift, then use the *epistemic*
part of the uncertainty to steer a style-consistency loss.

---

## 1. Hardware

| Machine | Role |
|---|---|
| Laptop (no GPU) | write code, read results |
| Remote PC (RTX 2060, 6 GB VRAM, 16 GB RAM) | run everything |

All training settings are tuned for 6 GB: 256x256 inputs, batch size 8, mixed precision.

## 2. Install (on the remote PC)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_env.py
```

`check_env.py` must print `CUDA: True` and a peak-memory number under ~5000 MB.

## 3. Get the datasets

Download by hand into `data/raw/`. Nothing is downloaded automatically.

| Dataset | Where | Put it in |
|---|---|---|
| TN3K | github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation (Google Drive link in README) | `data/raw/tn3k/` |
| DDTI | opencas.webarchiv.kit.edu / Kaggle "DDTI thyroid ultrasound images" | `data/raw/ddti/` |
| TNSCUI2020 *(optional, Step 7)* | MICCAI 2020 TN-SCUI challenge | `data/raw/tnscui/` |
| TG3K *(optional, gland masks only)* | same TRFE repo | `data/raw/tg3k/` |

Expected raw layouts (the script also searches subfolders, so slight differences are fine):

```
data/raw/tn3k/trainval-image/*.jpg   data/raw/tn3k/trainval-mask/*.jpg
data/raw/tn3k/test-image/*.jpg       data/raw/tn3k/test-mask/*.jpg

data/raw/ddti/image/*.PNG            data/raw/ddti/mask/*.PNG
   (or raw DDTI:  data/raw/ddti/**/*.jpg  +  data/raw/ddti/**/*.xml)

data/raw/tnscui/train/image/*.PNG    data/raw/tnscui/train/mask/*.PNG
data/raw/tg3k/thyroid-image/*.jpg    data/raw/tg3k/thyroid-mask/*.jpg
```

## 4. Build the unified dataset

```bash
python -m src.data.prepare --datasets tn3k ddti
python scripts/verify_data.py
```

This writes `data/processed/<domain>/images|masks/*.png` (256x256, grayscale image,
strictly binary mask) plus `data/processed/manifest.csv`, and puts the domain-gap
figures in `outputs/figures/data/`.

## 5. Project layout

```
configs/          one YAML per ablation configuration
src/data/         dataset building, loading, Fourier restyling
src/models/       U-Net + ResNet-34 with decoder dropout
src/              losses, uncertainty, metrics, train, evaluate
scripts/          environment check, data verification
app/              upload-an-image demo
data/raw/         downloaded datasets (never edited by hand)
data/processed/   unified 256x256 build
outputs/          checkpoints, metrics, figures
```

## 6. Protocol rule

No DDTI (or TNSCUI) image or label is ever used for training, hyperparameter choice,
early stopping, or checkpoint selection. Model selection uses only a validation split
held out from TN3K `trainval`.
