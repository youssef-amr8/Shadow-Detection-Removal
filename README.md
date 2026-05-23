# Shadow detection & removal

Dataset (2GB): [ISTD on Kaggle](https://www.kaggle.com/datasets/sabarinathan/istd-dataset/data) → `data/ISTD_Dataset`

```
train/train_A/  images with shadow
train/train_B/  shadow masks
train/train_C/  shadow-free ground truth
```

## Pipeline (default: `--removal-backend unet`)

```
Original image
    → Detection U-Net  (shadow_detection.pth)  →  shadow mask
    → Removal U-Net    (shadow_removal.pth)      →  uses [image + mask]
    → Shadow-free output
```

Training uses **ground-truth** masks (`train_B`). At run time the mask comes from the **detector**.

## 1) Train detection

```bash
python main.py train-detection --data data/ISTD_Dataset --epochs 50 --batch-size 4
```

→ `models/shadow_detection.pth`

## 2) Train removal (needs A + B + C)

```bash
python main.py train-removal --data data/ISTD_Dataset --epochs 50 --batch-size 8
```

→ `models/shadow_removal.pth` (4-channel input: RGB + mask)

**Colab:** [COLAB_TRAINING.md](COLAB_TRAINING.md) · `colab/train_shadow_removal.ipynb`

> If you trained removal before this update, **retrain on Colab** (architecture changed to 4-channel input).

## 3) Run

```bash
python main.py run --image photo.jpg --output outputs/out.png \
  --removal-backend unet \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth \
  --save-mask outputs/mask.png -v
```

**Legacy CV path** (no removal U-Net):

```bash
python main.py run --image photo.jpg --output outputs/out.png --removal-backend cv -v
```
