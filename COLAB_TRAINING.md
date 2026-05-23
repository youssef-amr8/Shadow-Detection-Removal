# Train shadow removal U-Net on Google Colab

**Guided removal** U-Net: `(train_A image + train_B mask) → train_C` clean image.  
At inference: **detector mask** replaces `train_B`.

## Dataset layout (ISTD)

```
ISTD_Dataset/
  train/
    train_A/   ← shadow image (input)
    train_B/   ← shadow mask (training; at run time = detector output)
    train_C/   ← shadow-free target
```

Download: [ISTD on Kaggle](https://www.kaggle.com/datasets/sabarinathan/istd-dataset/data)

---

## Colab steps

### 1) Open Colab

[Google Colab](https://colab.research.google.com/) → **New notebook** → **Runtime → Change runtime type → GPU** (T4 is fine).

### 2) Get the project code

**Option A — GitHub (if repo is pushed):**

```python
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
```

**Option B — Upload zip:** zip your project folder, upload in Colab, then:

```python
!unzip -q your_project.zip -d /content
%cd /content/DL   # your folder name
```

### 3) Install dependencies

```python
!pip install -q torch torchvision tqdm numpy opencv-python Pillow
```

### 4) Upload the dataset

**Option A — Google Drive** (recommended for ~2GB):

```python
from google.colab import drive
drive.mount("/content/drive")

# Adjust path to where you put ISTD_Dataset on Drive
DATA = "/content/drive/MyDrive/ISTD_Dataset"
```

**Option B — Kaggle API:**

```python
!pip install -q kaggle
# Upload kaggle.json first (Kaggle account → Create API token)
!mkdir -p ~/.kaggle
# upload kaggle.json to /root/.kaggle/kaggle.json then:
!kaggle datasets download -d sabarinathan/istd-dataset -p /content/data --unzip
DATA = "/content/data/ISTD_Dataset"  # fix path if unzip layout differs
```

**Option C — Direct upload** to `/content/ISTD_Dataset` (slow for 2GB).

Verify folders:

```python
import os
for sub in ["train/train_A", "train/train_B", "train/train_C"]:
    p = os.path.join(DATA, sub)
    print(p, "exists:", os.path.isdir(p), "files:", len(os.listdir(p)) if os.path.isdir(p) else 0)
```

### 5) Train

```python
!python main.py train-removal --data "{DATA}" --epochs 50 --batch-size 8 --device cuda --num-workers 0
```

Faster trial: `--epochs 20 --batch-size 16`

### 6) Check checkpoint

```python
import os
p = "models/shadow_removal.pth"
print("exists:", os.path.isfile(p), "size MB:", os.path.getsize(p)/1e6 if os.path.isfile(p) else 0)
```

### 7) Download `.pth` to your PC

```python
from google.colab import files
files.download("models/shadow_removal.pth")
```

Or copy to Drive:

```python
!cp models/shadow_removal.pth "/content/drive/MyDrive/shadow_removal.pth"
```

### 8) Use on your machine

Put the file here (either path works):

- `models/shadow_removal.pth`
- project root `shadow_removal.pth`

Run inference (detector → mask → removal U-Net):

```bash
python main.py run \
  --image your_photo.jpg \
  --output outputs/removed_unet.png \
  --removal-backend unet \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth \
  --save-mask outputs/mask.png \
  -v
```

---

## Train locally (optional)

```bash
python main.py train-removal --data data/ISTD_Dataset --epochs 50 --batch-size 8
```

---

## Tune training (code)

| File | What |
|------|------|
| `src/shadow_removal_train.py` | epochs, batch, lr, loss |
| `src/preprocessing.py` → `ShadowRemovalDataset` | `img_size` |
| `src/model.py` → `UNet(out_channels=3)` | architecture |

---

## Notes

- Removal model needs **both** checkpoints at run time: `shadow_detection.pth` + `shadow_removal.pth`.
- Retrain removal if you have an old 3-channel-only `shadow_removal.pth` from a previous version.
- CV pipeline still works: `python main.py run --removal-backend cv ...`
- Inference resizes to 256×256 inside the net, then resizes back to original resolution (same as detection).
