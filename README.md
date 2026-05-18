Dataset not included due to size (2GB). Download from Kaggle and place under `data/ISTD_Dataset`:

https://www.kaggle.com/datasets/sabarinathan/istd-dataset/data

## Folder layout (training detection)

Use this root in all commands: **`data/ISTD_Dataset`**

- `data/ISTD_Dataset/train/train_A/` — RGB images **with** shadow  
- `data/ISTD_Dataset/train/train_B/` — shadow **masks** (same filenames as `train_A`)

You only need **A + B** for training the detector. Removal is **classical CV** (no second neural net, no `train_C` required for inference).

## What to run (exact order)

From the project root, with your virtual environment activated.

### 1) Install dependencies

**Linux / WSL (recommended):** use the script (installs **CPU PyTorch** by default — smaller, fewer WSL failures):

```bash
cd "/mnt/d/ASU needs/DL"   # your path; match case (DL vs dl)
chmod +x install-deps-linux.sh
./install-deps-linux.sh
source .venv/bin/activate
```

- **Skip `sudo` apt** (if packages already installed): `SKIP_APT=1 ./install-deps-linux.sh`
- **GPU PyTorch** (large download): `INSTALL_TORCH_CUDA=1 ./install-deps-linux.sh`  
  Or follow [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and install `torch`/`torchvision` yourself.

**Manual pip (any OS):**

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### WSL: pip / torch install failed (`I/O error`, `Bus error`, timeout)

- Free disk space on **Windows** and in WSL (`df -h`, `df -h /tmp`).
- The script sets `TMPDIR` under **`$HOME/.cache/...`** (Linux FS) so pip does not unpack huge wheels only in `/tmp` or on `/mnt/d/`.
- Remove a broken venv and retry: `rm -rf .venv` then `./install-deps-linux.sh` again.

### Windows (cmd / PowerShell)

Use a venv created **on Windows** (not one created in WSL). From the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
### 2) Train shadow detection (writes the only learned model)

```bash
python main.py train-detection --data data/ISTD_Dataset --epochs 50 --batch-size 4
```

Checkpoint: **`models/shadow_detection.pth`**

Or run the script directly (same default data path):

```bash
python src/shadow_detection.py
```

### 3) Run detection + CV removal on one image

```bash
python main.py run --image path/to/your_photo.jpg --output outputs/result.png
```

Optional: also save the **predicted mask** so you can see what the detector did:

```bash
python main.py run --image path/to/your_photo.jpg --output outputs/result.png --save-mask outputs/mask.png
```

If your weights are not in the default location:

```bash
python main.py run --image in.jpg --output out.png --detector-weights /full/path/to/shadow_detection.pth
```

### Outputs

- **`--output`** — input image after LAB-based shadow reduction (CV).  
- **`--save-mask`** — grayscale PNG of the detector’s shadow probability map.

**If you see a `shadow_detection/` folder** with only `version` / `byteorder`: that is not a valid checkpoint. You need the single file **`models/shadow_detection.pth`** produced by training (or pass `--detector-weights` to your `.pth` file).
