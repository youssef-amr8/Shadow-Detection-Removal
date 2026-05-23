# Shadow Detection & Removal

Dataset: [ISTD on Kaggle](https://www.kaggle.com/datasets/sabarinathan/istd-dataset/data) → `data/ISTD_Dataset`

## Quick Start

### Train

```bash
# Detection model
python main.py train-detection --data data/ISTD_Dataset --epochs 50 --batch-size 4

# Removal model
python main.py train-removal --data data/ISTD_Dataset --epochs 50 --batch-size 8
```

### Run

```bash
# Single image (e.g., 99-4.png from test directory)
python main.py run --image data/ISTD_Dataset/test/test_A/99-4.png --output outputs/out.png \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth -v

# Batch of 20 random images with metrics (PSNR, SSIM)
python main.py eval --input-dir data/ISTD_Dataset/test/test_A \
  --gt-dir data/ISTD_Dataset/test/test_C \
  --output-dir outputs/removed --num-images 20 \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth -v
```

## Test Detection Alone
#10 random images with metrics
```bash
python src/shadow_detection_test.py
```

