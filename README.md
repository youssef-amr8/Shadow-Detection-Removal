# Shadow Detection & Removal

Dataset: [ISTD on Kaggle](https://www.kaggle.com/datasets/sabarinathan/istd-dataset/data) → `data/ISTD_Dataset`

## Quick Start

# Train
### Detection model
```bash
python main.py train-detection --data data/ISTD_Dataset --epochs 50 --batch-size 4
```
### Removal model
```bash
python main.py train-removal --data data/ISTD_Dataset --epochs 50 --batch-size 8
```

# Run 
### Single Image
Single image (e.g., 99-4.png from test directory)
result saved in output --> out.png
```bash
python main.py run --image data/ISTD_Dataset/test/test_A/99-4.png --output outputs/out.png \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth -v
```

### Batch of 20 random images with metrics (PSNR, SSIM)
results saved in outputs/removed

json file with results in also generated with the result images inside outputs/removed
```bash
python main.py eval --input-dir data/ISTD_Dataset/test/test_A \
  --gt-dir data/ISTD_Dataset/test/test_C \
  --output-dir outputs/removed --num-images 20 \
  --detector-weights models/shadow_detection.pth \
  --removal-weights models/shadow_removal.pth -v
```

## Test Detection Alone
10 random images with metrics
results saved in outputs/test_results
```bash
python src/shadow_detection_test.py
```

