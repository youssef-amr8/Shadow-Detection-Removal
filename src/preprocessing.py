import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ShadowDataset(Dataset):
    def __init__(self, base_path, img_size=256):

        self.A_path = os.path.join(base_path, "train", "train_A")
        self.B_path = os.path.join(base_path, "train", "train_B")

        self.files = sorted(os.listdir(self.A_path))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        img_path = os.path.join(self.A_path, file)
        mask_path = os.path.join(self.B_path, file)

        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size))

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


class ShadowRemovalDataset(Dataset):
    """
    ISTD guided removal: concat(train_A RGB, train_B mask) → train_C clean RGB.

    At train time uses ground-truth masks (train_B).
    At inference the mask comes from the detection U-Net.
    """

    def __init__(self, base_path, img_size=256):
        self.A_path = os.path.join(base_path, "train", "train_A")
        self.B_path = os.path.join(base_path, "train", "train_B")
        self.C_path = os.path.join(base_path, "train", "train_C")

        for name, p in [("train_A", self.A_path), ("train_B", self.B_path), ("train_C", self.C_path)]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Missing {name}: {p}")

        a_files = set(os.listdir(self.A_path))
        b_files = set(os.listdir(self.B_path))
        c_files = set(os.listdir(self.C_path))
        self.files = sorted(a_files & b_files & c_files)
        if not self.files:
            raise FileNotFoundError(
                f"No matching filenames in train_A, train_B, train_C under {base_path}"
            )
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        shadow = Image.open(os.path.join(self.A_path, file)).convert("RGB")
        mask = Image.open(os.path.join(self.B_path, file)).convert("L")
        clean = Image.open(os.path.join(self.C_path, file)).convert("RGB")

        size = (self.img_size, self.img_size)
        shadow = np.array(shadow.resize(size), dtype=np.float32) / 255.0
        mask = np.array(mask.resize(size), dtype=np.float32) / 255.0
        clean = np.array(clean.resize(size), dtype=np.float32) / 255.0

        shadow_t = torch.tensor(shadow).permute(2, 0, 1)
        mask_t = torch.tensor(mask).unsqueeze(0)
        clean_t = torch.tensor(clean).permute(2, 0, 1)

        # 4 channels: RGB image + shadow mask (what the net sees at inference)
        x = torch.cat([shadow_t, mask_t], dim=0)
        return x, clean_t
