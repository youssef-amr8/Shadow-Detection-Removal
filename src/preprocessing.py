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