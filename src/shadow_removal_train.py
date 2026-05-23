"""Train U-Net shadow removal (ISTD train_A → train_C)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from preprocessing import ShadowRemovalDataset
from utils import repo_root


def train_shadow_removal(
    data_path: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 256,
    lr: float = 1e-4,
    device: str | None = None,
    num_workers: int = 0,
    save_path: str | Path | None = None,
) -> Path:
    """
    Train U-Net: (RGB + mask) → shadow-free RGB.

    Uses train_A, train_B (mask), train_C. Saves ``models/shadow_removal.pth``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("Using device:", device)

    dataset = ShadowRemovalDataset(data_path, img_size=img_size)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    print(
        f"Dataset: {len(dataset)} triplets (train_A + train_B → train_C) "
        f"→ ~{n_batches} batches/epoch @ size {img_size}\n"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(in_channels=4, out_channels=3).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting shadow removal training...\n")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, clean in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            unit="batch",
        ):
            x = x.to(device)
            clean = clean.to(device)

            pred = model(x)
            loss = criterion(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] L1 Loss: {total_loss / len(train_loader):.4f}")

    if save_path is None:
        save_path = repo_root() / "models" / "shadow_removal.pth"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path.resolve()}")
    return save_path


if __name__ == "__main__":
    train_shadow_removal(
        data_path="data/ISTD_Dataset",
        epochs=50,
        batch_size=8,
    )
