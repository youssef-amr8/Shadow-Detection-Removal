import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from preprocessing import ShadowDataset


def train_shadow_detection(data_path, epochs=50, batch_size=4, device: str | None = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("Using device:", device)

    dataset = ShadowDataset(data_path, img_size=256)
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    print(
        f"Dataset: {len(dataset)} image pairs → ~{n_batches} batches/epoch "
        f"(CPU training can take many minutes per epoch; progress updates each batch.)\n"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...\n")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            unit="batch",
        ):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/shadow_detection.pth")

    print("\nModel saved → models/shadow_detection.pth")


if __name__ == "__main__":

    train_shadow_detection(
        data_path="data/ISTD_Dataset",
        epochs=50,
        batch_size=4
    )