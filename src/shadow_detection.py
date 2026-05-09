import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNet
from preprocessing import ShadowDataset


def train_shadow_detection(data_path, epochs=50, batch_size=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ShadowDataset(data_path, img_size=256)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = UNet().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\nStarting training...\n")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, masks in train_loader:

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