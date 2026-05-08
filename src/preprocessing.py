import os
import cv2
import numpy as np

IMG_SIZE = 256


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)


def load_mask(path):
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask.astype(np.float32)


def load_dataset(base_path):
    A_path = os.path.join(base_path, "train_A")
    B_path = os.path.join(base_path, "train_B")

    X, y = [], []

    files = sorted(os.listdir(A_path))

    for f in files:
        img_path = os.path.join(A_path, f)
        mask_path = os.path.join(B_path, f)

        if os.path.exists(mask_path):
            X.append(load_image(img_path))
            y.append(load_mask(mask_path))

    return np.array(X), np.array(y)