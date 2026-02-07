"""
Face Landmarks — Custom Dataset, Transforms, and DataLoader
============================================================
Demonstrates writing a custom ``Dataset`` subclass for face landmark data,
with custom ``Rescale``, ``RandomCrop``, and ``ToTensor`` transforms.

Based on the official PyTorch data loading tutorial.

Usage
-----
    python face_landmarks.py
"""

import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DOWNLOAD_DIR = "./data"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download():
    dest = os.path.join(DOWNLOAD_DIR, "faces")
    if os.path.isdir(dest):
        return
    url = "https://download.pytorch.org/tutorial/faces.zip"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, "faces.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DOWNLOAD_DIR)
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to: {DOWNLOAD_DIR}")


# ---------------------------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------------------------


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = np.array([self.landmarks_frame.iloc[idx, 1:]])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Custom Transforms
# ---------------------------------------------------------------------------


class Rescale:
    """Rescale the image to a given size, preserving aspect ratio."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {"image": img, "landmarks": landmarks}


class RandomCrop:
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]
        landmarks = landmarks - [left, top]

        return {"image": image, "landmarks": landmarks}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # numpy: H × W × C  →  torch: C × H × W
        image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "landmarks": torch.from_numpy(landmarks),
        }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data():
    dataset = FaceLandmarksDataset(
        csv_file="data/faces/face_landmarks.csv",
        root_dir="data/faces/",
        transform=transforms.Compose([
            Rescale(256),
            RandomCrop(224),
            ToTensor(),
        ]),
    )
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    return dataloader


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def show_one_image(df_landmarks, n=65):
    img_name = df_landmarks.iloc[n, 0]
    image = io.imread(os.path.join("data/faces/", img_name))
    landmarks = np.asarray(df_landmarks.iloc[n, 1:]).astype("float").reshape(-1, 2)

    _, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")
    ax.axis("off")
    plt.show()


def show_four_images(dataloader):
    _, axes = plt.subplots(1, 4, figsize=(12, 3))
    i = -1
    for d in dataloader:
        images = d["image"]
        landmarks = d["landmarks"]
        for image, landmark in zip(images, landmarks):
            i += 1
            axes[i].imshow(image.permute(1, 2, 0))
            axes[i].scatter(landmark[:, 0], landmark[:, 1], s=10, marker=".", c="r")
            axes[i].axis("off")
            if i == 3:
                break
        if i == 3:
            break
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    download()

    df = pd.read_csv(f"{DOWNLOAD_DIR}/faces/face_landmarks.csv")
    show_one_image(df, n=65)

    dataloader = load_data()
    show_four_images(dataloader)


if __name__ == "__main__":
    main()
