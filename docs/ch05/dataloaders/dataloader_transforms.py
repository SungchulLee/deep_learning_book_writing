"""
DataLoader with Transforms — FashionMNIST
==========================================
Demonstrates ``target_transform`` with ``Lambda`` to convert integer labels
into one-hot encoded tensors, combined with ``ToTensor`` for images.

Usage
-----
    python dataloader_transforms.py
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

LABELS_MAP = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat",    5: "Sandal",  6: "Shirt",    7: "Sneaker",
    8: "Bag",     9: "Ankle Boot",
}


def load_dataloader():
    one_hot = Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    )

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot,
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=one_hot,
    )

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=20, shuffle=True)
    return train_loader, test_loader


def draw_batch(dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, img, label in zip(axes.reshape(-1), images, labels):
        ax.imshow(img.squeeze(), cmap="binary")
        # Find the class index from the one-hot vector
        class_idx = label.argmax().item()
        ax.set_title(LABELS_MAP[class_idx])
        ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_batch(train_loader)

    # Show one-hot label for the first sample
    images, labels = next(iter(train_loader))
    print(f"Image shape:  {images[0].shape}")
    print(f"Label (one-hot): {labels[0]}")
    print(f"Class index:  {labels[0].argmax().item()}")
    print(f"Class name:   {LABELS_MAP[labels[0].argmax().item()]}")


if __name__ == "__main__":
    main()
