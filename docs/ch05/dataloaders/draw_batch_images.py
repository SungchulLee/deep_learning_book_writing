"""
Draw Batch Images — FashionMNIST
=================================
Loads a batch from FashionMNIST and displays 20 sample images in a 2 × 10
grid with their labels.

Usage
-----
    python draw_batch_images.py
"""

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

LABELS_MAP = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat",    5: "Sandal",  6: "Shirt",    7: "Sneaker",
    8: "Bag",     9: "Ankle Boot",
}


def load_dataloader():
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=20, shuffle=True)
    return train_loader, test_loader


def draw_batch(dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, img, label in zip(axes.reshape(-1), images, labels):
        ax.imshow(img.squeeze(), cmap="binary")
        ax.set_title(LABELS_MAP[label.item()])
        ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_batch(train_loader)


if __name__ == "__main__":
    main()
