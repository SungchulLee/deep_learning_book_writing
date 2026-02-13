"""
Draw One Image — FashionMNIST
==============================
Loads FashionMNIST via DataLoader and displays a single sample image with
its label.

Usage
-----
    python draw_one_image.py
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


def draw_one_image(dataloader):
    images, labels = next(iter(dataloader))
    img, label = images[0], labels[0]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img.squeeze(), cmap="binary")
    ax.set_title(LABELS_MAP[label.item()])
    ax.axis("off")
    plt.show()


def main():
    train_loader, _ = load_dataloader()
    draw_one_image(train_loader)


if __name__ == "__main__":
    main()
