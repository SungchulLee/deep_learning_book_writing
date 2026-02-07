"""
CIFAR-10 CNN Classifier
========================

A LeNet-style convolutional neural network for CIFAR-10.  Two
convolutional layers with max-pooling are followed by three fully-
connected layers (120 → 84 → 10).

This is the **base** CNN variant — batch normalisation and dropout are
defined in the model but **disabled** at runtime so that we can compare
the clean CNN against the regularised variants in the companion scripts.

Source: adapted from PyTorch CIFAR-10 tutorial
        https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py

Architecture
------------
::

    Conv2d(3→6, 5×5) → ReLU → MaxPool(2)
    Conv2d(6→16, 5×5) → ReLU → MaxPool(2)
    Flatten
    Linear(400→120) → ReLU
    Linear(120→84)  → ReLU
    Linear(84→10)           (logits)

Run
---
    python cifar10_cnn.py
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# 1. Configuration
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="test batch size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    args.train_kwargs = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 2}
    args.test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False, "num_workers": 2}
    if use_cuda:
        args.train_kwargs.update({"num_workers": 1, "pin_memory": True})
        args.test_kwargs.update({"num_workers": 1, "pin_memory": True})

    args.classes = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
    args.model_path = "./model/cnn.pth"
    os.makedirs("./model", exist_ok=True)

    # Regularisation flags — both OFF for the base CNN
    args.use_batchnorm = False
    args.use_dropout = False
    args.dropout_prob = 0.5

    return args


# =============================================================================
# 2. Data loading
# =============================================================================
def load_data(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, **args.train_kwargs)
    test_loader = DataLoader(testset, **args.test_kwargs)
    return train_loader, test_loader


# =============================================================================
# 3. Model
# =============================================================================
class ConvNet(nn.Module):
    """LeNet-style CNN with optional batch-norm and dropout.

    Parameters
    ----------
    use_batchnorm : bool
        If *True*, apply ``BatchNorm2d`` after each conv layer.
    use_dropout : bool
        If *True*, apply ``Dropout`` after the first two FC layers.
    dropout_prob : float
        Drop probability (used only when *use_dropout* is True).
    """

    def __init__(self, use_batchnorm=False, use_dropout=False, dropout_prob=0.5):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        # Classifier head
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.drop1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(120, 84)
        self.drop2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # --- Conv block 1 ---
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.pool(F.relu(x))

        # --- Conv block 2 ---
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.pool(F.relu(x))

        # --- Classifier ---
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.drop1(x)

        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.drop2(x)

        x = self.fc3(x)
        return x


# =============================================================================
# 4. Visualisation helper
# =============================================================================
def show_predictions(loader, model, args, title=""):
    images, labels = next(iter(loader))
    images, labels = images.to(args.device), labels.to(args.device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    _, preds = torch.max(outputs, 1)

    images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()
    n = min(len(images), 10)
    _, axes = plt.subplots(1, n, figsize=(20, 3))
    if title:
        plt.suptitle(title, fontsize=14)

    for i in range(n):
        img = images[i] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"label: {args.classes[labels[i]]}\npred: {args.classes[preds[i]]}"
        )
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. Training loop
# =============================================================================
def train(model, train_loader, optimizer, criterion, args):
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"  [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0


# =============================================================================
# 6. Evaluation
# =============================================================================
def evaluate(model, test_loader, args):
    model.eval()

    correct = 0
    total = 0
    correct_per_class = {c: 0 for c in args.classes}
    total_per_class = {c: 0 for c in args.classes}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for lbl, pred in zip(labels, predicted):
                cls_name = args.classes[lbl]
                total_per_class[cls_name] += 1
                if lbl == pred:
                    correct_per_class[cls_name] += 1

    print(f"\nOverall accuracy on 10 000 test images: {100 * correct // total}%")
    for cls_name in args.classes:
        acc = 100.0 * correct_per_class[cls_name] / total_per_class[cls_name]
        print(f"  {cls_name:>5s}: {acc:.1f}%")


# =============================================================================
# 7. Save / Load helpers
# =============================================================================
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"\nModel saved → {path}")


def load_model(path, device, args):
    model = ConvNet(
        use_batchnorm=args.use_batchnorm,
        use_dropout=args.use_dropout,
        dropout_prob=args.dropout_prob,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


# =============================================================================
# 8. Main
# =============================================================================
def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"BatchNorm: {args.use_batchnorm}  |  Dropout: {args.use_dropout}\n")

    train_loader, test_loader = load_data(args)

    model = ConvNet(
        use_batchnorm=args.use_batchnorm,
        use_dropout=args.use_dropout,
        dropout_prob=args.dropout_prob,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    show_predictions(test_loader, model, args, title="Before Training")
    train(model, train_loader, optimizer, criterion, args)
    show_predictions(test_loader, model, args, title="After Training")

    save_model(model, args.model_path)
    model = load_model(args.model_path, args.device, args)
    evaluate(model, test_loader, args)


if __name__ == "__main__":
    main()
