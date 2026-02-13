"""
CIFAR-10 Feedforward Classifier
================================

A simple fully-connected (feedforward) network for CIFAR-10 image
classification.  The model flattens each 32×32×3 image into a 3 072-dim
vector and passes it through a three-layer MLP (512 → 512 → 10) with
ReLU activations.

This serves as a **baseline** that ignores spatial structure — comparing
its accuracy with the CNN variants in this chapter highlights exactly why
convolutional layers matter for vision tasks.

Source: adapted from PyTorch "Build the Neural Network" tutorial
        https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/buildmodel_tutorial.py

Workflow
--------
1.  Parse hyperparameters (learning rate, momentum, epochs, batch size).
2.  Load CIFAR-10 with per-channel normalisation to [-1, 1].
3.  Build a ``Flatten → Linear(3072,512) → ReLU → Linear(512,512)
    → ReLU → Linear(512,10)`` network.
4.  Train with SGD + momentum and cross-entropy loss.
5.  Evaluate overall and per-class accuracy on the test set.
6.  Visualise sample predictions before and after training.

Run
---
    python cifar10_feedforward.py
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# 1. Configuration
# =============================================================================
def parse_args():
    """Parse command-line arguments (all have sensible defaults)."""
    parser = argparse.ArgumentParser(description="CIFAR-10 Feedforward")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="test batch size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # DataLoader kwargs
    args.train_kwargs = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 2}
    args.test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False, "num_workers": 2}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        args.train_kwargs.update(cuda_kwargs)
        args.test_kwargs.update(cuda_kwargs)

    args.classes = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
    args.model_path = "./model/feedforward.pth"
    os.makedirs("./model", exist_ok=True)

    return args


# =============================================================================
# 2. Data loading
# =============================================================================
def load_data(args):
    """Return CIFAR-10 train and test DataLoaders.

    Each channel is normalised to zero mean and unit range:
        x_norm = (x - 0.5) / 0.5  ∈  [-1, 1]
    """
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
class FeedforwardNet(nn.Module):
    """Three-layer MLP that flattens the 32×32×3 image first.

    Architecture
    ------------
    Flatten → Linear(3072, 512) → ReLU
            → Linear(512,  512) → ReLU
            → Linear(512,   10)           (logits)
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# =============================================================================
# 4. Visualisation helper
# =============================================================================
def show_predictions(loader, model, args, title=""):
    """Display up to 10 test images with true and predicted labels."""
    images, labels = next(iter(loader))
    images, labels = images.to(args.device), labels.to(args.device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # Move back to CPU for plotting
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    n = min(len(images), 10)
    _, axes = plt.subplots(1, n, figsize=(20, 3))
    if title:
        plt.suptitle(title, fontsize=14)

    for i in range(n):
        img = images[i] / 2 + 0.5                  # unnormalise → [0, 1]
        img = np.transpose(img.numpy(), (1, 2, 0))  # CHW → HWC
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
    """Train the model for ``args.epochs`` epochs."""
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
    """Compute overall and per-class accuracy on the test set."""
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


def load_model(path, device):
    model = FeedforwardNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


# =============================================================================
# 8. Main
# =============================================================================
def main():
    args = parse_args()
    print(f"Device: {args.device}\n")

    train_loader, test_loader = load_data(args)

    model = FeedforwardNet().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Predictions before training (random weights)
    show_predictions(test_loader, model, args, title="Before Training")

    # Train
    train(model, train_loader, optimizer, criterion, args)

    # Predictions after training
    show_predictions(test_loader, model, args, title="After Training")

    # Save → reload → evaluate
    save_model(model, args.model_path)
    model = load_model(args.model_path, args.device)
    evaluate(model, test_loader, args)


if __name__ == "__main__":
    main()
