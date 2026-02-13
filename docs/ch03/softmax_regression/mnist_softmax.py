"""
MNIST Softmax Regression with PyTorch
=======================================

Single-layer linear classifier (784 → 10) trained with
CrossEntropyLoss and SGD on the MNIST dataset.

Demonstrates:
- torchvision MNIST loading with transforms
- DataLoader with batching and shuffling
- nn.Linear as a softmax classifier (logits + CrossEntropyLoss)
- Training loop with epoch/batch loss reporting
- Model save/load with state_dict
- Overall and per-class accuracy evaluation
- Visualization of predictions on test images

Author: Deep Learning Foundations
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="mnist_softmax_regression")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--epochs", type=int, default=2, help="training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="train batch size")
parser.add_argument("--test_batch_size", type=int, default=1000, help="test batch size")
parser.add_argument("--input_size", type=int, default=784, help="flattened input dim")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--cuda", action="store_true", default=True, help="enable CUDA")
parser.add_argument("--mps", action="store_true", default=True, help="enable MPS")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

# Device selection
if ARGS.cuda and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif ARGS.mps and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
MODEL_PATH = "./model/model.pth"
os.makedirs("./model", exist_ok=True)

# DataLoader kwargs
train_kwargs = {"batch_size": ARGS.batch_size}
test_kwargs = {"batch_size": ARGS.test_batch_size}
if DEVICE.type == "cuda":
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = data.DataLoader(trainset, **train_kwargs)
    testloader = data.DataLoader(testset, **test_kwargs)
    return trainloader, testloader

# ============================================================================
# Model
# ============================================================================

class Net(nn.Module):
    """Single linear layer: flatten → 784 → 10 logits."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(ARGS.input_size, len(CLASSES))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x

# ============================================================================
# Visualization
# ============================================================================

def show_predictions(dataloader, model):
    """Display up to 10 test images with true and predicted labels."""
    model.eval()
    for images, labels in dataloader:
        outputs = model(images.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()

        n_show = min(ARGS.batch_size, 10)
        _, axes = plt.subplots(1, n_show, figsize=(12, 3))
        for i in range(n_show):
            img = images[i].numpy().transpose((1, 2, 0))
            axes[i].imshow(img, cmap="binary")
            axes[i].axis("off")
            axes[i].set_title(
                f"label: {CLASSES[labels[i]]}\npred: {CLASSES[predicted[i]]}",
                fontsize=9,
            )
        plt.tight_layout()
        plt.show()
        break

# ============================================================================
# Training
# ============================================================================

def train(model, trainloader, loss_ftn, optimizer, scheduler):
    model.train()
    for epoch in range(ARGS.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_ftn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(
                    f"  [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}"
                )
                running_loss = 0.0

        scheduler.step()

# ============================================================================
# Evaluation
# ============================================================================

def compute_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    correct_pred = {c: 0 for c in CLASSES}
    total_pred = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, pred in zip(labels, predicted):
                if label == pred:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    print(f"\nOverall accuracy: {100 * correct // total}%")
    for cls, cnt in correct_pred.items():
        acc = 100.0 * cnt / total_pred[cls]
        print(f"  Class {cls}: {acc:.1f}%")

# ============================================================================
# Save / Load
# ============================================================================

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    model = Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"Model loaded from {MODEL_PATH}")
    return model

# ============================================================================
# Main
# ============================================================================

def main():
    trainloader, testloader = load_data()

    model = Net().to(DEVICE)
    loss_ftn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print("Before training:")
    show_predictions(testloader, model)

    print(f"\nTraining for {ARGS.epochs} epochs...")
    train(model, trainloader, loss_ftn, optimizer, scheduler)

    print("\nAfter training:")
    show_predictions(testloader, model)

    save_model(model)
    model = load_model()
    compute_accuracy(model, testloader)


if __name__ == "__main__":
    main()
