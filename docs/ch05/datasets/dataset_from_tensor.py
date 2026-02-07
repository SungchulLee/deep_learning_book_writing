"""
Dataset from Tensor — Linear Regression
========================================
Demonstrates TensorDataset + DataLoader for a simple linear regression task.
The true model is  y = 1 + 2x + ε,  ε ~ N(0, 0.1²).

Usage
-----
    python dataset_from_tensor.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="dataset_from_tensor")
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--mps", action="store_true", default=True)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

ARGS.use_cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    ARGS.device = torch.device("cuda")
elif ARGS.use_mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

ARGS.train_kwargs = {"batch_size": ARGS.batch_size}
ARGS.test_kwargs = {"batch_size": ARGS.batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.path = "./model/model.pth"
os.makedirs("./model", exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data():
    x_train = np.random.uniform(size=(ARGS.batch_size, 1))
    x_test = np.random.uniform(size=(ARGS.batch_size, 1))

    y_train = 1 + 2 * x_train + np.random.normal(scale=0.1, size=(ARGS.batch_size, 1))
    y_test = 1 + 2 * x_test + np.random.normal(scale=0.1, size=(ARGS.batch_size, 1))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    trainloader = DataLoader(train_ds, **ARGS.train_kwargs)
    testloader = DataLoader(test_ds, **ARGS.test_kwargs)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def plot_test_model(model, testloader):
    for x_batch, y_batch in testloader:
        pred = model(x_batch.to(ARGS.device))
        _, ax = plt.subplots(figsize=(12, 3))
        ax.plot(x_batch.squeeze().detach(), y_batch.squeeze().detach(), "k.", label="data")
        ax.plot(x_batch.squeeze().detach(), pred.detach().cpu(), "r-", label="pred")
        ax.legend()
        plt.show()
        break


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, loss_fn, opt, trainloader):
    model.train()
    w_trace, b_trace, loss_trace = [], [], []

    for _ in range(ARGS.epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(ARGS.device), yb.to(ARGS.device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            w_trace.append(model.weight.item())
            b_trace.append(model.bias.item())
            loss_trace.append(loss.item())

    return np.array(w_trace), np.array(b_trace), np.array(loss_trace)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    trainloader, testloader = load_data()

    model = nn.Linear(1, 1).to(ARGS.device)
    loss_fn = F.mse_loss
    opt = optim.SGD(model.parameters(), lr=ARGS.lr)

    w_trace, b_trace, loss_trace = train(model, loss_fn, opt, trainloader)

    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(w_trace, label="estimated slope")
    axes[0].plot(2 * np.ones_like(w_trace), "--r", label="true slope")
    axes[1].plot(b_trace, label="estimated bias")
    axes[1].plot(np.ones_like(b_trace), "--r", label="true bias")
    axes[2].plot(loss_trace, label="loss")
    for ax in axes:
        ax.legend()
    plt.show()

    torch.save(model.state_dict(), ARGS.path)
    model = nn.Linear(1, 1).to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))

    plot_test_model(model, testloader)


if __name__ == "__main__":
    main()
