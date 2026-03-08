"""Wavenet basic."""
# ---
# title: "WaveNet: Basic Dilated Causal Convolutions"
# description: "PyTorch implementation of dilated causal convolutions for sequence modelling"
# ---
#
# WaveNet (van den Oord et al., 2016) uses stacks of dilated causal
# convolutions to achieve exponentially growing receptive fields without
# an exponential growth in parameters.
#
# This script implements the core building blocks in PyTorch:
#   1. Causal Conv1d (left-padded to ensure no future leakage)
#   2. Simple dilated-conv stack (no gating)
#   3. Training on a synthetic time-series forecasting task
#
# Adapted from: O'Reilly Hands-On ML, Chapter 15 (TF → PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# Main
# ========================================================================


# ─── Causal Conv1d ─────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """Conv1d with causal (left) padding so the output at time t depends
    only on inputs at times ≤ t."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,          # we pad manually
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.pad(x, (self.padding, 0))   # left-pad only
        return self.conv(x)


# ─── Simple Dilated Stack ──────────────────────────────────────────────────
class SimpleDilatedStack(nn.Module):
    """Stack of dilated causal convolutions with ReLU activations.
    Dilation rates double each layer: 1, 2, 4, 8, 1, 2, 4, 8, …"""

    def __init__(self, in_channels=1, hidden=20, n_layers=8, kernel_size=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** (i % 4)          # cycle: 1,2,4,8
            c_in = in_channels if i == 0 else hidden
            layers.append(CausalConv1d(c_in, hidden, kernel_size, dilation))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(hidden, 10, kernel_size=1))  # point-wise out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Data generation ───────────────────────────────────────────────────────
def generate_time_series(batch_size, n_steps):
    """Generate a simple multi-step-ahead prediction target."""
    freq1, freq2 = np.random.uniform(0.1, 0.5, (2, batch_size, 1))
    offsets1, offsets2 = np.random.uniform(0, 2 * np.pi, (2, batch_size, 1))
    time = np.linspace(0, 1, n_steps + 10).reshape(1, -1)
    series = 0.5 * np.sin((time - offsets1) * (n_steps * freq1))
    series += 0.2 * np.sin((time - offsets2) * (n_steps * freq2))
    series += 0.1 * (np.random.randn(batch_size, n_steps + 10) - 0.5)
    return series.astype(np.float32)


np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps)
X_train = torch.tensor(series[:7000, :n_steps]).unsqueeze(1)      # (B,1,T)
Y_train = torch.tensor(series[:7000, 1:n_steps + 1]).unsqueeze(1)
X_valid = torch.tensor(series[7000:9000, :n_steps]).unsqueeze(1)
Y_valid = torch.tensor(series[7000:9000, 1:n_steps + 1]).unsqueeze(1)
X_test  = torch.tensor(series[9000:, :n_steps]).unsqueeze(1)
Y_test  = torch.tensor(series[9000:, 1:n_steps + 1]).unsqueeze(1)

print(f"X_train shape: {X_train.shape}  Y_train shape: {Y_train.shape}")

# ─── Training ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDilatedStack(in_channels=1, hidden=20, n_layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}\n")

epochs = 20
batch_size = 128
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X_train[idx].to(device), Y_train[idx].to(device)
        pred = model(xb)
        # Only compare the last 10 time steps (model outputs 10 channels)
        loss = loss_fn(pred[:, :, -10:], yb[:, :, -10:].expand_as(pred[:, :, -10:]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    train_losses.append(epoch_loss / n_batches)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_valid.to(device))
        val_loss = loss_fn(val_pred[:, :, -10:], Y_valid[:, :, -10:].to(device).expand_as(val_pred[:, :, -10:]))
        val_losses.append(val_loss.item())

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}")

# ─── Plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="valid")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Simple Dilated Causal Conv — Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("wavenet_basic_training.png", dpi=150)
plt.show()
print("Done.")


if __name__ == "__main__":
    pass