"""
Multiple Output Linear Regression
===================================

Multi-target regression: nn.Linear(3, 2) mapping R^3 → R^2.

Demonstrates:
- nn.Linear(p, q) with q > 1
- Weight matrix shapes: stored as (q, p), computed as x @ W^T + b
- Per-output R² evaluation
- Normal equation verification for multi-output case

Author: Deep Learning Foundations Curriculum
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="multi-output linear regression")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)

# ============================================================================
# Synthetic Data: 3 inputs → 2 outputs
# ============================================================================

n, p, q = ARGS.n_samples, 3, 2
X = torch.randn(n, p)
W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.8]])  # (p, q) = (3, 2)
b_true = torch.tensor([1.0, -2.0])                                # (q,)
Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)

n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=ARGS.batch_size, shuffle=True)

print(f"Data shapes: X={X.shape}, Y={Y.shape}")
print(f"W_true (p×q):\n{W_true}")
print(f"b_true: {b_true}")

# ============================================================================
# Model: nn.Linear(3, 2)
# ============================================================================

model = nn.Linear(p, q)
optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr)

print(f"\nModel: {model}")
print(f"Weight shape: {model.weight.shape}  (stored as q×p = {q}×{p})")
print(f"Bias shape:   {model.bias.shape}")

# ============================================================================
# Training
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    epoch_loss = 0.0
    for X_b, Y_b in loader:
        Y_pred = model(X_b)                 # (B, q)
        loss = F.mse_loss(Y_pred, Y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(Y_b)

    avg_loss = epoch_loss / n_train
    history.append(avg_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# Evaluation
# ============================================================================

with torch.no_grad():
    Y_pred_test = model(X_test)
    overall_mse = F.mse_loss(Y_pred_test, Y_test).item()

    print(f"\nOverall Test MSE: {overall_mse:.6f}")
    for j in range(q):
        ss_res = ((Y_test[:, j] - Y_pred_test[:, j]) ** 2).sum().item()
        ss_tot = ((Y_test[:, j] - Y_test[:, j].mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot
        print(f"Output {j}: R² = {r2:.4f}")

# ============================================================================
# Compare with True Parameters
# ============================================================================

W_learned = model.weight.detach()  # (q, p)
b_learned = model.bias.detach()    # (q,)

print(f"\nLearned W^T (q×p):\n{W_learned}")
print(f"True    W^T (q×p):\n{W_true.T}")
print(f"\nLearned b: {b_learned}")
print(f"True    b: {b_true}")

# ============================================================================
# Normal Equation Verification
# ============================================================================

print("\n--- Normal Equation Verification ---")
X_np = np.column_stack([np.ones(n_train), X_train.numpy()])
Y_np = Y_train.numpy()
B_star = np.linalg.solve(X_np.T @ X_np, X_np.T @ Y_np)
print(f"Normal eq bias: {B_star[0]}")
print(f"Normal eq W:\n{B_star[1:]}")

# ============================================================================
# Convergence Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title(f"Multi-Output Regression ({p}→{q}) — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multiple_outputs.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: multiple_outputs.png")
