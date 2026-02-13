"""
Gradient Descent for Linear Regression — PyTorch (Manual)
==========================================================

Manual gradient computation with PyTorch tensors and DataLoader.
No nn.Module, no autograd — pure tensor operations.

Demonstrates:
- TensorDataset / DataLoader for batching
- Manual gradient: g = (2/B) X^T (Xw + b - y)
- In-place parameter updates
- Convergence tracking

Author: Deep Learning Foundations Curriculum
"""

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="manual PyTorch gradient descent")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

# ============================================================================
# Synthetic Data
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

# Split
n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=True)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"True w: {w_true.numpy()}")
print(f"True b: {b_true}")

# ============================================================================
# Manual Training Loop
# ============================================================================

w = torch.zeros(p)
b = torch.zeros(1)

history = []
for epoch in range(ARGS.epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in loader:
        # Forward
        y_pred = X_batch @ w + b
        residual = y_pred - y_batch
        loss = (residual ** 2).mean()

        # Manual gradients
        B = len(y_batch)
        grad_w = (2.0 / B) * (X_batch.T @ residual)
        grad_b = (2.0 / B) * residual.sum()

        # Update
        w -= ARGS.lr * grad_w
        b -= ARGS.lr * grad_b

        epoch_loss += loss.item() * B

    avg_loss = epoch_loss / n_train
    history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# Evaluation
# ============================================================================

with torch.no_grad():
    y_pred_test = X_test @ w + b
    test_mse = ((y_pred_test - y_test) ** 2).mean().item()
    ss_res = ((y_test - y_pred_test) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    test_r2 = 1.0 - ss_res / ss_tot

print(f"\nTest MSE: {test_mse:.6f}")
print(f"Test R²:  {test_r2:.6f}")
print(f"\nLearned w: {w.numpy()}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b.item():.4f}  (true: {b_true})")

# ============================================================================
# Convergence Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Manual PyTorch GD — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gradient_descent_torch.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: gradient_descent_torch.png")
