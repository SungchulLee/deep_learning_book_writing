"""
Minimal PyTorch Linear Regression — Autograd
==============================================

Uses requires_grad=True on raw tensors and loss.backward() for
automatic gradient computation.  No nn.Module, no optimizer object.

Demonstrates:
- requires_grad=True
- loss.backward() for autograd
- torch.no_grad() context for parameter updates
- .grad.zero_() to clear accumulated gradients
- .detach() to extract values from the computation graph

Author: Deep Learning Foundations Curriculum
"""

import argparse
import torch
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="minimal PyTorch autograd")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

# ============================================================================
# Data
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# ============================================================================
# Parameters with Gradient Tracking
# ============================================================================

w = torch.zeros(p, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ============================================================================
# Training Loop
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    # Forward pass (full batch)
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()

    # Backward pass — autograd fills w.grad and b.grad
    loss.backward()

    # Parameter update — must not be tracked by autograd
    with torch.no_grad():
        w -= ARGS.lr * w.grad
        b -= ARGS.lr * b.grad

    # CRITICAL: zero gradients (PyTorch accumulates by default)
    w.grad.zero_()
    b.grad.zero_()

    history.append(loss.item())

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {loss.item():.6f}")

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
print(f"\nLearned w: {w.detach().numpy()}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b.detach().item():.4f}  (true: {b_true})")

# ============================================================================
# Convergence Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Minimal PyTorch (Autograd) — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("minimal_torch.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: minimal_torch.png")
