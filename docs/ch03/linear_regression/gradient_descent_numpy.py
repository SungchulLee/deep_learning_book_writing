"""
Gradient Descent for Linear Regression — NumPy
================================================

Implements batch and mini-batch gradient descent from scratch.

Demonstrates:
- Design matrix with bias column
- Full-batch gradient: g = (2/n) X^T (Xθ - y)
- Mini-batch shuffling via rng.permutation
- Learning-rate sensitivity analysis
- Convergence comparison: batch vs mini-batch

Author: Deep Learning Foundations Curriculum
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="gradient descent linear regression")
parser.add_argument("--n-samples", type=int, default=300)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--noise", type=float, default=10.0)
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

rng = np.random.default_rng(ARGS.seed)

# ============================================================================
# Data
# ============================================================================

x, y = make_regression(
    n_samples=ARGS.n_samples,
    n_features=ARGS.n_features,
    noise=ARGS.noise,
    random_state=ARGS.seed,
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=ARGS.seed,
)


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x])


X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)
n, d = X_train.shape
print(f"Train: ({n}, {d}), Test: {X_test.shape}")

# ============================================================================
# Batch Gradient Descent
# ============================================================================


def batch_gd(X, y, lr, epochs):
    theta = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        residual = X @ theta - y
        loss = np.mean(residual ** 2)
        losses.append(loss)
        grad = (2.0 / len(y)) * (X.T @ residual)
        theta -= lr * grad
    return theta, losses


theta_batch, losses_batch = batch_gd(X_train, y_train, ARGS.lr, ARGS.epochs)
print(f"\nBatch GD — final MSE: {losses_batch[-1]:.4f}")

# ============================================================================
# Mini-Batch Gradient Descent
# ============================================================================


def minibatch_gd(X, y, lr, epochs, batch_size, rng):
    n = len(y)
    theta = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        X_s, y_s = X[perm], y[perm]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_b, y_b = X_s[start:end], y_s[start:end]
            residual = X_b @ theta - y_b
            grad = (2.0 / len(y_b)) * (X_b.T @ residual)
            theta -= lr * grad
        full_loss = np.mean((X @ theta - y) ** 2)
        losses.append(full_loss)
    return theta, losses


theta_mini, losses_mini = minibatch_gd(
    X_train, y_train, ARGS.lr, ARGS.epochs, ARGS.batch_size, rng,
)
print(f"Mini-batch GD — final MSE: {losses_mini[-1]:.4f}")

# ============================================================================
# Normal Equation Reference
# ============================================================================

theta_exact = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
y_pred_exact = X_test @ theta_exact
mse_exact = np.mean((y_test - y_pred_exact) ** 2)
print(f"Normal equation — test MSE: {mse_exact:.4f}")

# ============================================================================
# Evaluation
# ============================================================================

for name, theta in [("Batch GD", theta_batch), ("Mini-batch GD", theta_mini)]:
    y_pred = X_test @ theta
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    diff = np.max(np.abs(theta - theta_exact))
    print(f"{name:15s}  test MSE={mse:.4f}  R²={r2:.4f}  max|Δθ|={diff:.2e}")

# ============================================================================
# Convergence Plot
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(losses_batch, label="Batch GD", alpha=0.8)
ax.plot(losses_mini, label="Mini-batch GD", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Convergence: Batch vs Mini-batch")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(losses_batch, label="Batch GD", alpha=0.8)
ax.semilogy(losses_mini, label="Mini-batch GD", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Convergence (Log Scale)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent_numpy.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: gradient_descent_numpy.png")
