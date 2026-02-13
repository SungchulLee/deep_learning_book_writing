"""
Normal Equation — NumPy Implementation
========================================

Closed-form solution for linear regression:

    θ* = (X^T X)^{-1} X^T y

Demonstrates:
- Design matrix construction (prepending ones for bias)
- Normal equation solve via np.linalg.solve (numerically stable)
- Comparison with np.linalg.lstsq
- Train / test split and evaluation

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

parser = argparse.ArgumentParser(description="Normal equation linear regression")
parser.add_argument("--n-samples", type=int, default=200, help="number of samples")
parser.add_argument("--n-features", type=int, default=3, help="number of features")
parser.add_argument("--noise", type=float, default=10.0, help="noise std dev")
parser.add_argument("--seed", type=int, default=42, help="random seed")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

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
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ============================================================================
# Design Matrix
# ============================================================================


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones: X = [1 | x]."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)

# ============================================================================
# Normal Equation
# ============================================================================


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """θ* = (X^T X)^{-1} X^T y via solve (no explicit inverse)."""
    return np.linalg.solve(X.T @ X, X.T @ y)


theta = fit_normal_equation(X_train, y_train)
print(f"\nFitted parameters (bias first): {theta}")

# ============================================================================
# Comparison with lstsq
# ============================================================================

theta_lstsq, residuals, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)
print(f"lstsq parameters:               {theta_lstsq}")
print(f"Max difference: {np.max(np.abs(theta - theta_lstsq)):.2e}")

# ============================================================================
# Prediction and Evaluation
# ============================================================================


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X @ theta


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    mse = np.mean(residual ** 2)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(residual)),
        "r2": 1.0 - ss_res / ss_tot,
    }


y_pred_train = predict(X_train, theta)
y_pred_test = predict(X_test, theta)

print(f"\nTrain: {evaluate(y_train, y_pred_train)}")
print(f"Test:  {evaluate(y_test, y_pred_test)}")

# ============================================================================
# Visualisation
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="black", linewidths=0.5)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, "r--", lw=1.5)
ax.set_xlabel("True y")
ax.set_ylabel("Predicted y")
ax.set_title("Predicted vs True")
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals_test = y_test - y_pred_test
ax.hist(residuals_test, bins=20, density=True, alpha=0.7, edgecolor="black")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution (σ = {residuals_test.std():.2f})")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("normal_equation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: normal_equation.png")
