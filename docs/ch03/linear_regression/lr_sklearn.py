"""
Linear Regression with Scikit-learn
=====================================

Comparison of OLS, Ridge, Lasso, and ElasticNet on synthetic data.

Demonstrates:
- sklearn.linear_model.{LinearRegression, Ridge, Lasso, ElasticNet}
- Cross-validated variants {RidgeCV, LassoCV, ElasticNetCV}
- Pipeline with StandardScaler
- Coefficient comparison across models

Author: Deep Learning Foundations Curriculum
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="sklearn linear regression comparison")
parser.add_argument("--n-samples", type=int, default=300, help="number of samples")
parser.add_argument("--n-features", type=int, default=10, help="total features")
parser.add_argument("--n-informative", type=int, default=5, help="informative features")
parser.add_argument("--noise", type=float, default=15.0, help="noise std dev")
parser.add_argument("--seed", type=int, default=42, help="random seed")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

# ============================================================================
# Data
# ============================================================================

X, y = make_regression(
    n_samples=ARGS.n_samples,
    n_features=ARGS.n_features,
    n_informative=ARGS.n_informative,
    noise=ARGS.noise,
    random_state=ARGS.seed,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=ARGS.seed,
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# Models
# ============================================================================

models = {
    "OLS": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1),
    "ElasticNet (α=0.1, ρ=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coef = pipe.named_steps["model"].coef_
    n_nonzero = np.sum(np.abs(coef) > 1e-6)
    results[name] = {"mse": mse, "r2": r2, "coef": coef, "n_nonzero": n_nonzero}
    print(f"{name:30s}  MSE={mse:8.2f}  R²={r2:.4f}  nonzero={n_nonzero}/{len(coef)}")

# ============================================================================
# Cross-Validated Hyperparameter Selection
# ============================================================================

print("\n--- Cross-Validated Selection ---")

ridge_cv = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]))])
ridge_cv.fit(X_train, y_train)
print(f"Best Ridge α: {ridge_cv.named_steps['model'].alpha_:.4f}")

lasso_cv = Pipeline([("scaler", StandardScaler()), ("model", LassoCV(cv=5, random_state=ARGS.seed))])
lasso_cv.fit(X_train, y_train)
print(f"Best Lasso α: {lasso_cv.named_steps['model'].alpha_:.4f}")

elastic_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9], random_state=ARGS.seed)),
])
elastic_cv.fit(X_train, y_train)
m = elastic_cv.named_steps["model"]
print(f"Best ElasticNet α: {m.alpha_:.4f}, l1_ratio: {m.l1_ratio_:.2f}")

# ============================================================================
# Coefficient Comparison Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(ARGS.n_features)
width = 0.2

for i, (name, res) in enumerate(results.items()):
    ax.bar(x_pos + i * width, res["coef"], width, label=name, alpha=0.8)

ax.set_xlabel("Feature Index")
ax.set_ylabel("Coefficient Value")
ax.set_title("Coefficient Comparison Across Regularisation Methods")
ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels([f"x{i}" for i in range(ARGS.n_features)])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("sklearn_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: sklearn_comparison.png")
