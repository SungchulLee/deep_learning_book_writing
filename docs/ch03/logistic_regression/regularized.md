# Regularized Logistic Regression

## Learning Objectives

By the end of this section, you will be able to:

- Understand why regularization is necessary for logistic regression
- Derive the regularized loss functions (L2, L1, Elastic Net) and their gradients
- Explain the geometric and Bayesian interpretations of regularization
- Implement a complete, production-ready logistic regression pipeline in PyTorch
- Apply regularized logistic regression to real-world datasets with proper evaluation

---

## Why Regularize?

### The Perfect Separation Problem

When the training data is **linearly separable**, the maximum likelihood estimate for logistic regression does not exist. The optimal parameters diverge to $\|\boldsymbol{\beta}\| \to \infty$ because the model can always improve the likelihood by making predictions more confident.

Mathematically, if there exists a $\boldsymbol{\beta}^*$ such that $y_i(\mathbf{x}_i^\top \boldsymbol{\beta}^*) > 0$ for all $i$ (using the $y_i \in \{-1, +1\}$ encoding), then scaling $\boldsymbol{\beta}^* \to c\boldsymbol{\beta}^*$ with $c \to \infty$ drives every prediction toward certainty, and the log-likelihood approaches 0 (its supremum) without ever attaining it.

### Overfitting in High Dimensions

Even without perfect separation, logistic regression can overfit when:

- The number of features $d$ is large relative to $n$
- Features are highly correlated (multicollinearity)
- The model memorizes noise in the training data

Regularization constrains $\|\boldsymbol{\beta}\|$, preventing overly confident predictions and improving generalization.

---

## L2 Regularization (Ridge)

### The Regularized Objective

L2 regularization adds a penalty on the squared L2 norm of the weights:

$$
\mathcal{L}_{\text{ridge}}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \frac{\lambda}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2
$$

where $\lambda > 0$ is the regularization strength and $\boldsymbol{\beta}_{1:d}$ excludes the intercept $\beta_0$ (the intercept is typically not penalized).

### Gradient with L2 Regularization

$$
\nabla_{\boldsymbol{\beta}} \mathcal{L}_{\text{ridge}} = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y}) + \lambda \boldsymbol{\beta}
$$

(with $\lambda \beta_0 = 0$ for the intercept component). The regularization term adds a force pulling $\boldsymbol{\beta}$ toward zero at each gradient step.

### Effect on the Decision Boundary

From the [Decision Boundary](decision_boundary.md) section, recall that $\|\boldsymbol{\beta}\|$ controls the steepness of the probability transition. L2 regularization shrinks $\|\boldsymbol{\beta}\|$, producing softer probability transitions near the boundary, less confident predictions (probabilities closer to 0.5), and better calibrated models in practice.

### Bayesian Interpretation

L2 regularization is equivalent to placing a **Gaussian prior** on the weights:

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma_\beta^2 \mathbf{I}), \quad \text{where } \lambda = \frac{1}{\sigma_\beta^2}
$$

The regularized objective corresponds to **maximum a posteriori (MAP)** estimation:

$$
\boldsymbol{\beta}_{\text{MAP}} = \arg\max_{\boldsymbol{\beta}} \left[ \log P(\mathcal{D}|\boldsymbol{\beta}) + \log P(\boldsymbol{\beta}) \right]
$$

Since $\log P(\boldsymbol{\beta}) = -\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2 + \text{const}$, maximizing the posterior is equivalent to minimizing the ridge-penalized loss.

---

## L1 Regularization (Lasso)

### The Regularized Objective

L1 regularization penalizes the sum of absolute values:

$$
\mathcal{L}_{\text{lasso}}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \lambda \|\boldsymbol{\beta}_{1:d}\|_1
$$

### Sparsity-Inducing Property

The key distinction from L2 is that L1 regularization produces **sparse** solutions — many coefficients are driven exactly to zero. This makes L1 regularization a form of **automatic feature selection**.

Geometrically, the L1 constraint set $\|\boldsymbol{\beta}\|_1 \leq t$ is a diamond (cross-polytope). The loss contours are more likely to intersect the diamond at a vertex, where one or more coordinates are exactly zero.

### Bayesian Interpretation

L1 regularization corresponds to a **Laplace prior** on the weights:

$$
P(\beta_j) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)
$$

The heavy tails of the Laplace distribution allow large coefficients for truly important features, while the sharp peak at zero encourages sparsity.

---

## Elastic Net

### Combining L1 and L2

Elastic Net combines both penalties:

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\beta}) = \text{BCE} + \lambda_1 \|\boldsymbol{\beta}_{1:d}\|_1 + \frac{\lambda_2}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2
$$

Or equivalently, with a mixing parameter $\alpha \in [0, 1]$:

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\beta}) = \text{BCE} + \lambda \left[\alpha \|\boldsymbol{\beta}_{1:d}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}_{1:d}\|_2^2\right]
$$

### When to Use Each

| Method | When to Use |
|--------|-------------|
| L2 (Ridge) | Many moderately important features; correlated features |
| L1 (Lasso) | Few truly important features; feature selection desired |
| Elastic Net | Correlated features AND sparsity desired; groups of correlated features |

### Regularization Summary

| Regularizer | Penalty | Prior | Sparsity | Gradient |
|-------------|---------|-------|----------|----------|
| L2 (Ridge) | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ | Gaussian | No | $\lambda \boldsymbol{\beta}$ |
| L1 (Lasso) | $\lambda\|\boldsymbol{\beta}\|_1$ | Laplace | Yes | $\lambda \operatorname{sign}(\boldsymbol{\beta})$ |
| Elastic Net | $\lambda[\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2]$ | Mixture | Partial | $\lambda[\alpha\operatorname{sign}(\boldsymbol{\beta}) + (1-\alpha)\boldsymbol{\beta}]$ |

---

## PyTorch Implementation

```python
"""
Regularized Logistic Regression — Complete Implementation
==========================================================

A production-ready implementation covering:
- L2 regularization via weight_decay
- L1 regularization via manual penalty
- Elastic Net regularization
- Data handling with train/val/test splits
- Training with early stopping
- Comprehensive evaluation metrics
- Regularization strength tuning

Author: Deep Learning Foundations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("REGULARIZED LOGISTIC REGRESSION — COMPLETE PIPELINE")
print("=" * 70)

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================


class BinaryClassificationDataset(Dataset):
    """
    Custom Dataset for binary classification.

    Handles data preprocessing including standardization.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,) with values in {0, 1}
        scaler: Optional pre-fitted StandardScaler
        fit_scaler: Whether to fit the scaler (True for training data)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
) -> Dict:
    """Prepare train/val/test DataLoaders with stratified splits."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp,
    )

    train_dataset = BinaryClassificationDataset(X_train, y_train, fit_scaler=True)
    scaler = train_dataset.get_scaler()
    val_dataset = BinaryClassificationDataset(X_val, y_val, scaler=scaler)
    test_dataset = BinaryClassificationDataset(X_test, y_test, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nData prepared:")
    print(f"  Training:   {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test:       {len(test_dataset):,} samples")
    print(f"  Features:   {X.shape[1]}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "n_features": X.shape[1],
    }


# ============================================================================
# PART 2: MODEL DEFINITION
# ============================================================================


class LogisticRegression(nn.Module):
    """
    Logistic Regression with optional regularization.

    Architecture: Linear → Sigmoid

    For numerically stable training, use BCEWithLogitsLoss
    and call logits() instead of forward().

    Args:
        n_features: Number of input features
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities in [0, 1]."""
        return torch.sigmoid(self.linear(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for BCEWithLogitsLoss)."""
        return self.linear(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions."""
        self.eval()
        with torch.no_grad():
            return (self.forward(x) >= threshold).float()

    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on weights (excluding bias)."""
        return self.linear.weight.abs().sum()

    def l2_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on weights (excluding bias)."""
        return (self.linear.weight ** 2).sum()


# ============================================================================
# PART 3: TRAINING WITH REGULARIZATION
# ============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    patience: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Train logistic regression with optional L1/L2 regularization.

    L2 regularization is applied via optimizer weight_decay.
    L1 regularization is applied as a manual penalty term in the loss.
    For Elastic Net, set both l1_lambda > 0 and l2_lambda > 0.

    Args:
        model: LogisticRegression model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        l1_lambda: L1 regularization strength
        l2_lambda: L2 regularization strength (weight_decay)
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        Training history dictionary
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_lambda
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    if verbose:
        reg_desc = []
        if l2_lambda > 0:
            reg_desc.append(f"L2(λ={l2_lambda})")
        if l1_lambda > 0:
            reg_desc.append(f"L1(λ={l1_lambda})")
        reg_str = " + ".join(reg_desc) if reg_desc else "None"
        print(f"\nRegularization: {reg_str}")
        print(f"Training for up to {num_epochs} epochs (patience={patience})...")
        print("-" * 60)

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            logits = model.logits(batch_X)
            loss = criterion(logits, batch_y)

            # Add L1 penalty manually (weight_decay handles L2)
            if l1_lambda > 0:
                loss = loss + l1_lambda * model.l1_penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_X)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model.logits(batch_X)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item() * len(batch_X)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_X)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history["best_epoch"] = epoch - patience_counter + 1
    return history


# ============================================================================
# PART 4: EVALUATION
# ============================================================================


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """Comprehensive model evaluation."""
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            probs = model(batch_X)
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "auc": roc_auc_score(all_targets, all_probs),
        "confusion_matrix": confusion_matrix(all_targets, all_preds),
    }


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report."""
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted")
    print(f"               0     1")
    print(f"  Actual 0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         1   {cm[1,0]:4d}  {cm[1,1]:4d}")


# ============================================================================
# PART 5: REGULARIZATION COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("REGULARIZATION COMPARISON ON BREAST CANCER DATASET")
print("=" * 70)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
data_dict = prepare_data(X, y, batch_size=32)

configs = [
    {"name": "No Regularization", "l1": 0.0, "l2": 0.0},
    {"name": "L2 (Ridge, λ=0.01)", "l1": 0.0, "l2": 0.01},
    {"name": "L1 (Lasso, λ=0.001)", "l1": 0.001, "l2": 0.0},
    {"name": "Elastic Net", "l1": 0.0005, "l2": 0.005},
]

results = {}

for cfg in configs:
    print(f"\n--- {cfg['name']} ---")
    torch.manual_seed(42)
    model = LogisticRegression(data_dict["n_features"])
    history = train_model(
        model,
        data_dict["train_loader"],
        data_dict["val_loader"],
        num_epochs=200,
        learning_rate=0.01,
        l1_lambda=cfg["l1"],
        l2_lambda=cfg["l2"],
        patience=15,
        verbose=False,
    )
    metrics = evaluate_model(model, data_dict["test_loader"])
    print_evaluation_report(metrics)

    # Store weight statistics
    weights = model.linear.weight.data.numpy().flatten()
    results[cfg["name"]] = {
        "metrics": metrics,
        "history": history,
        "weights": weights,
        "weight_norm": np.linalg.norm(weights),
        "n_nonzero": np.sum(np.abs(weights) > 1e-3),
    }

print(f"\n\nWeight Statistics Comparison:")
print("-" * 70)
print(f"{'Method':<30} {'||β||₂':>8} {'Nonzero':>10} {'AUC':>8}")
print("-" * 70)
for name, r in results.items():
    print(
        f"{name:<30} {r['weight_norm']:>8.4f} "
        f"{r['n_nonzero']:>10d}/{len(r['weights'])} "
        f"{r['metrics']['auc']:>8.4f}"
    )

# ============================================================================
# PART 6: REGULARIZATION STRENGTH TUNING
# ============================================================================

print("\n" + "=" * 70)
print("L2 REGULARIZATION STRENGTH TUNING")
print("=" * 70)

lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
tuning_results = []

for lam in lambdas:
    torch.manual_seed(42)
    model = LogisticRegression(data_dict["n_features"])
    history = train_model(
        model,
        data_dict["train_loader"],
        data_dict["val_loader"],
        num_epochs=200,
        learning_rate=0.01,
        l2_lambda=lam,
        patience=15,
        verbose=False,
    )
    metrics = evaluate_model(model, data_dict["test_loader"])
    weights = model.linear.weight.data.numpy().flatten()
    tuning_results.append({
        "lambda": lam,
        "val_loss": min(history["val_loss"]),
        "test_auc": metrics["auc"],
        "test_acc": metrics["accuracy"],
        "weight_norm": np.linalg.norm(weights),
    })
    print(
        f"λ={lam:<8.4f}: Val Loss={tuning_results[-1]['val_loss']:.4f}, "
        f"Test AUC={metrics['auc']:.4f}, ||β||={tuning_results[-1]['weight_norm']:.4f}"
    )

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Plot 1: Training curves for each regularization type
ax = axes[0, 0]
for name, r in results.items():
    ax.plot(r["history"]["val_loss"], label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss Across Regularization Types")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Weight distributions
ax = axes[0, 1]
for i, (name, r) in enumerate(results.items()):
    ax.hist(r["weights"], bins=20, alpha=0.5, label=name)
ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Weight Value")
ax.set_ylabel("Count")
ax.set_title("Weight Distributions by Regularization")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: λ tuning — AUC and weight norm
ax = axes[1, 0]
lam_vals = [r["lambda"] for r in tuning_results]
aucs = [r["test_auc"] for r in tuning_results]
norms = [r["weight_norm"] for r in tuning_results]

ax.semilogx([max(l, 1e-5) for l in lam_vals], aucs, "b-o", label="Test AUC")
ax.set_xlabel("λ (L2 regularization)")
ax.set_ylabel("Test AUC", color="b")
ax.tick_params(axis="y", labelcolor="b")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.semilogx([max(l, 1e-5) for l in lam_vals], norms, "r--s", label="||β||₂")
ax2.set_ylabel("Weight Norm ||β||₂", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax.set_title("Effect of L2 Regularization Strength")

# Plot 4: Sorted absolute weights for sparsity comparison
ax = axes[1, 1]
for name, r in results.items():
    sorted_abs = np.sort(np.abs(r["weights"]))[::-1]
    ax.plot(sorted_abs, label=name)
ax.axhline(y=1e-3, color="gray", linestyle=":", label="Sparsity threshold")
ax.set_xlabel("Feature Index (sorted by |β|)")
ax.set_ylabel("|β|")
ax.set_title("Weight Magnitude Profiles (Sparsity)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("regularized_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Results visualization saved!")
```

---

## Best Practices

### Data Handling

| Practice | Reason |
|----------|--------|
| Use `DataLoader` | Efficient batching, shuffling, multiprocessing |
| Standardize features | Faster convergence, prevents feature dominance |
| Stratified splits | Preserves class distribution |
| Three-way split | Prevents overfitting to validation set |

### Model Design

| Practice | Reason |
|----------|--------|
| `nn.Module` subclass | Clean, reusable, integrates with PyTorch ecosystem |
| `BCEWithLogitsLoss` | Numerical stability |
| Xavier initialization | Helps with training dynamics |

### Training

| Practice | Reason |
|----------|--------|
| Early stopping | Prevents overfitting |
| `weight_decay` for L2 | Efficient, built into optimizer |
| Manual L1 penalty | PyTorch doesn't have built-in L1 regularization |
| Checkpoint best model | Don't lose best performance |

### Evaluation

| Practice | Reason |
|----------|--------|
| Multiple metrics | Accuracy alone is insufficient for imbalanced data |
| Confusion matrix | Understand error types |
| AUC-ROC | Threshold-independent performance measure |

---

## Exercises

### Mathematical

1. **Derive** the proximal operator for L1 regularization (soft-thresholding) and explain why it produces sparse solutions while L2 does not.

2. **Show** that for the regularized logistic regression loss $\mathcal{L}_{\text{ridge}}$, the Hessian becomes $\mathbf{H}_{\text{ridge}} = \mathbf{X}^\top\mathbf{B}\mathbf{X} + n\lambda\mathbf{I}$. How does regularization affect the condition number?

3. **Prove** that the MLE for regularized logistic regression always exists (even with perfectly separable data) when $\lambda > 0$.

### Computational

4. Implement k-fold cross-validation for selecting the optimal $\lambda$. Plot the validation curve.

5. Add GPU support with `.to(device)` and learning rate scheduling with `ReduceLROnPlateau`.

6. Implement coordinate descent for L1-regularized logistic regression and compare convergence with proximal gradient descent.

---

## Summary

| Concept | Formula | Key Insight |
|---------|---------|-------------|
| L2 penalty | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ | Shrinks all weights; Gaussian prior |
| L1 penalty | $\lambda\|\boldsymbol{\beta}\|_1$ | Produces sparsity; Laplace prior |
| Elastic Net | $\lambda[\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2]$ | Groups + sparsity |
| L2 gradient | $\lambda\boldsymbol{\beta}$ | Weight decay |
| L1 subgradient | $\lambda\operatorname{sign}(\boldsymbol{\beta})$ | Constant push toward zero |
| PyTorch L2 | `weight_decay` in optimizer | Built-in, efficient |
| PyTorch L1 | Manual penalty in loss | Must add explicitly |

Regularization transforms logistic regression from a method that can fail on separable data into a robust, well-posed algorithm with guaranteed convergence. The choice between L1, L2, and Elastic Net depends on the problem structure — whether we expect sparse or dense solutions and how features are correlated.
