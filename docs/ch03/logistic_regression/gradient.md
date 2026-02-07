# Gradient Computation

## Learning Objectives

By the end of this section, you will be able to:

- Derive the gradient of the BCE loss with respect to model parameters
- Understand the elegant cancellation that simplifies the logistic regression gradient
- Derive the Hessian matrix and prove convexity of the loss
- Understand Newton's method and the IRLS algorithm as second-order optimization
- Implement and compare gradient descent, Newton's method, and IRLS

---

## The Optimization Problem

We want to find parameters $\boldsymbol{\beta}$ that minimize the BCE loss:

$$
\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
$$

where $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$ is the linear predictor and $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

---

## Step-by-Step Gradient Derivation

### Step 1: Chain Rule Setup

For a single sample $i$, the loss contribution is:

$$
\ell_i = -y_i \log(p_i) - (1-y_i) \log(1-p_i)
$$

where $p_i = \sigma(z_i)$ and $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$. By the chain rule:

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = \frac{\partial \ell_i}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \boldsymbol{\beta}}
$$

### Step 2: Derivative of Loss w.r.t. Probability

$$
\frac{\partial \ell_i}{\partial p_i} = -\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i} = \frac{-y_i(1-p_i) + (1-y_i)p_i}{p_i(1-p_i)} = \frac{p_i - y_i}{p_i(1-p_i)}
$$

### Step 3: Derivative of Sigmoid

The sigmoid has a beautiful derivative property:

$$
\frac{\partial p_i}{\partial z_i} = \sigma'(z_i) = \sigma(z_i)(1-\sigma(z_i)) = p_i(1-p_i)
$$

### Step 4: Derivative of Linear Predictor

$$
\frac{\partial z_i}{\partial \boldsymbol{\beta}} = \frac{\partial}{\partial \boldsymbol{\beta}}\left(\mathbf{x}_i^\top \boldsymbol{\beta}\right) = \mathbf{x}_i
$$

### Step 5: The Elegant Cancellation

Combining via the chain rule:

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = \frac{p_i - y_i}{p_i(1-p_i)} \cdot p_i(1-p_i) \cdot \mathbf{x}_i
$$

The $p_i(1-p_i)$ terms **cancel**:

$$
\boxed{\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = (p_i - y_i)\mathbf{x}_i = (\sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) - y_i)\mathbf{x}_i}
$$

This cancellation is no accident — it is a consequence of the **canonical link** in the GLM framework. When we use the logit link with a Bernoulli response, the gradient simplifies to this elegant form.

---

## The Complete Gradient

### For the Average Loss

The gradient of the average BCE loss is:

$$
\nabla_{\boldsymbol{\beta}} \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (\sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) - y_i)\mathbf{x}_i
$$

### In Matrix Form

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ be the design matrix (rows are samples), $\mathbf{p} = \sigma(\mathbf{X}\boldsymbol{\beta}) \in \mathbb{R}^n$ the predicted probabilities, and $\mathbf{y} \in \{0,1\}^n$ the true labels. Then:

$$
\boxed{\nabla_{\boldsymbol{\beta}} \mathcal{L} = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})}
$$

This elegant form is the **error-weighted features**: each sample's feature vector $\mathbf{x}_i$ is scaled by the prediction error $(p_i - y_i)$.

### Comparison with Linear Regression

| Model | Gradient (per-sample) |
|-------|----------------------|
| Linear Regression | $(y_i - \hat{y}_i)\mathbf{x}_i$ |
| Logistic Regression | $(\hat{p}_i - y_i)\mathbf{x}_i$ |

The forms are identical except for the sign convention and the use of probability $\hat{p}_i$ instead of continuous prediction $\hat{y}_i$.

---

## Gradient Interpretation

### Error Signal

The term $(p_i - y_i)$ is the **prediction error**:

| Scenario | $y_i$ | $p_i$ | $p_i - y_i$ | Effect on gradient |
|----------|-------|-------|-------------|-------------------|
| Correct, confident | 1 | 0.99 | -0.01 | Small update |
| Correct, uncertain | 1 | 0.6 | -0.4 | Medium update |
| Wrong, confident | 0 | 0.99 | +0.99 | **Large update** |
| Wrong, uncertain | 0 | 0.6 | +0.6 | Medium update |

### Gradient Properties

1. **Bounded errors**: Since $p \in (0, 1)$ and $y \in \{0, 1\}$, errors are bounded: $|p - y| < 1$
2. **Feature scaling matters**: Large features → large gradients → potentially unstable training
3. **Vanishing gradients near optimum**: When $p \approx y$, gradient is small

---

## Gradient Descent Update Rule

### Standard (Batch) Gradient Descent

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \frac{\eta}{n}\mathbf{X}^\top(\mathbf{p}^{(t)} - \mathbf{y})
$$

### Stochastic Gradient Descent

For a single sample $i$:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta (p_i^{(t)} - y_i)\mathbf{x}_i
$$

### Mini-Batch SGD

For a batch $\mathcal{B}$ of size $B$:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \frac{\eta}{B}\sum_{i \in \mathcal{B}} (p_i^{(t)} - y_i)\mathbf{x}_i
$$

---

## Hessian Derivation

### Per-Sample Second Derivative

The gradient contribution from sample $i$ is $(p_i - y_i)\mathbf{x}_i$. Since $y_i$ is constant, we differentiate only the $p_i$ term:

$$
\frac{\partial^2 \mathcal{L}}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^{\top}} \bigg|_{\text{sample } i} = \mathbf{x}_i \frac{\partial p_i}{\partial \boldsymbol{\beta}^{\top}} = p_i(1-p_i) \, \mathbf{x}_i \mathbf{x}_i^{\top}
$$

### Full Hessian

Summing over all samples (for the unaveraged loss):

$$
\mathbf{H} = \nabla^2_{\boldsymbol{\beta}} \mathcal{L} = \sum_{i=1}^{n} p_i(1-p_i) \, \mathbf{x}_i \mathbf{x}_i^{\top}
$$

Define the diagonal weight matrix $\mathbf{B} = \operatorname{diag}(p_1(1-p_1), \ldots, p_n(1-p_n))$. Then:

$$
\boxed{\mathbf{H} = \mathbf{X}^{\top} \mathbf{B} \mathbf{X}}
$$

### Positive Semi-Definiteness and Convexity

For any vector $\mathbf{v} \in \mathbb{R}^d$:

$$
\mathbf{v}^{\top} \mathbf{H} \mathbf{v} = \mathbf{v}^{\top} \mathbf{X}^{\top} \mathbf{B} \mathbf{X} \mathbf{v} = (\mathbf{X}\mathbf{v})^{\top} \mathbf{B} (\mathbf{X}\mathbf{v}) = \sum_{i=1}^{n} p_i(1-p_i)(\mathbf{x}_i^{\top}\mathbf{v})^2
$$

Since $p_i \in (0, 1)$ we have $p_i(1-p_i) > 0$, and $(\mathbf{x}_i^{\top}\mathbf{v})^2 \geq 0$, so every term is non-negative:

$$
\mathbf{v}^{\top} \mathbf{H} \mathbf{v} \geq 0 \quad \forall \, \mathbf{v}
$$

Therefore $\mathbf{H}$ is **positive semi-definite**, which means the negative log-likelihood is **convex**. If $\mathbf{X}$ has full column rank, the Hessian is strictly positive definite and the loss is strictly convex, guaranteeing a unique global minimum.

---

## Newton's Method

### Update Rule

Newton's method uses the Hessian to take curvature-informed steps:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1} \mathbf{g}
$$

Substituting the gradient and Hessian:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - (\mathbf{X}^{\top}\mathbf{B}\mathbf{X})^{-1} \mathbf{X}^{\top}(\mathbf{p} - \mathbf{y})
$$

### Comparison with Gradient Descent

| Property | Gradient Descent | Newton's Method |
|----------|-----------------|-----------------|
| Update | $\boldsymbol{\beta} - \eta \mathbf{g}$ | $\boldsymbol{\beta} - \mathbf{H}^{-1}\mathbf{g}$ |
| Convergence rate | Linear | Quadratic (near optimum) |
| Per-step cost | $O(nd)$ | $O(nd^2 + d^3)$ |
| Hyperparameters | Learning rate $\eta$ | None (or damping factor) |
| Memory | $O(d)$ | $O(d^2)$ for Hessian |

Newton's method converges in far fewer iterations but each iteration is more expensive due to the Hessian computation and matrix inversion.

---

## Iteratively Reweighted Least Squares (IRLS)

### Derivation from Newton's Method

Starting from Newton's update and rearranging into normal-equations form. Multiply both sides by $(\mathbf{X}^{\top}\mathbf{B}\mathbf{X})$:

$$
(\mathbf{X}^{\top}\mathbf{B}\mathbf{X})\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{B}\mathbf{X})\boldsymbol{\beta}^{(t)} - \mathbf{X}^{\top}(\mathbf{p} - \mathbf{y})
$$

Factor $\mathbf{X}^{\top}$ from the right-hand side:

$$
(\mathbf{X}^{\top}\mathbf{B}\mathbf{X})\boldsymbol{\beta}^{(t+1)} = \mathbf{X}^{\top}\bigl[\mathbf{B}\mathbf{X}\boldsymbol{\beta}^{(t)} - (\mathbf{p} - \mathbf{y})\bigr]
$$

Since $\mathbf{B}$ is invertible, write $(\mathbf{p} - \mathbf{y}) = \mathbf{B}\mathbf{B}^{-1}(\mathbf{p} - \mathbf{y})$:

$$
(\mathbf{X}^{\top}\mathbf{B}\mathbf{X})\boldsymbol{\beta}^{(t+1)} = \mathbf{X}^{\top}\mathbf{B}\bigl[\mathbf{X}\boldsymbol{\beta}^{(t)} - \mathbf{B}^{-1}(\mathbf{p} - \mathbf{y})\bigr]
$$

### The Working Response

Define the **working response** (or adjusted dependent variable):

$$
\mathbf{z} = \mathbf{X}\boldsymbol{\beta}^{(t)} - \mathbf{B}^{-1}(\mathbf{p} - \mathbf{y})
$$

Then the update becomes:

$$
\boxed{\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{B}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{B}\mathbf{z}}
$$

### Connection to Weighted Least Squares

This is exactly the **normal equation** for the weighted least squares problem:

$$
\min_{\boldsymbol{\beta}} \; (\mathbf{z} - \mathbf{X}\boldsymbol{\beta})^{\top} \mathbf{B} (\mathbf{z} - \mathbf{X}\boldsymbol{\beta})
$$

At each iteration, we solve a weighted least squares problem where:

- The **response** $\mathbf{z}$ is the linearized version of the nonlinear model
- The **weights** $\mathbf{B}$ reflect the variance of each observation under the current parameters
- Both $\mathbf{z}$ and $\mathbf{B}$ depend on $\boldsymbol{\beta}^{(t)}$ and must be recomputed each iteration

Because the weight matrix $\mathbf{B}$ changes at every step, this procedure is called **Iteratively Reweighted Least Squares (IRLS)**.

### IRLS Algorithm

1. Initialize $\boldsymbol{\beta}^{(0)}$ (e.g., zeros)
2. **Repeat** until convergence:
    - Compute predictions: $\mathbf{p} = \sigma(\mathbf{X}\boldsymbol{\beta}^{(t)})$
    - Compute weights: $\mathbf{B} = \operatorname{diag}(p_i(1-p_i))$
    - Compute working response: $\mathbf{z} = \mathbf{X}\boldsymbol{\beta}^{(t)} - \mathbf{B}^{-1}(\mathbf{p} - \mathbf{y})$
    - Solve: $\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{B}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{B}\mathbf{z}$
3. Return $\boldsymbol{\beta}^{(t+1)}$

### Working Response Interpretation

The working response for observation $i$ is:

$$
z_i = \mathbf{x}_i^{\top}\boldsymbol{\beta}^{(t)} - \frac{p_i - y_i}{p_i(1-p_i)}
$$

This is the current linear predictor adjusted by a linearized correction term. Note that $p_i(1-p_i) = \sigma'(\mathbf{x}_i^{\top}\boldsymbol{\beta}^{(t)})$, so the adjustment is the residual divided by the derivative of the link function — precisely the first-order Taylor expansion of the link function applied to the response.

### GLM Perspective

IRLS is not specific to logistic regression. It applies to the entire family of **generalized linear models (GLMs)**. For any GLM with canonical link:

$$
\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{W}\mathbf{z}
$$

where $\mathbf{W}$ and $\mathbf{z}$ depend on the specific distribution and link function.

---

## PyTorch Implementation

```python
"""
Gradient, Hessian, and IRLS for Logistic Regression
=====================================================

Demonstrates:
- Manual gradient computation and verification against autograd
- Hessian computation and PSD verification
- Newton's method with quadratic convergence
- IRLS algorithm and equivalence to Newton's method
- Convergence comparison across all three methods

Author: Deep Learning Foundations
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("GRADIENT, HESSIAN, AND IRLS FOR LOGISTIC REGRESSION")
print("=" * 70)

# ============================================================================
# Part 1: Core Functions
# ============================================================================


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def compute_loss(X, y, beta):
    p = sigmoid(X @ beta)
    eps = 1e-12
    return -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()


def compute_gradient(X, y, beta):
    """g = (1/n) X^T (p - y)"""
    p = sigmoid(X @ beta)
    return (1.0 / len(y)) * X.T @ (p - y)


def compute_hessian(X, beta):
    """H = (1/n) X^T B X where B = diag(p_i(1-p_i))"""
    p = sigmoid(X @ beta)
    b = p * (1 - p)  # (n, 1)
    return (1.0 / len(p)) * (X * b).T @ X


# ============================================================================
# Part 2: Verify Gradient via Autograd
# ============================================================================

print("\n1. Verifying Gradient Computation")
print("-" * 50)

n_samples, n_features = 100, 5
X = torch.randn(n_samples, n_features)
y = torch.randint(0, 2, (n_samples, 1)).float()
beta = torch.randn(n_features, 1, requires_grad=True)

# Manual gradient
with torch.no_grad():
    manual_grad = compute_gradient(X, y, beta)

# Autograd gradient
loss = compute_loss(X, y, beta)
loss.backward()
autograd_grad = beta.grad

print(f"Manual gradient (first 3):    {manual_grad[:3].flatten().tolist()}")
print(f"Autograd gradient (first 3):  {autograd_grad[:3].flatten().tolist()}")
print(f"Max difference: {(manual_grad - autograd_grad).abs().max().item():.2e}")
print(f"Gradients match: {torch.allclose(manual_grad, autograd_grad, atol=1e-6)}")

# ============================================================================
# Part 3: Gradient Components Breakdown
# ============================================================================

print("\n" + "=" * 70)
print("GRADIENT COMPONENTS BREAKDOWN")
print("=" * 70)

X_small = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y_small = torch.tensor([[1.0], [0.0], [1.0]])
beta_small = torch.tensor([[0.5], [-0.3]])

z_small = X_small @ beta_small
p_small = sigmoid(z_small)
error_small = p_small - y_small

print("\nStep-by-step gradient calculation:")
print("-" * 50)
print(f"Linear predictor z = Xβ:  {z_small.T.tolist()}")
print(f"Predictions p = σ(z):     {[f'{v:.3f}' for v in p_small.flatten().tolist()]}")
print(f"True labels y:            {y_small.T.tolist()}")
print(f"Errors (p - y):           {[f'{v:.3f}' for v in error_small.flatten().tolist()]}")

print("\nPer-sample gradient contributions:")
for i in range(len(X_small)):
    contrib = error_small[i] * X_small[i : i + 1].T
    print(
        f"  Sample {i+1}: error={error_small[i].item():.3f} "
        f"× features={X_small[i].tolist()} = {[f'{v:.3f}' for v in contrib.flatten().tolist()]}"
    )

gradient_small = (1 / 3) * X_small.T @ error_small
print(f"\nTotal gradient (averaged): {[f'{v:.3f}' for v in gradient_small.flatten().tolist()]}")

# ============================================================================
# Part 4: Hessian and Convexity Verification
# ============================================================================

print("\n" + "=" * 70)
print("HESSIAN AND CONVEXITY")
print("=" * 70)

X_raw, y_raw = make_classification(
    n_samples=200, n_features=5, n_informative=4, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Add bias column
X_train = torch.cat([torch.ones(len(X_train), 1), X_train], dim=1)
X_test = torch.cat([torch.ones(len(X_test), 1), X_test], dim=1)

n, d = X_train.shape
print(f"Training data: n={n}, d={d}")

beta_init = torch.zeros(d, 1)
H = compute_hessian(X_train, beta_init)
eigenvalues = torch.linalg.eigvalsh(H)

print(f"Hessian eigenvalues: {eigenvalues.numpy().round(6)}")
print(f"All non-negative: {(eigenvalues >= -1e-10).all().item()}")
print(f"Smallest eigenvalue: {eigenvalues.min().item():.6f}")
print("=> Loss is convex (PSD Hessian confirmed)")

# ============================================================================
# Part 5: Convergence Comparison — GD vs Newton vs IRLS
# ============================================================================

print("\n" + "=" * 70)
print("CONVERGENCE COMPARISON")
print("=" * 70)

# --- Gradient Descent ---
beta_gd = torch.zeros(d, 1)
lr = 1.0
gd_losses = []

for epoch in range(50):
    loss = compute_loss(X_train, y_train, beta_gd)
    gd_losses.append(loss.item())
    g = compute_gradient(X_train, y_train, beta_gd)
    beta_gd = beta_gd - lr * g

# --- Newton's Method ---
beta_newton = torch.zeros(d, 1)
newton_losses = []

for epoch in range(10):
    loss = compute_loss(X_train, y_train, beta_newton)
    newton_losses.append(loss.item())
    g = compute_gradient(X_train, y_train, beta_newton)
    H = compute_hessian(X_train, beta_newton)
    beta_newton = beta_newton - torch.linalg.solve(H, g)

# --- IRLS ---
beta_irls = torch.zeros(d, 1)
irls_losses = []

for epoch in range(10):
    loss = compute_loss(X_train, y_train, beta_irls)
    irls_losses.append(loss.item())
    p = sigmoid(X_train @ beta_irls)
    B_diag = p * (1 - p)
    z = X_train @ beta_irls - (p - y_train) / B_diag  # working response
    XtBX = (X_train * B_diag).T @ X_train
    XtBz = (X_train * B_diag).T @ z
    beta_irls = torch.linalg.solve(XtBX, XtBz)

print(f"GD final loss (50 iters):     {gd_losses[-1]:.6f}")
print(f"Newton final loss (10 iters): {newton_losses[-1]:.6f}")
print(f"IRLS final loss (10 iters):   {irls_losses[-1]:.6f}")

# Verify Newton == IRLS
print(f"\nNewton ≈ IRLS: {torch.allclose(beta_newton, beta_irls, atol=1e-5)}")
print(f"Max difference: {(beta_newton - beta_irls).abs().max().item():.2e}")

# ============================================================================
# Part 6: Visualization
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Convergence comparison
ax = axes[0]
ax.plot(gd_losses, "b-o", ms=3, label=f"Gradient Descent ({len(gd_losses)} iters)")
ax.plot(newton_losses, "r-s", ms=5, label=f"Newton ({len(newton_losses)} iters)")
ax.plot(irls_losses, "g--^", ms=5, label=f"IRLS ({len(irls_losses)} iters)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Convergence Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Log-scale convergence (quadratic vs linear)
ax = axes[1]
loss_star = min(newton_losses[-1], gd_losses[-1])
gd_gap = [abs(l - loss_star) + 1e-16 for l in gd_losses]
newton_gap = [abs(l - loss_star) + 1e-16 for l in newton_losses]
ax.semilogy(gd_gap, "b-o", ms=3, label="GD (linear)")
ax.semilogy(newton_gap, "r-s", ms=5, label="Newton (quadratic)")
ax.set_xlabel("Iteration")
ax.set_ylabel("$|\\mathcal{L} - \\mathcal{L}^*|$")
ax.set_title("Convergence Rate (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Gradient norm during GD training
beta_gd2 = torch.zeros(d, 1)
grad_norms = []
for epoch in range(50):
    g = compute_gradient(X_train, y_train, beta_gd2)
    grad_norms.append(torch.norm(g).item())
    beta_gd2 = beta_gd2 - lr * g

ax = axes[2]
ax.plot(grad_norms, "g-", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("||∇L||")
ax.set_title("Gradient Norm During Training")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_hessian_irls.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Visualization saved!")
```

---

## Exercises

### Mathematical

1. **Show** that when the data is linearly separable, the MLE does not exist (parameters diverge to infinity) and explain how this manifests in the Hessian becoming ill-conditioned.

2. **Derive** the IRLS update for Poisson regression with the canonical log link and identify the corresponding weight matrix $\mathbf{B}$.

3. **Analyze** how the gradient magnitude changes as predictions become more confident, and relate this to the vanishing gradient problem in deep networks.

### Computational

4. Implement a damped Newton's method with line search and compare its robustness to pure Newton's method on ill-conditioned problems.

5. Create a visualization showing the optimization trajectory of gradient descent vs Newton's method on the log-likelihood surface.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Per-sample gradient | $(p_i - y_i)\mathbf{x}_i$ |
| Batch gradient | $\frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$ |
| Key cancellation | $\frac{p-y}{p(1-p)} \cdot p(1-p) = p - y$ |
| Hessian | $\mathbf{H} = \mathbf{X}^{\top}\mathbf{B}\mathbf{X}$ |
| Weight matrix | $\mathbf{B} = \operatorname{diag}(p_i(1-p_i))$ |
| Newton update | $\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1}\mathbf{g}$ |
| IRLS update | $\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{B}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{B}\mathbf{z}$ |
| Working response | $\mathbf{z} = \mathbf{X}\boldsymbol{\beta}^{(t)} - \mathbf{B}^{-1}(\mathbf{p} - \mathbf{y})$ |

The elegant gradient formula $(\sigma(\mathbf{x}^\top\boldsymbol{\beta}) - y)\mathbf{x}$ enables efficient first-order optimization, while the Hessian $\mathbf{X}^\top\mathbf{B}\mathbf{X}$ provides both a proof of convexity and the foundation for second-order methods (Newton/IRLS) with quadratic convergence.

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 4.3.3
2. McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
3. Green, P. J. (1984). Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives. *JRSS-B*, 46(2), 149–192.
