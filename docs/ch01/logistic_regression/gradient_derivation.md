# Gradient Derivation for Logistic Regression

## Learning Objectives

By the end of this section, you will be able to:

- Derive the gradient of the BCE loss with respect to model parameters
- Understand the elegant form of the logistic regression gradient
- Connect gradients to the update rules in gradient descent
- Implement manual gradient computation and verify against autograd

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

For a single sample $i$, the loss is:

$$
\ell_i = -y_i \log(p_i) - (1-y_i) \log(1-p_i)
$$

where $p_i = \sigma(z_i)$ and $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$.

Using the chain rule:

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = \frac{\partial \ell_i}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \boldsymbol{\beta}}
$$

### Step 2: Derivative of Loss w.r.t. Probability

$$
\frac{\partial \ell_i}{\partial p_i} = \frac{\partial}{\partial p_i}\left[-y_i \log(p_i) - (1-y_i) \log(1-p_i)\right]
$$

$$
= -\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}
$$

$$
= \frac{-y_i(1-p_i) + (1-y_i)p_i}{p_i(1-p_i)}
$$

$$
= \frac{-y_i + y_i p_i + p_i - y_i p_i}{p_i(1-p_i)}
$$

$$
= \frac{p_i - y_i}{p_i(1-p_i)}
$$

### Step 3: Derivative of Sigmoid

The sigmoid has a beautiful derivative property:

$$
\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z)) = p(1-p)
$$

**Proof:**
$$
\sigma(z) = (1+e^{-z})^{-1}
$$

Using the power rule:
$$
\sigma'(z) = -1 \cdot (1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}
$$

Recognizing that $\sigma(z) = \frac{1}{1+e^{-z}}$ and $1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}$:

$$
\sigma'(z) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))
$$

So: $\frac{\partial p_i}{\partial z_i} = p_i(1-p_i)$

### Step 4: Derivative of Linear Predictor

$$
\frac{\partial z_i}{\partial \boldsymbol{\beta}} = \frac{\partial}{\partial \boldsymbol{\beta}}\left(\mathbf{x}_i^\top \boldsymbol{\beta}\right) = \mathbf{x}_i
$$

### Step 5: Combine Using Chain Rule

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = \frac{p_i - y_i}{p_i(1-p_i)} \cdot p_i(1-p_i) \cdot \mathbf{x}_i
$$

The $p_i(1-p_i)$ terms **cancel**!

$$
\boxed{\frac{\partial \ell_i}{\partial \boldsymbol{\beta}} = (p_i - y_i)\mathbf{x}_i = (\sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) - y_i)\mathbf{x}_i}
$$

---

## The Complete Gradient

### For the Average Loss

The gradient of the average BCE loss is:

$$
\nabla_{\boldsymbol{\beta}} \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (\sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) - y_i)\mathbf{x}_i
$$

### In Matrix Form

Let:
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ be the design matrix (rows are samples)
- $\mathbf{p} = \sigma(\mathbf{X}\boldsymbol{\beta}) \in \mathbb{R}^n$ be predicted probabilities
- $\mathbf{y} \in \{0,1\}^n$ be true labels

Then:
$$
\boxed{\nabla_{\boldsymbol{\beta}} \mathcal{L} = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})}
$$

This elegant form is the **error-weighted features**: each sample's feature vector $\mathbf{x}_i$ is scaled by the prediction error $(p_i - y_i)$.

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

### Feature Weighting

The gradient for parameter $\beta_j$ is:

$$
\frac{\partial \mathcal{L}}{\partial \beta_j} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)x_{ij}
$$

Features that are:
- **Large** when the model makes errors → Large gradient → Quick correction
- **Small** or **zero** → Small gradient → Slower/no update

---

## Gradient Descent Update Rule

### Standard Gradient Descent

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla_{\boldsymbol{\beta}} \mathcal{L}
$$

Expanding:
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

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("GRADIENT DERIVATION FOR LOGISTIC REGRESSION")
print("="*70)

# ============================================================================
# Part 1: Manual Gradient Computation
# ============================================================================

def compute_gradient_manual(X, y, beta):
    """
    Compute gradient manually using derived formula.
    
    ∇L = (1/n) X^T (σ(Xβ) - y)
    
    Args:
        X: Feature matrix (n, d)
        y: Labels (n, 1)
        beta: Parameters (d, 1)
        
    Returns:
        Gradient (d, 1)
    """
    n = X.shape[0]
    z = X @ beta                    # Linear predictor: (n, 1)
    p = torch.sigmoid(z)            # Predictions: (n, 1)
    error = p - y                   # Prediction error: (n, 1)
    gradient = (1/n) * X.T @ error  # Gradient: (d, 1)
    return gradient


def compute_loss_manual(X, y, beta):
    """Compute BCE loss manually."""
    z = X @ beta
    p = torch.sigmoid(z)
    eps = 1e-12
    loss = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps))
    return loss.mean()


# Create simple test case
print("\n1. Verifying Gradient Computation")
print("-" * 50)

n_samples, n_features = 100, 5
X = torch.randn(n_samples, n_features)
y = torch.randint(0, 2, (n_samples, 1)).float()
beta = torch.randn(n_features, 1, requires_grad=True)

# Manual gradient
with torch.no_grad():
    manual_grad = compute_gradient_manual(X, y, beta)

# Autograd gradient
loss = compute_loss_manual(X, y, beta)
loss.backward()
autograd_grad = beta.grad

print(f"Manual gradient (first 3):    {manual_grad[:3].flatten().tolist()}")
print(f"Autograd gradient (first 3):  {autograd_grad[:3].flatten().tolist()}")
print(f"Max difference: {(manual_grad - autograd_grad).abs().max().item():.2e}")
print(f"Gradients match: {torch.allclose(manual_grad, autograd_grad, atol=1e-6)}")

# ============================================================================
# Part 2: Gradient Components Breakdown
# ============================================================================

print("\n" + "="*70)
print("GRADIENT COMPONENTS BREAKDOWN")
print("="*70)

# Create a small example for detailed analysis
X_small = torch.tensor([
    [1.0, 2.0],   # Sample 1
    [3.0, 4.0],   # Sample 2
    [5.0, 6.0],   # Sample 3
])
y_small = torch.tensor([[1.0], [0.0], [1.0]])
beta_small = torch.tensor([[0.5], [-0.3]])

# Step-by-step computation
z_small = X_small @ beta_small
p_small = torch.sigmoid(z_small)
error_small = p_small - y_small

print("\nStep-by-step gradient calculation:")
print("-" * 50)
print(f"Features X:\n{X_small}")
print(f"\nWeights β: {beta_small.T}")
print(f"\nLinear predictor z = Xβ:\n{z_small.T}")
print(f"\nPredictions p = σ(z):\n{p_small.T}")
print(f"\nTrue labels y:\n{y_small.T}")
print(f"\nErrors (p - y):\n{error_small.T}")

# Per-sample contribution to gradient
print("\nPer-sample gradient contributions:")
print("-" * 50)
for i in range(len(X_small)):
    contrib = error_small[i] * X_small[i:i+1].T
    print(f"Sample {i+1}: error={error_small[i].item():.3f} × features={X_small[i].tolist()} = {contrib.flatten().tolist()}")

# Total gradient
gradient_small = (1/3) * X_small.T @ error_small
print(f"\nTotal gradient (averaged): {gradient_small.flatten().tolist()}")

# ============================================================================
# Part 3: Manual Training Loop
# ============================================================================

print("\n" + "="*70)
print("MANUAL GRADIENT DESCENT IMPLEMENTATION")
print("="*70)

# Generate dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Add bias column
X_train_bias = torch.cat([torch.ones(len(X_train), 1), X_train], dim=1)
X_test_bias = torch.cat([torch.ones(len(X_test), 1), X_test], dim=1)

# Initialize parameters
n_features_with_bias = X_train_bias.shape[1]
beta = torch.zeros(n_features_with_bias, 1)

# Training hyperparameters
learning_rate = 0.5
num_epochs = 100

# Training history
history = {
    'loss': [],
    'accuracy': [],
    'gradient_norm': []
}

print(f"\nTraining with manual gradient descent:")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {num_epochs}")
print("-" * 50)

for epoch in range(num_epochs):
    # Forward pass
    z = X_train_bias @ beta
    p = torch.sigmoid(z)
    
    # Compute loss
    loss = compute_loss_manual(X_train_bias, y_train, beta)
    
    # Compute gradient manually
    gradient = compute_gradient_manual(X_train_bias, y_train, beta)
    gradient_norm = torch.norm(gradient).item()
    
    # Update parameters
    beta = beta - learning_rate * gradient
    
    # Compute accuracy
    predictions = (p >= 0.5).float()
    accuracy = (predictions == y_train).float().mean().item()
    
    # Store history
    history['loss'].append(loss.item())
    history['accuracy'].append(accuracy)
    history['gradient_norm'].append(gradient_norm)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
              f"Acc={accuracy:.4f}, ||∇||={gradient_norm:.4f}")

# Test accuracy
with torch.no_grad():
    test_probs = torch.sigmoid(X_test_bias @ beta)
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds == y_test).float().mean().item()

print(f"\nFinal test accuracy: {test_acc:.4f}")

# ============================================================================
# Part 4: Comparison with PyTorch Autograd
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: Manual vs PyTorch Autograd")
print("="*70)

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Train with PyTorch
torch.manual_seed(42)
model = LogisticRegression(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

pytorch_history = {'loss': [], 'accuracy': []}

for epoch in range(num_epochs):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    accuracy = ((predictions >= 0.5).float() == y_train).float().mean().item()
    pytorch_history['loss'].append(loss.item())
    pytorch_history['accuracy'].append(accuracy)

print(f"\nFinal comparison:")
print(f"  Manual:  Loss={history['loss'][-1]:.4f}, Accuracy={history['accuracy'][-1]:.4f}")
print(f"  PyTorch: Loss={pytorch_history['loss'][-1]:.4f}, Accuracy={pytorch_history['accuracy'][-1]:.4f}")

# ============================================================================
# Part 5: Visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Loss comparison
ax1 = axes[0, 0]
ax1.plot(history['loss'], 'b-', linewidth=2, label='Manual GD')
ax1.plot(pytorch_history['loss'], 'r--', linewidth=2, alpha=0.7, label='PyTorch')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('BCE Loss', fontsize=11)
ax1.set_title('Loss: Manual vs PyTorch', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Gradient norm over training
ax2 = axes[0, 1]
ax2.plot(history['gradient_norm'], 'g-', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('||∇L||', fontsize=11)
ax2.set_title('Gradient Norm During Training', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax3 = axes[1, 0]
with torch.no_grad():
    final_p = torch.sigmoid(X_train_bias @ beta)
    errors = (final_p - y_train).numpy().flatten()

ax3.hist(errors[y_train.numpy().flatten() == 0], bins=30, alpha=0.6, 
        label='y=0', color='blue')
ax3.hist(errors[y_train.numpy().flatten() == 1], bins=30, alpha=0.6, 
        label='y=1', color='red')
ax3.axvline(x=0, color='black', linestyle='--')
ax3.set_xlabel('Prediction Error (p - y)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Error Distribution by True Class', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Gradient formula visualization
ax4 = axes[1, 1]
# Show gradient formula
formula_text = r"""
Gradient Derivation Summary:

$\frac{\partial \ell}{\partial \beta} = \frac{\partial \ell}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial \beta}$

$= \frac{p - y}{p(1-p)} \cdot p(1-p) \cdot x$

$= (p - y) \cdot x$

Final form (matrix):
$\nabla_\beta \mathcal{L} = \frac{1}{n} X^T (\sigma(X\beta) - y)$

Key insight: The $p(1-p)$ terms cancel!
"""
ax4.text(0.1, 0.5, formula_text, fontsize=11, family='serif',
        verticalalignment='center', transform=ax4.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.axis('off')
ax4.set_title('Gradient Formula', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('gradient_derivation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved!")
```

---

## Key Insights

### Why the Gradient is Elegant

The cancellation of $p(1-p)$ terms is no accident—it's a consequence of the **canonical link** in the GLM framework. When we use the logit link with a Bernoulli response, the gradient simplifies to this elegant form.

### Comparison with Linear Regression

| Model | Gradient (per-sample) |
|-------|----------------------|
| Linear Regression | $(y_i - \hat{y}_i)\mathbf{x}_i$ |
| Logistic Regression | $(\hat{p}_i - y_i)\mathbf{x}_i$ |

The forms are identical except for the sign convention and the use of probability $\hat{p}_i$ instead of continuous prediction $\hat{y}_i$.

### Gradient Properties

1. **Bounded errors**: Since $p \in (0, 1)$ and $y \in \{0, 1\}$, errors are bounded: $|p - y| < 1$
2. **Feature scaling matters**: Large features → large gradients → potentially unstable training
3. **Vanishing gradients**: When $p \approx y$, gradient is small (good! we're near the optimum)

---

## Exercises

1. **Derive** the Hessian matrix for logistic regression and show it's negative semi-definite (proving convexity of the loss).

2. **Implement** Newton-Raphson optimization using the Hessian and compare convergence with gradient descent.

3. **Analyze** how the gradient magnitude changes as predictions become more confident.

---

## Summary

| Component | Formula |
|-----------|---------|
| Per-sample gradient | $(p_i - y_i)\mathbf{x}_i$ |
| Batch gradient | $\frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$ |
| Sigmoid derivative | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ |
| Key cancellation | $\frac{p-y}{p(1-p)} \cdot p(1-p) = p - y$ |

The elegant gradient formula $(\sigma(\mathbf{x}^\top\boldsymbol{\beta}) - y)\mathbf{x}$ enables efficient training of logistic regression models and forms the foundation for understanding backpropagation in deeper networks.
