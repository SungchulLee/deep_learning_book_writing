"""
==============================================================================
08_regularization.py
==============================================================================
DIFFICULTY: ⭐⭐⭐⭐ (Advanced)

DESCRIPTION:
    L1 and L2 regularization to prevent overfitting.
    Ridge and Lasso regression with PyTorch.

TOPICS COVERED:
    - L1 (Lasso) and L2 (Ridge) regularization
    - Weight decay in optimizers
    - Feature selection with L1
    - Regularization parameter tuning

PREREQUISITES:
    - Tutorial 07 (Polynomial regression)

TIME: ~30 minutes
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("REGULARIZATION: L1 AND L2")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE DATA WITH MANY FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE HIGH-DIMENSIONAL DATA")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# Generate data where only a few features are truly important
n_samples = 100
n_features = 20
n_informative = 5  # Only 5 features actually matter

X = np.random.randn(n_samples, n_features)
true_weights = np.zeros(n_features)
true_weights[:n_informative] = np.array([3, -2, 1.5, -1, 2])  # Important features
y = X @ true_weights + np.random.randn(n_samples) * 0.5

print(f"Dataset:")
print(f"  Samples: {n_samples}")
print(f"  Total features: {n_features}")
print(f"  Informative features: {n_informative}")
print(f"\nTrue important weights (first 5 features):")
print(f"  {true_weights[:n_informative]}")

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# ============================================================================
# PART 2: NO REGULARIZATION (BASELINE)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: BASELINE MODEL (NO REGULARIZATION)")
print("=" * 70)

model_baseline = nn.Linear(n_features, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model_baseline.parameters(), lr=0.01)

n_epochs = 500
for epoch in range(n_epochs):
    y_pred = model_baseline(X_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

baseline_weights = model_baseline.weight.detach().numpy().flatten()
print(f"Baseline model trained")
print(f"  Number of near-zero weights (|w| < 0.1): "
      f"{np.sum(np.abs(baseline_weights) < 0.1)}/{n_features}")

# ============================================================================
# PART 3: L2 REGULARIZATION (RIDGE REGRESSION)
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: L2 REGULARIZATION (RIDGE)")
print("=" * 70)

print("""
L2 Regularization (Ridge):
  Loss = MSE + λ * Σ(w²)
  
  - Penalizes large weights
  - Weights shrink towards zero
  - All features kept (no selection)
  - Implemented via 'weight_decay' in optimizer
""")

# Train with different L2 penalties
l2_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
l2_models = {}

for lambda_val in l2_lambdas:
    model = nn.Linear(n_features, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                                weight_decay=lambda_val)  # ← L2 regularization
    
    for epoch in range(n_epochs):
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    l2_models[lambda_val] = model
    weights = model.weight.detach().numpy().flatten()
    print(f"λ={lambda_val:<6.3f}: Mean |weight|={np.mean(np.abs(weights)):.4f}, "
          f"Max |weight|={np.max(np.abs(weights)):.4f}")

# ============================================================================
# PART 4: L1 REGULARIZATION (LASSO REGRESSION)
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: L1 REGULARIZATION (LASSO)")
print("=" * 70)

print("""
L1 Regularization (Lasso):
  Loss = MSE + λ * Σ(|w|)
  
  - Penalizes absolute values of weights
  - Drives some weights exactly to zero
  - Performs feature selection
  - Must be implemented manually in PyTorch
""")

def train_with_l1(lambda_l1, n_epochs=500):
    """Train model with L1 regularization"""
    model = nn.Linear(n_features, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        y_pred = model(X_tensor)
        mse_loss = criterion(y_pred, y_tensor)
        
        # Add L1 penalty manually
        l1_penalty = lambda_l1 * torch.sum(torch.abs(model.weight))
        total_loss = mse_loss + l1_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return model

# Train with different L1 penalties
l1_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
l1_models = {}

for lambda_val in l1_lambdas:
    model = train_with_l1(lambda_val)
    l1_models[lambda_val] = model
    weights = model.weight.detach().numpy().flatten()
    n_zero = np.sum(np.abs(weights) < 0.01)  # Nearly zero
    print(f"λ={lambda_val:<6.3f}: Nearly zero weights={n_zero}/{n_features}, "
          f"Mean |weight|={np.mean(np.abs(weights)):.4f}")

# ============================================================================
# PART 5: VISUALIZE WEIGHT MAGNITUDES
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: VISUALIZE REGULARIZATION EFFECTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Baseline
ax = axes[0, 0]
ax.bar(range(n_features), baseline_weights)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set_title('No Regularization')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Weight')
ax.grid(True, alpha=0.3)

# L2 regularization
for idx, lambda_val in enumerate([0.001, 0.01, 0.1, 1.0]):
    ax = axes[idx//2, idx%2 + 1] if idx < 2 else axes[1, idx-2]
    if idx < 2:
        # L2
        weights = l2_models[lambda_val].weight.detach().numpy().flatten()
        ax.bar(range(n_features), weights)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.set_title(f'L2 (λ={lambda_val})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

# L1 regularization
for idx, lambda_val in enumerate([0.01, 0.1]):
    ax = axes[1, idx + 1]
    weights = l1_models[lambda_val].weight.detach().numpy().flatten()
    ax.bar(range(n_features), weights)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_title(f'L1 (λ={lambda_val})')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Weight')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/08_regularization_weights.png', dpi=100)
print("Saved weight visualization")

# ============================================================================
# PART 6: COMPARISON PLOT
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot L2 effect
ax = axes[0]
for lambda_val in l2_lambdas:
    weights = l2_models[lambda_val].weight.detach().numpy().flatten()
    ax.plot(sorted(np.abs(weights), reverse=True), 
            marker='o', label=f'λ={lambda_val}')
ax.set_xlabel('Feature (sorted by magnitude)')
ax.set_ylabel('|Weight|')
ax.set_title('L2 Regularization Effect')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot L1 effect
ax = axes[1]
for lambda_val in l1_lambdas:
    weights = l1_models[lambda_val].weight.detach().numpy().flatten()
    ax.plot(sorted(np.abs(weights), reverse=True), 
            marker='o', label=f'λ={lambda_val}')
ax.set_xlabel('Feature (sorted by magnitude)')
ax.set_ylabel('|Weight|')
ax.set_title('L1 Regularization Effect (Feature Selection)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/08_regularization_comparison.png', dpi=100)
print("Saved comparison visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
REGULARIZATION SUMMARY:

L2 Regularization (Ridge):
✓ Shrinks all weights towards zero
✓ Keeps all features
✓ Easy to implement (weight_decay in optimizer)
✓ Good for correlated features
✗ Doesn't perform feature selection

L1 Regularization (Lasso):
✓ Drives some weights exactly to zero
✓ Automatic feature selection
✓ Sparse models (fewer active features)
✗ Requires manual implementation
✗ Unstable with correlated features

When to use:
- L2: When all features might be important
- L1: When you want feature selection
- Both (Elastic Net): Combines advantages

Implementation in PyTorch:
  # L2: Built-in
  optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.01)
  
  # L1: Manual
  l1_loss = lambda_l1 * torch.sum(torch.abs(model.weight))
  total_loss = mse_loss + l1_loss

Next: Tutorial 09 - Mini-batch training with DataLoader!
""")
