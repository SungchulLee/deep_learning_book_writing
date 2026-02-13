"""
==============================================================================
06_multivariate_regression.py
==============================================================================
DIFFICULTY: ⭐⭐⭐ (Intermediate)

DESCRIPTION:
    Linear regression with multiple input features (multivariate).
    Uses California housing dataset for real-world example.

TOPICS COVERED:
    - Multiple input features
    - Real-world dataset
    - Feature scaling/normalization
    - Train/test split
    - Model evaluation metrics

PREREQUISITES:
    - Tutorial 05 (nn.Module)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 70)

# ============================================================================
# PART 1: LOAD REAL-WORLD DATASET
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: LOAD CALIFORNIA HOUSING DATASET")
print("=" * 70)

# Load dataset
housing = fetch_california_housing()
X_numpy = housing.data
y_numpy = housing.target

print(f"Dataset loaded:")
print(f"  Samples: {X_numpy.shape[0]}")
print(f"  Features: {X_numpy.shape[1]}")
print(f"\nFeature names:")
for i, name in enumerate(housing.feature_names):
    print(f"  {i}: {name}")

print(f"\nTarget: Median house value ($100k)")
print(f"  Min: ${y_numpy.min()*100:.1f}k")
print(f"  Max: ${y_numpy.max()*100:.1f}k")
print(f"  Mean: ${y_numpy.mean()*100:.1f}k")

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DATA PREPROCESSING")
print("=" * 70)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42
)

print(f"Data split:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# Feature scaling (standardization)
# Important: Fit scaler on training data only!
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\nFeature scaling applied (StandardScaler)")
print(f"  Train X: mean≈0, std≈1")
print(f"  Train y: mean≈0, std≈1")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_t.shape}")
print(f"  y_train: {y_train_t.shape}")

# ============================================================================
# PART 3: DEFINE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE MULTIVARIATE MODEL")
print("=" * 70)

class MultiLinearRegression(nn.Module):
    def __init__(self, n_features):
        super(MultiLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x)

n_features = X_train_t.shape[1]
model = MultiLinearRegression(n_features)

print(f"Model created with {n_features} input features")
print(model)
print(f"\nParameter shapes:")
print(f"  Weight: {model.linear.weight.shape}")
print(f"  Bias: {model.linear.bias.shape}")

# ============================================================================
# PART 4: TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING")
print("=" * 70)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

n_epochs = 200
train_losses = []
test_losses = []

print(f"Training for {n_epochs} epochs with Adam optimizer...")
print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15}")
print("-" * 45)

for epoch in range(n_epochs):
    # Training
    model.train()
    y_pred_train = model(X_train_t)
    loss_train = criterion(y_pred_train, y_train_t)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        loss_test = criterion(y_pred_test, y_test_t)
    
    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss_train.item():<15.6f} {loss_test.item():<15.6f}")

print(f"\nTraining completed!")

# ============================================================================
# PART 5: EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: EVALUATION")
print("=" * 70)

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_t)
    y_pred_test = model(X_test_t)

# Convert back to original scale
y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.numpy())
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.numpy())

# Calculate R² score
from sklearn.metrics import r2_score, mean_absolute_error

r2_train = r2_score(y_train, y_pred_train_orig)
r2_test = r2_score(y_test, y_pred_test_orig)
mae_train = mean_absolute_error(y_train, y_pred_train_orig)
mae_test = mean_absolute_error(y_test, y_pred_test_orig)

print(f"Model Performance:")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²: {r2_test:.4f}")
print(f"  Train MAE: ${mae_train*100:.2f}k")
print(f"  Test MAE: ${mae_test*100:.2f}k")

print(f"\nFeature Importance (absolute weights):")
weights = model.linear.weight.detach().numpy().flatten()
for i, (name, weight) in enumerate(zip(housing.feature_names, weights)):
    print(f"  {name:20s}: {abs(weight):8.4f}")

# ============================================================================
# PART 6: VISUALIZE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training History')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Predictions vs Actual (Test Set)
axes[0, 1].scatter(y_test, y_pred_test_orig, alpha=0.5, s=10)
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price ($100k)')
axes[0, 1].set_ylabel('Predicted Price ($100k)')
axes[0, 1].set_title(f'Predictions vs Actual (Test R²={r2_test:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred_test_orig.flatten()
axes[1, 0].scatter(y_pred_test_orig, residuals, alpha=0.5, s=10)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Price ($100k)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# Feature importance
axes[1, 1].barh(housing.feature_names, np.abs(weights))
axes[1, 1].set_xlabel('Absolute Weight')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/06_multivariate_results.png', dpi=100)
print("\nSaved visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Points for Multivariate Regression:

1. Multiple Features: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

2. Feature Scaling is Critical:
   - Features on different scales can cause training issues
   - StandardScaler: (x - mean) / std
   - Always fit on training data only!

3. Train/Test Split:
   - Evaluate on unseen data
   - Prevents overfitting assessment

4. Adam Optimizer:
   - Often works better than SGD for multivariate
   - Adaptive learning rates per parameter

5. Evaluation Metrics:
   - R²: Proportion of variance explained (1.0 is perfect)
   - MAE: Mean Absolute Error (interpretable)

Next: Tutorial 07 - Polynomial Regression!
""")
