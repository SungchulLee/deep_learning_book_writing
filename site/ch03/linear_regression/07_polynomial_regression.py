"""
==============================================================================
07_polynomial_regression.py
==============================================================================
DIFFICULTY: ⭐⭐⭐ (Intermediate-Advanced)

DESCRIPTION:
    Polynomial regression to fit non-linear relationships.
    Demonstrates feature engineering and overfitting.

TOPICS COVERED:
    - Polynomial feature expansion
    - Overfitting vs underfitting
    - Model complexity trade-offs
    - Feature engineering

PREREQUISITES:
    - Tutorial 06 (Multivariate regression)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("POLYNOMIAL REGRESSION")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE NON-LINEAR DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE NON-LINEAR DATA")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# Generate data from a non-linear function
n_samples = 100
X = np.linspace(-3, 3, n_samples)
y_true = 0.5 * X**3 - 2*X**2 + X + 3  # True cubic function
y = y_true + np.random.normal(0, 2, n_samples)  # Add noise

print(f"Generated {n_samples} samples from cubic function")
print(f"True function: y = 0.5x³ - 2x² + x + 3")

# ============================================================================
# PART 2: POLYNOMIAL FEATURE EXPANSION
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: POLYNOMIAL FEATURE EXPANSION")
print("=" * 70)

def create_polynomial_features(X, degree):
    """
    Create polynomial features up to specified degree
    
    For X and degree=3:
        Returns: [1, X, X², X³]
    """
    X = X.reshape(-1, 1)
    features = []
    for d in range(degree + 1):
        features.append(X ** d)
    return np.concatenate(features, axis=1)

# Convert to tensors
X_tensor = torch.FloatTensor(X).reshape(-1, 1)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

print("""
Polynomial Features:
- Degree 1 (Linear): [1, X]
- Degree 2 (Quadratic): [1, X, X²]
- Degree 3 (Cubic): [1, X, X², X³]
- Higher degrees: More complex curves
""")

# ============================================================================
# PART 3: TRAIN MODELS WITH DIFFERENT POLYNOMIAL DEGREES
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: TRAIN MODELS WITH DIFFERENT DEGREES")
print("=" * 70)

degrees = [1, 2, 3, 5, 10]
models = {}
results = {}

for degree in degrees:
    print(f"\nTraining polynomial degree {degree}...")
    
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)
    X_poly_tensor = torch.FloatTensor(X_poly)
    
    # Model
    model = nn.Linear(degree + 1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    n_epochs = 1000
    losses = []
    
    for epoch in range(n_epochs):
        y_pred = model(X_poly_tensor)
        loss = criterion(y_pred, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Store results
    models[degree] = model
    results[degree] = {
        'losses': losses,
        'final_loss': losses[-1],
        'X_poly': X_poly_tensor
    }
    
    print(f"  Final loss: {losses[-1]:.4f}")

# ============================================================================
# PART 4: VISUALIZE DIFFERENT MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: VISUALIZE MODEL COMPLEXITY")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    
    # Original data
    ax.scatter(X, y, alpha=0.5, s=20, label='Data')
    ax.plot(X, y_true, 'g--', linewidth=2, label='True Function', alpha=0.7)
    
    # Model predictions
    model = models[degree]
    with torch.no_grad():
        X_poly = results[degree]['X_poly']
        y_pred = model(X_poly).numpy()
    
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Polynomial Degree {degree} (Loss: {results[degree]["final_loss"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 15)

# Loss comparison
ax = axes[5]
for degree in degrees:
    ax.plot(results[degree]['losses'], label=f'Degree {degree}', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/07_polynomial_comparison.png', dpi=100)
print("Saved visualization")
plt.show()

# ============================================================================
# PART 5: UNDERSTANDING OVER/UNDERFITTING
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: ANALYSIS")
print("=" * 70)

print("""
MODEL COMPLEXITY ANALYSIS:

Degree 1 (Linear):
  ❌ UNDERFITTING
  - Too simple to capture the cubic relationship
  - High training error
  - High test error (if we had test data)

Degree 2 (Quadratic):
  ⚠️ STILL UNDERFITTING
  - Better than linear but not enough
  - Can't capture cubic term
  
Degree 3 (Cubic):
  ✅ JUST RIGHT
  - Matches the true function degree
  - Good fit to data
  - Generalizes well
  
Degree 5:
  ⚠️ STARTING TO OVERFIT
  - More flexible than needed
  - Fits noise in training data
  - May not generalize well

Degree 10:
  ❌ SEVERE OVERFITTING
  - Extremely flexible
  - Fits training data too closely
  - Wiggly, unrealistic predictions
  - Poor generalization
  
KEY INSIGHT: Choose model complexity to match problem complexity!
""")

# ============================================================================
# PART 6: CUSTOM POLYNOMIAL MODEL
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: CUSTOM POLYNOMIAL MODEL CLASS")
print("=" * 70)

class PolynomialRegression(nn.Module):
    """Custom polynomial regression model"""
    
    def __init__(self, degree):
        super(PolynomialRegression, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(degree + 1, 1)
        
    def create_features(self, x):
        """Create polynomial features"""
        features = []
        for d in range(self.degree + 1):
            features.append(x ** d)
        return torch.cat(features, dim=1)
    
    def forward(self, x):
        x_poly = self.create_features(x)
        return self.linear(x_poly)

# Example usage
model = PolynomialRegression(degree=3)
print(f"Created PolynomialRegression model with degree 3")
print(model)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Polynomial Regression Key Points:

1. Feature Engineering:
   - Transform X into [1, X, X², X³, ...]
   - Still linear in parameters (it's still linear regression!)
   - Can fit non-linear relationships

2. Model Complexity:
   - Higher degree = more flexible
   - Too simple = underfitting (high bias)
   - Too complex = overfitting (high variance)

3. Choosing Degree:
   - Use domain knowledge
   - Use validation set
   - Try different degrees and compare

4. Overfitting Prevention (next tutorial):
   - Regularization (L1, L2)
   - More training data
   - Cross-validation
   - Early stopping

Next: Tutorial 08 - Regularization!
""")
