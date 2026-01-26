# Sigmoid Function and Odds Ratio

## Learning Objectives

By the end of this section, you will be able to:

- Derive the sigmoid function from first principles
- Understand the properties that make sigmoid ideal for probability modeling
- Interpret model coefficients in terms of odds and odds ratios
- Explain the probabilistic meaning of the decision boundary
- Implement and visualize sigmoid transformations in PyTorch

---

## The Sigmoid Function

### Derivation

The sigmoid function emerges naturally from modeling log-odds as a linear function:

$$
\log\frac{p}{1-p} = z
$$

Solving for $p$:

$$
\frac{p}{1-p} = e^z \implies p = e^z(1-p) \implies p(1+e^z) = e^z
$$

$$
p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}} \equiv \sigma(z)
$$

### Mathematical Definition

The **sigmoid function** (also called the logistic function) is defined as:

$$
\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{1+e^z}
$$

Both forms are equivalent and useful in different contexts. The first form is more numerically stable for large positive $z$, while the second is stable for large negative $z$.

### Key Properties

| Property | Mathematical Form | Significance |
|----------|------------------|--------------|
| Range | $\sigma: \mathbb{R} \to (0, 1)$ | Maps any real number to a valid probability |
| Symmetry | $\sigma(-z) = 1 - \sigma(z)$ | Complementary probabilities are symmetric |
| Center | $\sigma(0) = 0.5$ | Zero log-odds gives equal probability |
| Derivative | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Elegant gradient for backpropagation |
| Limits | $\lim_{z\to\infty}\sigma(z) = 1$, $\lim_{z\to-\infty}\sigma(z) = 0$ | Saturates smoothly at extremes |
| Inverse | $\sigma^{-1}(p) = \log\frac{p}{1-p}$ | Logit function recovers log-odds |

### The Symmetry Property

The symmetry $\sigma(-z) = 1 - \sigma(z)$ is fundamental to binary classification. It means:

$$
P(Y=0|z) = 1 - P(Y=1|z) = 1 - \sigma(z) = \sigma(-z)
$$

This ensures that the model is internally consistent: the probabilities of the two classes always sum to 1.

**Proof:**
$$
\sigma(-z) = \frac{1}{1+e^{-(-z)}} = \frac{1}{1+e^z} = \frac{1+e^z - e^z}{1+e^z} = 1 - \frac{e^z}{1+e^z} = 1 - \sigma(z)
$$

### The Derivative

The derivative of the sigmoid function has a remarkably elegant form:

$$
\frac{d\sigma}{dz} = \sigma(z)(1-\sigma(z))
$$

**Proof using quotient rule:**

Let $\sigma(z) = \frac{1}{1+e^{-z}}$. Then:

$$
\sigma'(z) = \frac{0 \cdot (1+e^{-z}) - 1 \cdot (-e^{-z})}{(1+e^{-z})^2} = \frac{e^{-z}}{(1+e^{-z})^2}
$$

Notice that:
$$
\sigma(z)(1-\sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma'(z)
$$

This property is crucial for backpropagation: the gradient can be computed directly from the forward pass output without storing additional values.

---

## Odds and Odds Ratio

### What Are Odds?

**Odds** express the ratio of the probability of an event occurring to the probability of it not occurring:

$$
\text{Odds}(Y=1) = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}
$$

| Probability $p$ | Odds | Interpretation |
|-----------------|------|----------------|
| 0.5 | 1:1 | Equal chance |
| 0.75 | 3:1 | Three times more likely to occur |
| 0.9 | 9:1 | Nine times more likely |
| 0.1 | 1:9 | Nine times more likely NOT to occur |

### Log-Odds (Logit)

The **log-odds** or **logit** is the natural logarithm of the odds:

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

Log-odds have a key advantage: they range from $-\infty$ to $+\infty$, making them suitable targets for linear modeling.

| Probability $p$ | Odds | Log-Odds |
|-----------------|------|----------|
| 0.01 | 0.0101 | -4.60 |
| 0.10 | 0.111 | -2.20 |
| 0.50 | 1.0 | 0.00 |
| 0.90 | 9.0 | 2.20 |
| 0.99 | 99.0 | 4.60 |

### Interpreting Coefficients as Log-Odds Ratios

In logistic regression, we model:

$$
\log\frac{P(Y=1|\mathbf{x})}{P(Y=0|\mathbf{x})} = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d
$$

Consider a single feature $x_1$. The **odds ratio** for a one-unit increase in $x_1$ (holding other variables constant) is:

$$
\text{OR} = \frac{\text{Odds}(Y=1|x_1+1)}{\text{Odds}(Y=1|x_1)} = \frac{e^{\beta_0 + \beta_1(x_1+1)}}{e^{\beta_0 + \beta_1 x_1}} = e^{\beta_1}
$$

This is a powerful result: **$e^{\beta_j}$ represents the multiplicative change in odds for a one-unit increase in $x_j$.**

| $\beta_1$ | $e^{\beta_1}$ | Interpretation |
|-----------|---------------|----------------|
| 0 | 1.0 | No effect on odds |
| 0.5 | 1.65 | 65% increase in odds per unit |
| 1.0 | 2.72 | Odds nearly triple |
| -0.5 | 0.61 | 39% decrease in odds |
| -1.0 | 0.37 | Odds reduce to ~37% |

---

## The Decision Boundary

### Where the Boundary Lies

The **decision boundary** is the set of points where $P(Y=1|\mathbf{x}) = 0.5$, which occurs when:

$$
\sigma(\mathbf{x}^\top\boldsymbol{\beta}) = 0.5 \implies \mathbf{x}^\top\boldsymbol{\beta} = 0
$$

For two features, this is a line:
$$
\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0 \implies x_2 = -\frac{\beta_0}{\beta_2} - \frac{\beta_1}{\beta_2}x_1
$$

### Geometric Interpretation

- The weight vector $\boldsymbol{\beta}_{1:d} = [\beta_1, \ldots, \beta_d]$ is **perpendicular** to the decision boundary
- The intercept $\beta_0$ controls the **offset** from the origin
- The magnitude $\|\boldsymbol{\beta}\|$ controls how **steep** the probability transition is

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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Part 1: Sigmoid Function Analysis
# ============================================================================

print("="*70)
print("SIGMOID FUNCTION PROPERTIES")
print("="*70)

def sigmoid(z):
    """Numerically stable sigmoid function."""
    return torch.sigmoid(z)

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))."""
    s = sigmoid(z)
    return s * (1 - s)

# Demonstrate key properties
z_vals = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
print("\nSigmoid values at key points:")
print("-" * 50)
for z in z_vals:
    s = sigmoid(z)
    print(f"σ({z:+.1f}) = {s.item():.6f}")

# Verify symmetry property
print("\nVerifying symmetry σ(-z) = 1 - σ(z):")
print("-" * 50)
for z in [1.0, 2.0, 3.0]:
    z_tensor = torch.tensor(z)
    left = sigmoid(-z_tensor)
    right = 1 - sigmoid(z_tensor)
    print(f"σ({-z:.1f}) = {left.item():.6f}, 1 - σ({z:.1f}) = {right.item():.6f}, "
          f"Difference: {abs(left - right).item():.2e}")

# Verify derivative property
print("\nVerifying derivative σ'(z) = σ(z)(1 - σ(z)):")
print("-" * 50)
z_test = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
s_test = sigmoid(z_test)
for i, z in enumerate(z_test):
    # Compute analytical derivative
    analytical = sigmoid_derivative(z)
    
    # Compute numerical derivative using autograd
    s_test[i].backward(retain_graph=True)
    numerical = z_test.grad[i]
    z_test.grad.zero_()
    
    print(f"z = {z.item():+.1f}: Analytical = {analytical.item():.6f}, "
          f"Autograd = {numerical.item():.6f}")

# ============================================================================
# Part 2: Odds and Log-Odds Visualization
# ============================================================================

print("\n" + "="*70)
print("ODDS AND LOG-ODDS RELATIONSHIPS")
print("="*70)

probabilities = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

print("\nProbability → Odds → Log-Odds conversion:")
print("-" * 60)
print(f"{'Probability':>12} {'Odds':>12} {'Log-Odds':>12}")
print("-" * 60)

for p in probabilities:
    odds = p / (1 - p)
    log_odds = torch.log(odds)
    print(f"{p.item():>12.4f} {odds.item():>12.4f} {log_odds.item():>12.4f}")

# ============================================================================
# Part 3: Logistic Regression with Interpretable Coefficients
# ============================================================================

print("\n" + "="*70)
print("LOGISTIC REGRESSION COEFFICIENT INTERPRETATION")
print("="*70)

# Generate synthetic data with known feature effects
# Feature 1: Strong positive effect (increases log-odds)
# Feature 2: Moderate negative effect (decreases log-odds)

n_samples = 1000
X = np.random.randn(n_samples, 2)
true_beta = np.array([0.5, 1.5, -0.8])  # [intercept, beta1, beta2]
z = true_beta[0] + X[:, 0] * true_beta[1] + X[:, 1] * true_beta[2]
p = 1 / (1 + np.exp(-z))
y = np.random.binomial(1, p)

print(f"\nTrue coefficients:")
print(f"  Intercept (β₀): {true_beta[0]:.3f}")
print(f"  Feature 1 (β₁): {true_beta[1]:.3f} → Odds Ratio: {np.exp(true_beta[1]):.3f}")
print(f"  Feature 2 (β₂): {true_beta[2]:.3f} → Odds Ratio: {np.exp(true_beta[2]):.3f}")

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Define and train model
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def log_odds(self, x):
        """Return log-odds (linear predictor)."""
        return self.linear(x)

model = LogisticRegression(2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(500):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract learned coefficients
learned_weights = model.linear.weight.data.numpy().flatten()
learned_bias = model.linear.bias.data.numpy()[0]

print(f"\nLearned coefficients (on standardized features):")
print(f"  Intercept (β₀): {learned_bias:.3f}")
print(f"  Feature 1 (β₁): {learned_weights[0]:.3f} → Odds Ratio: {np.exp(learned_weights[0]):.3f}")
print(f"  Feature 2 (β₂): {learned_weights[1]:.3f} → Odds Ratio: {np.exp(learned_weights[1]):.3f}")

# ============================================================================
# Part 4: Decision Boundary Visualization
# ============================================================================

print("\n" + "="*70)
print("DECISION BOUNDARY ANALYSIS")
print("="*70)

# Calculate decision boundary equation
# β₀ + β₁x₁ + β₂x₂ = 0  →  x₂ = -(β₀ + β₁x₁)/β₂
print(f"\nDecision boundary equation:")
print(f"  {learned_weights[1]:.3f}x₂ = {-learned_bias:.3f} + {-learned_weights[0]:.3f}x₁")
print(f"  x₂ = {-learned_bias/learned_weights[1]:.3f} + {-learned_weights[0]/learned_weights[1]:.3f}x₁")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Sigmoid function and its derivative
z_range = torch.linspace(-6, 6, 200)
axes[0].plot(z_range.numpy(), sigmoid(z_range).numpy(), 'b-', linewidth=2, label='σ(z)')
axes[0].plot(z_range.numpy(), sigmoid_derivative(z_range).numpy(), 'r--', linewidth=2, label="σ'(z)")
axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
axes[0].axvline(x=0, color='gray', linestyle=':', alpha=0.7)
axes[0].fill_between(z_range.numpy(), 0, sigmoid_derivative(z_range).numpy(), alpha=0.2, color='red')
axes[0].set_xlabel('z (log-odds)', fontsize=11)
axes[0].set_ylabel('Value', fontsize=11)
axes[0].set_title('Sigmoid Function and Derivative', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-0.05, 1.05])

# Plot 2: Decision boundary with data points
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().flatten()

# Create mesh for probability contours
xx, yy = np.meshgrid(
    np.linspace(X_train_np[:, 0].min() - 0.5, X_train_np[:, 0].max() + 0.5, 100),
    np.linspace(X_train_np[:, 1].min() - 0.5, X_train_np[:, 1].max() + 0.5, 100)
)
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

with torch.no_grad():
    Z = model(grid).reshape(xx.shape).numpy()

contour = axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.8)
axes[1].colorbar(contour, label='P(Y=1|x)')

# Plot decision boundary
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Scatter plot
axes[1].scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
               c='blue', marker='o', label='Class 0', alpha=0.6, edgecolors='w')
axes[1].scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
               c='red', marker='o', label='Class 1', alpha=0.6, edgecolors='w')

# Draw weight vector (perpendicular to decision boundary)
scale = 0.5
axes[1].arrow(0, 0, learned_weights[0]*scale, learned_weights[1]*scale, 
             head_width=0.1, head_length=0.05, fc='green', ec='green', linewidth=2)
axes[1].text(learned_weights[0]*scale + 0.1, learned_weights[1]*scale + 0.1, 
            'β', fontsize=12, fontweight='bold', color='green')

axes[1].set_xlabel('Feature 1 (standardized)', fontsize=11)
axes[1].set_ylabel('Feature 2 (standardized)', fontsize=11)
axes[1].set_title('Decision Boundary and Probability Contours', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Odds ratio interpretation
features = ['Feature 1', 'Feature 2']
odds_ratios = [np.exp(learned_weights[0]), np.exp(learned_weights[1])]
colors = ['green' if or_val > 1 else 'red' for or_val in odds_ratios]

bars = axes[2].bar(features, odds_ratios, color=colors, alpha=0.7, edgecolor='black')
axes[2].axhline(y=1, color='black', linestyle='--', linewidth=1)
axes[2].set_ylabel('Odds Ratio', fontsize=11)
axes[2].set_title('Coefficient Interpretation as Odds Ratios', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, or_val, coef in zip(bars, odds_ratios, learned_weights):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'OR={or_val:.2f}\n(β={coef:.2f})', ha='center', va='bottom', fontsize=9)

# Add interpretation text
axes[2].text(0.5, 0.02, 
            'OR > 1: Increases odds of Y=1\nOR < 1: Decreases odds of Y=1',
            transform=axes[2].transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('sigmoid_odds_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 5: Numerical Stability Considerations
# ============================================================================

print("\n" + "="*70)
print("NUMERICAL STABILITY OF SIGMOID")
print("="*70)

def naive_sigmoid(z):
    """Naive implementation (can overflow)."""
    return 1 / (1 + np.exp(-z))

def stable_sigmoid(z):
    """Numerically stable implementation."""
    return np.where(z >= 0,
                   1 / (1 + np.exp(-z)),
                   np.exp(z) / (1 + np.exp(z)))

# Test with extreme values
extreme_values = np.array([-1000, -100, 0, 100, 1000])

print("\nComparing naive vs stable sigmoid for extreme values:")
print("-" * 60)
print(f"{'z':>10} {'Naive':>15} {'Stable':>15} {'PyTorch':>15}")
print("-" * 60)

for z in extreme_values:
    with np.errstate(over='ignore'):  # Suppress overflow warnings for demo
        naive_result = naive_sigmoid(z)
    stable_result = stable_sigmoid(z)
    pytorch_result = torch.sigmoid(torch.tensor(z, dtype=torch.float32)).item()
    
    print(f"{z:>10.0f} {naive_result:>15.6e} {stable_result:>15.6e} {pytorch_result:>15.6e}")

print("\n✓ PyTorch's sigmoid is numerically stable for all input ranges!")
```

---

## Key Insights

### The Sigmoid as a "Soft" Step Function

The sigmoid function can be viewed as a smooth approximation to the Heaviside step function. The "steepness" of the transition is controlled by the magnitude of the linear predictor's coefficients.

When $\|\boldsymbol{\beta}\|$ is large:
- Predictions are more confident (closer to 0 or 1)
- Decision boundary is "sharper"
- Model is more certain about classifications

When $\|\boldsymbol{\beta}\|$ is small:
- Predictions are more uncertain (closer to 0.5)
- Decision boundary is "softer"
- Model expresses more uncertainty

### Why Log-Odds?

Modeling log-odds rather than probability directly has several advantages:

1. **Unbounded range**: Linear predictors can take any value, which matches the $(-\infty, +\infty)$ range of log-odds
2. **Multiplicative effects**: Coefficients have a clean interpretation as log odds ratios
3. **Natural for exponential family**: Log-odds is the natural parameter for Bernoulli
4. **Symmetric treatment of classes**: Log-odds of 0 means equal probability for both classes

---

## Exercises

### Conceptual

1. **Prove** that the derivative of the sigmoid achieves its maximum at $z=0$. What is this maximum value, and why is it significant for gradient-based learning?

2. **Explain** why the sigmoid function is sometimes called a "squashing function." What problems can arise when inputs are very large in magnitude?

3. A model has coefficients $\beta_1 = 2.0$ for age (in decades). **Interpret** this coefficient in terms of odds ratios and provide a clinical interpretation.

### Computational

4. Implement a function that computes the $95\%$ confidence interval for an odds ratio, given the coefficient and its standard error.

5. Create a visualization showing how the steepness of the decision boundary changes as you scale all coefficients by a constant factor.

6. Implement the probit function (inverse CDF of standard normal) and compare its shape to the sigmoid.

---

## Summary

| Concept | Formula | Key Insight |
|---------|---------|-------------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Maps $\mathbb{R} \to (0,1)$ |
| Symmetry | $\sigma(-z) = 1 - \sigma(z)$ | Complementary probabilities |
| Derivative | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Enables efficient backprop |
| Odds | $\frac{p}{1-p}$ | Ratio of success to failure |
| Odds Ratio | $e^{\beta_j}$ | Multiplicative effect per unit |
| Decision Boundary | $\mathbf{x}^\top\boldsymbol{\beta} = 0$ | Where $p = 0.5$ |

The sigmoid function's mathematical elegance—combining smooth saturation, complementary symmetry, and a self-referential derivative—makes it the canonical choice for binary classification, while the log-odds framework provides interpretable coefficients that quantify how features influence the odds of an outcome.
