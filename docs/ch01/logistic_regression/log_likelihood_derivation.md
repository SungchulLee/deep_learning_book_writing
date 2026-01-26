# Log-Likelihood Derivation for Logistic Regression

## Learning Objectives

By the end of this section, you will be able to:

- Derive the likelihood function for logistic regression from first principles
- Understand why we maximize log-likelihood instead of likelihood
- Connect the log-likelihood to the probability model
- Derive the complete mathematical form used in optimization

---

## The Likelihood Function

### From Probability to Likelihood

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{0, 1\}$, the **likelihood function** measures how probable the observed data is under a specific parameter setting $\boldsymbol{\theta} = \{\boldsymbol{\beta}\}$.

For a single observation, the probability of observing $y_i$ given $\mathbf{x}_i$ and parameters $\boldsymbol{\beta}$ is:

$$
P(Y = y_i | \mathbf{x}_i, \boldsymbol{\beta}) = 
\begin{cases}
p_i & \text{if } y_i = 1 \\
1 - p_i & \text{if } y_i = 0
\end{cases}
$$

where $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$ is our model's predicted probability.

### Compact Form Using the Bernoulli Distribution

We can write this more compactly using the Bernoulli probability mass function:

$$
P(Y = y_i | \mathbf{x}_i, \boldsymbol{\beta}) = p_i^{y_i} (1-p_i)^{1-y_i}
$$

**Verification:**
- When $y_i = 1$: $p_i^1(1-p_i)^0 = p_i$ ✓
- When $y_i = 0$: $p_i^0(1-p_i)^1 = 1-p_i$ ✓

### The Full Likelihood Function

Assuming observations are **independent and identically distributed (i.i.d.)**, the joint probability of all observations is the product of individual probabilities:

$$
\mathcal{L}(\boldsymbol{\beta}) = P(\mathcal{D} | \boldsymbol{\beta}) = \prod_{i=1}^{n} P(y_i | \mathbf{x}_i, \boldsymbol{\beta}) = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}
$$

This is the **likelihood function** — it tells us how "likely" the observed data is for a given choice of parameters.

---

## From Likelihood to Log-Likelihood

### Why Take the Logarithm?

Working with the log-likelihood has several advantages:

| Issue with Likelihood | How Log-Likelihood Helps |
|----------------------|-------------------------|
| Product of many small numbers → numerical underflow | Sum of log-probabilities is numerically stable |
| Product rule for derivatives is complex | Sum rule for derivatives is simpler |
| Non-convex for some models | Log-likelihood is concave for logistic regression |

### The Log-Likelihood Function

Taking the natural logarithm of the likelihood:

$$
\ell(\boldsymbol{\beta}) = \log \mathcal{L}(\boldsymbol{\beta}) = \log \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}
$$

Using the property $\log(ab) = \log a + \log b$:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \log \left[ p_i^{y_i} (1-p_i)^{1-y_i} \right]
$$

Using the property $\log(a^b) = b \log a$:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

This is the **log-likelihood** for logistic regression.

---

## Expanding with the Logistic Model

### Substituting the Sigmoid

Recall that $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}_i^\top \boldsymbol{\beta}}}$.

Let's define $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$ for convenience. Then:

$$
p_i = \sigma(z_i) = \frac{e^{z_i}}{1 + e^{z_i}}
$$

$$
1 - p_i = \frac{1}{1 + e^{z_i}}
$$

### Computing $\log p_i$ and $\log(1-p_i)$

$$
\log p_i = \log \frac{e^{z_i}}{1 + e^{z_i}} = z_i - \log(1 + e^{z_i})
$$

$$
\log(1-p_i) = \log \frac{1}{1 + e^{z_i}} = -\log(1 + e^{z_i})
$$

### Substituting Back

The log-likelihood becomes:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i (z_i - \log(1 + e^{z_i})) + (1-y_i)(-\log(1 + e^{z_i})) \right]
$$

Distributing:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i z_i - y_i \log(1 + e^{z_i}) - \log(1 + e^{z_i}) + y_i \log(1 + e^{z_i}) \right]
$$

The $y_i \log(1 + e^{z_i})$ terms cancel:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i z_i - \log(1 + e^{z_i}) \right]
$$

Substituting $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$:

$$
\boxed{\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log(1 + e^{\mathbf{x}_i^\top \boldsymbol{\beta}}) \right]}
$$

This is the **final form of the log-likelihood** for logistic regression.

---

## Alternative Forms

### Matrix Notation

For computational efficiency, we can write in matrix form:

$$
\ell(\boldsymbol{\beta}) = \mathbf{y}^\top \mathbf{X} \boldsymbol{\beta} - \mathbf{1}^\top \log(\mathbf{1} + \exp(\mathbf{X}\boldsymbol{\beta}))
$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix
- $\mathbf{y} \in \{0,1\}^n$ is the label vector
- $\mathbf{1}$ is a vector of ones
- Operations are element-wise where appropriate

### Per-Sample Log-Likelihood

The contribution of a single sample to the log-likelihood:

$$
\ell_i(\boldsymbol{\beta}) = y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log(1 + e^{\mathbf{x}_i^\top \boldsymbol{\beta}})
$$

This form is useful for:
- Stochastic gradient descent (computing gradients sample-by-sample)
- Understanding which samples have high/low likelihood
- Detecting outliers (samples with unusually low likelihood)

---

## Properties of the Log-Likelihood

### Concavity

The logistic regression log-likelihood is **concave** (negative of convex). This means:

1. **Any local maximum is a global maximum**
2. Gradient descent converges to the global optimum
3. No risk of getting stuck in local minima

**Proof of concavity:**

The Hessian of the log-likelihood is:

$$
\nabla^2 \ell(\boldsymbol{\beta}) = -\sum_{i=1}^{n} p_i(1-p_i) \mathbf{x}_i \mathbf{x}_i^\top = -\mathbf{X}^\top \mathbf{W} \mathbf{X}
$$

where $\mathbf{W} = \text{diag}(p_1(1-p_1), \ldots, p_n(1-p_n))$.

Since $0 < p_i < 1$ for finite $\mathbf{x}_i^\top\boldsymbol{\beta}$, we have $p_i(1-p_i) > 0$, making $\mathbf{W}$ positive definite. Therefore $-\mathbf{X}^\top \mathbf{W} \mathbf{X}$ is negative semi-definite, confirming concavity.

### Scale of Log-Likelihood

The log-likelihood is typically negative (since we're taking log of probabilities < 1). Better models have log-likelihood values closer to zero.

| Log-Likelihood | Interpretation |
|----------------|----------------|
| Close to 0 | Excellent fit (high probability of observed data) |
| Large negative | Poor fit (low probability of observed data) |
| $-n \log 2$ | No better than random guessing |

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

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Part 1: Manual Log-Likelihood Computation
# ============================================================================

print("="*70)
print("LOG-LIKELIHOOD DERIVATION AND COMPUTATION")
print("="*70)

def compute_log_likelihood_manual(X, y, beta):
    """
    Compute log-likelihood using the derived formula.
    
    ℓ(β) = Σᵢ [yᵢ xᵢᵀβ - log(1 + exp(xᵢᵀβ))]
    
    Args:
        X: Feature matrix (n, d)
        y: Labels (n,) with values in {0, 1}
        beta: Parameters (d,)
        
    Returns:
        Log-likelihood value (scalar)
    """
    z = X @ beta  # Linear predictor: (n,)
    
    # Log-likelihood: y*z - log(1 + exp(z))
    # Use logsumexp trick for numerical stability
    ll = y * z - torch.logsumexp(torch.stack([torch.zeros_like(z), z], dim=0), dim=0)
    
    return ll.sum()

def compute_log_likelihood_direct(X, y, beta):
    """
    Compute log-likelihood using the direct formula.
    
    ℓ(β) = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
    
    This is equivalent but computed differently.
    """
    z = X @ beta
    p = torch.sigmoid(z)
    
    # Add small epsilon for numerical stability
    eps = 1e-10
    ll = y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)
    
    return ll.sum()

# Generate simple dataset
n_samples = 100
X = torch.randn(n_samples, 2)
true_beta = torch.tensor([0.5, 1.0, -0.5])  # Including bias

# Add bias column
X_with_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)

# Generate labels
z_true = X_with_bias @ true_beta
p_true = torch.sigmoid(z_true)
y = torch.bernoulli(p_true)

print("\nDataset generated:")
print(f"  Samples: {n_samples}")
print(f"  Features: {X.shape[1]}")
print(f"  Class distribution: {y.sum().item():.0f} positive, {(1-y).sum().item():.0f} negative")

# Compare both computation methods
test_beta = torch.tensor([0.3, 0.8, -0.3])
ll_manual = compute_log_likelihood_manual(X_with_bias, y, test_beta)
ll_direct = compute_log_likelihood_direct(X_with_bias, y, test_beta)

print(f"\nLog-likelihood computation comparison (β = {test_beta.tolist()}):")
print(f"  Formula 1 (y*z - log(1+exp(z))): {ll_manual.item():.6f}")
print(f"  Formula 2 (y*log(p) + (1-y)*log(1-p)): {ll_direct.item():.6f}")
print(f"  Difference: {abs(ll_manual - ll_direct).item():.2e}")

# ============================================================================
# Part 2: Log-Likelihood vs Parameters
# ============================================================================

print("\n" + "="*70)
print("LOG-LIKELIHOOD LANDSCAPE")
print("="*70)

# Create a grid of parameter values to explore
beta1_range = torch.linspace(-2, 2, 50)
beta2_range = torch.linspace(-2, 2, 50)

# Fix intercept at true value for 2D visualization
fixed_intercept = true_beta[0]

log_likelihood_surface = torch.zeros(len(beta1_range), len(beta2_range))

for i, b1 in enumerate(beta1_range):
    for j, b2 in enumerate(beta2_range):
        beta = torch.tensor([fixed_intercept, b1, b2])
        log_likelihood_surface[i, j] = compute_log_likelihood_manual(X_with_bias, y, beta)

# Find maximum
max_idx = torch.argmax(log_likelihood_surface)
max_i, max_j = max_idx // len(beta2_range), max_idx % len(beta2_range)
mle_beta1, mle_beta2 = beta1_range[max_i], beta2_range[max_j]

print(f"\nApproximate MLE (grid search):")
print(f"  β₁: {mle_beta1:.3f} (true: {true_beta[1]:.3f})")
print(f"  β₂: {mle_beta2:.3f} (true: {true_beta[2]:.3f})")
print(f"  Max log-likelihood: {log_likelihood_surface.max():.3f}")

# ============================================================================
# Part 3: PyTorch Training and Log-Likelihood Tracking
# ============================================================================

print("\n" + "="*70)
print("MAXIMUM LIKELIHOOD ESTIMATION VIA GRADIENT DESCENT")
print("="*70)

class LogisticRegressionML(nn.Module):
    """Logistic Regression with explicit log-likelihood tracking."""
    
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def log_likelihood(self, x, y):
        """
        Compute the log-likelihood of the data under current parameters.
        
        Args:
            x: Features (n, d)
            y: Labels (n, 1)
            
        Returns:
            Total log-likelihood
        """
        z = self.linear(x)
        # log p = log σ(z) = -log(1 + exp(-z))
        # log (1-p) = log(1 - σ(z)) = -log(1 + exp(z))
        log_p = -torch.log1p(torch.exp(-z))
        log_1_minus_p = -torch.log1p(torch.exp(z))
        
        ll = y * log_p + (1 - y) * log_1_minus_p
        return ll.sum()

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Train model
model = LogisticRegressionML(2)
criterion = nn.BCELoss()  # This is equivalent to negative log-likelihood (scaled)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 200
history = {
    'log_likelihood': [],
    'nll_loss': [],
    'accuracy': []
}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train)
    nll_loss = criterion(predictions, y_train)  # Negative log-likelihood (averaged)
    
    # Compute log-likelihood explicitly
    with torch.no_grad():
        ll = model.log_likelihood(X_train, y_train)
        acc = ((predictions >= 0.5).float() == y_train).float().mean()
    
    # Backward pass
    optimizer.zero_grad()
    nll_loss.backward()
    optimizer.step()
    
    # Store history
    history['log_likelihood'].append(ll.item())
    history['nll_loss'].append(nll_loss.item())
    history['accuracy'].append(acc.item())
    
    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d}: Log-Likelihood = {ll.item():8.3f}, "
              f"NLL Loss = {nll_loss.item():.4f}, Accuracy = {acc.item():.4f}")

# Final parameters
learned_weights = model.linear.weight.data.numpy().flatten()
learned_bias = model.linear.bias.data.numpy()[0]

print(f"\nLearned parameters:")
print(f"  Intercept: {learned_bias:.4f}")
print(f"  Weights: {learned_weights}")

# ============================================================================
# Part 4: Visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Log-likelihood surface
ax1 = axes[0, 0]
B1, B2 = torch.meshgrid(beta1_range, beta2_range, indexing='ij')
contour = ax1.contourf(B1.numpy(), B2.numpy(), log_likelihood_surface.numpy(), 
                       levels=30, cmap='viridis')
ax1.colorbar(contour, label='Log-Likelihood')
ax1.scatter([true_beta[1]], [true_beta[2]], color='red', s=100, marker='*', 
           label='True β', zorder=5)
ax1.scatter([mle_beta1], [mle_beta2], color='white', s=100, marker='o', 
           edgecolors='black', label='MLE (grid)', zorder=5)
ax1.set_xlabel('β₁', fontsize=11)
ax1.set_ylabel('β₂', fontsize=11)
ax1.set_title('Log-Likelihood Surface\n(Intercept Fixed)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')

# Plot 2: Log-likelihood over training
ax2 = axes[0, 1]
ax2.plot(history['log_likelihood'], 'b-', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Log-Likelihood', fontsize=11)
ax2.set_title('Log-Likelihood During Training\n(Maximization)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
# Add arrow showing direction of optimization
ax2.annotate('', xy=(num_epochs-1, max(history['log_likelihood'])), 
            xytext=(0, history['log_likelihood'][0]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(num_epochs/2, np.mean(history['log_likelihood']), 'Maximizing ↑', 
        fontsize=10, color='green')

# Plot 3: NLL Loss over training
ax3 = axes[1, 0]
ax3.plot(history['nll_loss'], 'r-', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('BCE Loss (NLL)', fontsize=11)
ax3.set_title('Negative Log-Likelihood (Loss) During Training\n(Minimization)', 
             fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.annotate('', xy=(num_epochs-1, min(history['nll_loss'])), 
            xytext=(0, history['nll_loss'][0]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(num_epochs/2, np.mean(history['nll_loss']), 'Minimizing ↓', 
        fontsize=10, color='red')

# Plot 4: Relationship between LL and NLL
ax4 = axes[1, 1]
ll_vals = np.array(history['log_likelihood'])
nll_vals = np.array(history['nll_loss'])

# NLL loss is averaged, so multiply by n to compare
n = len(X_train)
ax4.scatter(ll_vals, -nll_vals * n, alpha=0.5, s=20)
ax4.plot([ll_vals.min(), ll_vals.max()], [ll_vals.min(), ll_vals.max()], 
        'r--', label='y = x')
ax4.set_xlabel('Log-Likelihood (computed)', fontsize=11)
ax4.set_ylabel('-n × BCE Loss', fontsize=11)
ax4.set_title('Log-Likelihood vs Scaled BCE Loss\n(Should be equal)', 
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('log_likelihood_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 5: Per-Sample Log-Likelihood Analysis
# ============================================================================

print("\n" + "="*70)
print("PER-SAMPLE LOG-LIKELIHOOD ANALYSIS")
print("="*70)

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    
    # Compute per-sample log-likelihood
    z = model.linear(X_test)
    log_p = -torch.log1p(torch.exp(-z))
    log_1_minus_p = -torch.log1p(torch.exp(z))
    
    per_sample_ll = (y_test * log_p + (1 - y_test) * log_1_minus_p).flatten()

print("\nPer-sample log-likelihood statistics:")
print(f"  Mean: {per_sample_ll.mean().item():.4f}")
print(f"  Std:  {per_sample_ll.std().item():.4f}")
print(f"  Min:  {per_sample_ll.min().item():.4f} (hardest sample)")
print(f"  Max:  {per_sample_ll.max().item():.4f} (easiest sample)")

# Identify challenging samples
threshold = per_sample_ll.mean() - 2 * per_sample_ll.std()
hard_samples = (per_sample_ll < threshold).sum().item()
print(f"\nSamples with unusually low log-likelihood (< μ - 2σ): {hard_samples}")
```

---

## Key Insights

### Maximum Likelihood Estimation

Finding the optimal parameters means finding $\boldsymbol{\beta}^* = \arg\max_{\boldsymbol{\beta}} \ell(\boldsymbol{\beta})$.

Since there's no closed-form solution (unlike linear regression), we use iterative optimization:

1. **Gradient Descent**: Move in the direction of increasing likelihood
2. **Newton-Raphson**: Use second-order information for faster convergence
3. **Stochastic Methods**: Scale to large datasets

### Connection to Cross-Entropy

The **negative** average log-likelihood is exactly the **Binary Cross-Entropy (BCE)** loss:

$$
\text{BCE} = -\frac{1}{n}\ell(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

Minimizing BCE = Maximizing log-likelihood = Maximum Likelihood Estimation

---

## Exercises

### Mathematical

1. **Show** that the log-likelihood is concave by computing its second derivative with respect to a single coefficient $\beta_j$.

2. **Derive** the expected log-likelihood under the true data distribution and show that it equals the negative cross-entropy between the true and predicted distributions.

3. **Prove** that the MLE for logistic regression exists and is unique when the data is not perfectly separable.

### Computational

4. Implement Newton-Raphson optimization for logistic regression and compare convergence speed to gradient descent.

5. Create a visualization showing how the per-sample log-likelihood varies across the feature space.

---

## Summary

| Expression | Formula | Interpretation |
|------------|---------|----------------|
| Bernoulli PMF | $p^y(1-p)^{1-y}$ | Probability of single observation |
| Likelihood | $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$ | Probability of all data |
| Log-likelihood | $\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$ | Sum of log-probabilities |
| Simplified form | $\sum_i [y_i z_i - \log(1+e^{z_i})]$ | Efficient computation |
| BCE Loss | $-\frac{1}{n}\ell(\boldsymbol{\beta})$ | Negative average log-likelihood |

The log-likelihood provides a principled objective function that emerges directly from probabilistic assumptions about the data-generating process. Maximizing log-likelihood is equivalent to minimizing BCE loss, connecting classical statistics to modern deep learning.
