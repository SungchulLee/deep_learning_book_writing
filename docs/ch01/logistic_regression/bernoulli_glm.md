# Bernoulli Distribution and GLM Framework

## Learning Objectives

By the end of this section, you will be able to:

- Understand the Bernoulli distribution as the foundation of binary classification
- Connect logistic regression to the Generalized Linear Model (GLM) framework
- Explain the role of the link function in transforming linear predictions to probabilities
- Derive logistic regression as a natural consequence of GLM assumptions

---

## The Binary Classification Problem

Binary classification involves predicting one of two mutually exclusive outcomes. Examples span virtually every domain:

| Domain | Class 0 | Class 1 |
|--------|---------|---------|
| Medical | Healthy | Disease present |
| Finance | No default | Default |
| Email | Legitimate | Spam |
| Manufacturing | Conforming | Defective |

The fundamental question: given features $\mathbf{x} \in \mathbb{R}^d$, how do we model the probability $P(Y=1|\mathbf{x})$?

---

## The Bernoulli Distribution

### Definition

A random variable $Y$ follows a **Bernoulli distribution** if it takes only two values: $Y \in \{0, 1\}$. The distribution is characterized by a single parameter $p \in [0, 1]$, representing the probability of success:

$$
P(Y = y) = p^y (1-p)^{1-y}, \quad y \in \{0, 1\}
$$

This compact form elegantly captures both cases:

- When $y = 1$: $P(Y=1) = p^1(1-p)^0 = p$
- When $y = 0$: $P(Y=0) = p^0(1-p)^1 = 1-p$

### Properties

The Bernoulli distribution has the following fundamental properties:

| Property | Formula | Interpretation |
|----------|---------|----------------|
| Mean | $\mathbb{E}[Y] = p$ | Expected value equals success probability |
| Variance | $\text{Var}(Y) = p(1-p)$ | Maximum variance at $p=0.5$ |
| Skewness | $\frac{1-2p}{\sqrt{p(1-p)}}$ | Symmetric only when $p=0.5$ |

The variance function $p(1-p)$ plays a crucial role in GLM theory, as we'll see shortly.

### Exponential Family Formulation

The Bernoulli distribution belongs to the **exponential family**, which can be written in canonical form:

$$
P(Y=y) = \exp\left(\eta y - A(\eta) + B(y)\right)
$$

where:

- $\eta = \log\frac{p}{1-p}$ is the **natural parameter** (log-odds)
- $A(\eta) = \log(1 + e^\eta)$ is the **log-partition function**
- $B(y) = 0$ for the Bernoulli case

This representation is profound: the natural parameter $\eta$ is exactly the **logit** of $p$, establishing the deep connection between the Bernoulli distribution and logistic regression.

---

## Generalized Linear Models Framework

### Why We Need GLMs

Linear regression assumes:

$$
Y = \mathbf{x}^\top \boldsymbol{\beta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

This implies $Y \in \mathbb{R}$ and follows a Gaussian distribution. For binary outcomes:

1. The response $Y \in \{0, 1\}$ is clearly not Gaussian
2. A linear predictor $\mathbf{x}^\top \boldsymbol{\beta}$ can produce any value in $\mathbb{R}$, but probabilities must lie in $[0, 1]$

GLMs extend linear regression to handle non-Gaussian responses while preserving the linear relationship between features and a transformed response.

### The Three Components of a GLM

A GLM consists of three components:

**1. Random Component**: The response $Y$ follows a distribution from the exponential family (Bernoulli, Poisson, Gaussian, etc.)

**2. Systematic Component**: A linear predictor combining features:

$$
\eta = \mathbf{x}^\top \boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d
$$

**3. Link Function**: A monotonic, differentiable function $g(\cdot)$ that connects the expected value to the linear predictor:

$$
g(\mu) = \eta, \quad \text{where } \mu = \mathbb{E}[Y|\mathbf{x}]
$$

### The Canonical Link Function

For each exponential family distribution, there exists a **canonical link function** that equals the natural parameter transformation. The canonical link has special properties:

- Simplifies maximum likelihood estimation
- Provides natural interpretations
- Ensures identifiability

For the Bernoulli distribution, the canonical link is the **logit function**:

$$
g(p) = \log\frac{p}{1-p} = \text{logit}(p)
$$

---

## Deriving Logistic Regression from GLM

### Setting Up the Model

We want to model $P(Y=1|\mathbf{x}) = p(\mathbf{x})$ where:

1. **Random Component**: $Y | \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$
2. **Systematic Component**: $\eta = \mathbf{x}^\top \boldsymbol{\beta}$
3. **Link Function**: $g(p) = \log\frac{p}{1-p}$

### The Logistic Function Emerges

Connecting the components through the link function:

$$
\log\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta}
$$

Solving for $p(\mathbf{x})$:

$$
\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x}) = e^{\mathbf{x}^\top \boldsymbol{\beta}} (1-p(\mathbf{x}))
$$

$$
p(\mathbf{x}) = e^{\mathbf{x}^\top \boldsymbol{\beta}} - p(\mathbf{x})e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x})(1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}) = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol{\beta}}}{1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}} = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}} = \sigma(\mathbf{x}^\top \boldsymbol{\beta})
$$

The **sigmoid function** $\sigma(z) = \frac{1}{1+e^{-z}}$ naturally emerges from the GLM framework!

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Part 1: Bernoulli Distribution Exploration
# ============================================================================

def bernoulli_properties(p: float) -> dict:
    """
    Compute theoretical properties of Bernoulli distribution.
    
    Args:
        p: Success probability, must be in [0, 1]
        
    Returns:
        Dictionary containing mean, variance, and skewness
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    
    mean = p
    variance = p * (1 - p)
    skewness = (1 - 2*p) / np.sqrt(p * (1 - p) + 1e-10)  # Avoid division by zero
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness
    }

# Demonstrate Bernoulli properties for various p values
print("="*60)
print("Bernoulli Distribution Properties")
print("="*60)

for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    props = bernoulli_properties(p)
    print(f"p = {p:.2f}: Mean = {props['mean']:.3f}, "
          f"Variance = {props['variance']:.3f}, "
          f"Skewness = {props['skewness']:+.3f}")

# ============================================================================
# Part 2: GLM Components Implementation
# ============================================================================

class LogisticRegressionGLM(nn.Module):
    """
    Logistic Regression as a Generalized Linear Model.
    
    This implementation explicitly separates the GLM components:
    - Linear predictor (systematic component)
    - Inverse link function (sigmoid)
    
    The random component (Bernoulli) is implicit in the BCE loss.
    """
    
    def __init__(self, n_features: int):
        """
        Initialize the GLM model.
        
        Args:
            n_features: Number of input features
        """
        super().__init__()
        # Systematic component: linear predictor
        self.linear = nn.Linear(n_features, 1)
    
    def linear_predictor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear predictor η = Xβ.
        
        This is the systematic component of the GLM.
        
        Args:
            x: Input features of shape (batch_size, n_features)
            
        Returns:
            Linear predictor η of shape (batch_size, 1)
        """
        return self.linear(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse link function to get probabilities.
        
        The canonical link for Bernoulli is logit, so the inverse
        link is the sigmoid function: p = σ(η) = 1/(1 + e^{-η})
        
        Args:
            x: Input features of shape (batch_size, n_features)
            
        Returns:
            Predicted probabilities of shape (batch_size, 1)
        """
        eta = self.linear_predictor(x)
        return torch.sigmoid(eta)  # Inverse link function
    
    def log_odds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log-odds (natural parameter).
        
        This is the link function applied to the mean:
        η = log(p/(1-p)) = logit(p)
        
        For our model, this equals the linear predictor directly.
        """
        return self.linear_predictor(x)

# ============================================================================
# Part 3: Visualizing the GLM Structure
# ============================================================================

# Create model
model = LogisticRegressionGLM(n_features=1)

# Initialize weights for clear visualization
model.linear.weight.data = torch.tensor([[2.0]])
model.linear.bias.data = torch.tensor([-1.0])

# Generate data for visualization
x_vals = torch.linspace(-3, 5, 200).reshape(-1, 1)

# Compute GLM components
with torch.no_grad():
    eta = model.linear_predictor(x_vals)  # Linear predictor
    p = model(x_vals)                      # Probability (inverse link applied)

# Plot the GLM transformation
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Linear predictor
axes[0].plot(x_vals.numpy(), eta.numpy(), 'b-', linewidth=2)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Feature x', fontsize=11)
axes[0].set_ylabel('Linear Predictor η', fontsize=11)
axes[0].set_title('Systematic Component\nη = β₀ + β₁x', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-8, 10])

# Plot 2: Sigmoid (inverse link)
z = torch.linspace(-6, 6, 200)
sigmoid_z = torch.sigmoid(z)
axes[1].plot(z.numpy(), sigmoid_z.numpy(), 'g-', linewidth=2)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Linear Predictor η', fontsize=11)
axes[1].set_ylabel('Probability p', fontsize=11)
axes[1].set_title('Inverse Link Function\np = σ(η) = 1/(1+e⁻η)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Final model (composition)
axes[2].plot(x_vals.numpy(), p.numpy(), 'r-', linewidth=2)
axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Feature x', fontsize=11)
axes[2].set_ylabel('P(Y=1|x)', fontsize=11)
axes[2].set_title('Complete GLM\np = σ(β₀ + β₁x)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('glm_structure.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("GLM Model Summary")
print("="*60)
print(f"Coefficient (β₁): {model.linear.weight.item():.3f}")
print(f"Intercept (β₀): {model.linear.bias.item():.3f}")
print(f"Decision boundary (where p=0.5): x = {-model.linear.bias.item()/model.linear.weight.item():.3f}")
```

---

## Key Insights

### Why the GLM Framework Matters

1. **Unified Theory**: GLMs provide a common language for regression (Gaussian), classification (Bernoulli), and count data (Poisson) problems.

2. **Natural Constraints**: The link function automatically maps unbounded linear predictors to the appropriate range for each distribution.

3. **Interpretation**: The logit link provides interpretable coefficients in terms of log-odds ratios.

4. **Principled Inference**: GLM theory provides asymptotic results for confidence intervals and hypothesis tests.

### Connecting to Neural Networks

Logistic regression is the simplest neural network:

- Single linear layer followed by sigmoid activation
- Loss function derived from maximum likelihood (BCE)
- Gradient descent for optimization

This perspective bridges classical statistics and deep learning, showing that neural networks extend GLM ideas to more complex architectures.

---

## Exercises

### Conceptual

1. **Prove** that the Bernoulli variance $p(1-p)$ is maximized at $p=0.5$. What is the significance of this result for classification?

2. **Explain** why the logit function is called the "canonical" link for the Bernoulli distribution. What happens if we use a different link function (e.g., probit)?

3. **Derive** the relationship between the natural parameter $\eta$ and the mean parameter $p$ for the Bernoulli distribution from first principles.

### Computational

4. Implement a probit regression model (using the normal CDF as the inverse link) and compare it to logistic regression on synthetic data.

5. Create a visualization showing how the Bernoulli variance $p(1-p)$ changes as a function of the linear predictor $\eta = \mathbf{x}^\top \boldsymbol{\beta}$.

---

## Summary

| Concept | Mathematical Form | Role in GLM |
|---------|------------------|-------------|
| Response Distribution | $Y \sim \text{Bernoulli}(p)$ | Random component |
| Linear Predictor | $\eta = \mathbf{x}^\top \boldsymbol{\beta}$ | Systematic component |
| Link Function | $g(p) = \log\frac{p}{1-p}$ | Connects $\mu$ to $\eta$ |
| Inverse Link | $g^{-1}(\eta) = \sigma(\eta)$ | Maps $\eta$ to valid $p$ |

The GLM framework provides the theoretical foundation for understanding logistic regression not as an arbitrary choice, but as the natural model for binary classification when we assume a Bernoulli response and linear effects of features on the log-odds scale.

---

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- Agresti, A. (2015). *Foundations of Linear and Generalized Linear Models*. Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.
