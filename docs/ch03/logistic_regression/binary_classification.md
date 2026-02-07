# Binary Classification

## Learning Objectives

By the end of this section, you will be able to:

- Understand the binary classification problem and why linear regression fails for it
- Derive the Bernoulli probability model and its exponential family form
- Connect logistic regression to the Generalized Linear Model (GLM) framework
- Derive the likelihood and log-likelihood functions from first principles
- Explain why minimizing BCE is equivalent to maximum likelihood estimation

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

### Why Not Linear Regression?

Linear regression assumes:

$$
Y = \mathbf{x}^\top \boldsymbol{\beta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

This implies $Y \in \mathbb{R}$ and follows a Gaussian distribution. For binary outcomes:

1. The response $Y \in \{0, 1\}$ is clearly not Gaussian
2. A linear predictor $\mathbf{x}^\top \boldsymbol{\beta}$ can produce any value in $\mathbb{R}$, but probabilities must lie in $[0, 1]$

We need a framework that respects the discrete nature of $Y$ and the bounded range of probabilities.

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

| Property | Formula | Interpretation |
|----------|---------|----------------|
| Mean | $\mathbb{E}[Y] = p$ | Expected value equals success probability |
| Variance | $\text{Var}(Y) = p(1-p)$ | Maximum variance at $p=0.5$ |
| Skewness | $\frac{1-2p}{\sqrt{p(1-p)}}$ | Symmetric only when $p=0.5$ |

The variance function $p(1-p)$ plays a crucial role in GLM theory and in the Hessian of the logistic regression loss (see [Gradient Computation](gradient.md)).

### Exponential Family Formulation

The Bernoulli distribution belongs to the **exponential family**, which can be written in canonical form:

$$
P(Y=y) = \exp\left(\eta y - A(\eta) + B(y)\right)
$$

where:

- $\eta = \log\frac{p}{1-p}$ is the **natural parameter** (log-odds)
- $A(\eta) = \log(1 + e^\eta)$ is the **log-partition function**
- $B(y) = 0$ for the Bernoulli case

**Derivation.** Starting from the Bernoulli PMF:

$$
P(Y=y) = p^y(1-p)^{1-y}
$$

Take logarithms:

$$
\log P(Y=y) = y \log p + (1-y)\log(1-p) = y \log\frac{p}{1-p} + \log(1-p)
$$

Defining $\eta = \log\frac{p}{1-p}$, we can solve for $p$:

$$
p = \frac{e^\eta}{1+e^\eta}, \quad 1-p = \frac{1}{1+e^\eta}
$$

So $\log(1-p) = -\log(1+e^\eta)$, giving:

$$
\log P(Y=y) = \eta \, y - \log(1+e^\eta)
$$

This representation is profound: the natural parameter $\eta$ is exactly the **logit** of $p$, establishing the deep connection between the Bernoulli distribution and logistic regression.

---

## Generalized Linear Models Framework

### The Three Components of a GLM

Generalized Linear Models (GLMs) extend linear regression to handle non-Gaussian responses while preserving the linear relationship between features and a transformed response. A GLM consists of three components:

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

### Deriving Logistic Regression from GLM

We want to model $P(Y=1|\mathbf{x}) = p(\mathbf{x})$ where:

1. **Random Component**: $Y | \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$
2. **Systematic Component**: $\eta = \mathbf{x}^\top \boldsymbol{\beta}$
3. **Link Function**: $g(p) = \log\frac{p}{1-p}$

Connecting the components through the link function:

$$
\log\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta}
$$

Solving for $p(\mathbf{x})$:

$$
\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x})(1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}) = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol{\beta}}}{1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}} = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}} = \sigma(\mathbf{x}^\top \boldsymbol{\beta})
$$

The **sigmoid function** $\sigma(z) = \frac{1}{1+e^{-z}}$ naturally emerges from the GLM framework! Its properties are developed in detail in the [Sigmoid Function](sigmoid.md) section.

---

## The Likelihood Function

### From Probability to Likelihood

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{0, 1\}$, the **likelihood function** measures how probable the observed data is under a specific parameter setting $\boldsymbol{\beta}$.

For a single observation, the probability of observing $y_i$ given $\mathbf{x}_i$ and parameters $\boldsymbol{\beta}$ is:

$$
P(Y = y_i | \mathbf{x}_i, \boldsymbol{\beta}) = p_i^{y_i} (1-p_i)^{1-y_i}
$$

where $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$ is the model's predicted probability.

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
\ell(\boldsymbol{\beta}) = \log \mathcal{L}(\boldsymbol{\beta}) = \sum_{i=1}^{n} \log \left[ p_i^{y_i} (1-p_i)^{1-y_i} \right]
$$

Using the property $\log(a^b) = b \log a$:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

This is the **log-likelihood** for logistic regression.

### Expanding with the Logistic Model

Substituting $p_i = \sigma(z_i)$ where $z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$:

$$
\log p_i = \log \frac{e^{z_i}}{1 + e^{z_i}} = z_i - \log(1 + e^{z_i})
$$

$$
\log(1-p_i) = \log \frac{1}{1 + e^{z_i}} = -\log(1 + e^{z_i})
$$

Substituting back and simplifying (the $y_i \log(1+e^{z_i})$ terms cancel):

$$
\boxed{\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log(1 + e^{\mathbf{x}_i^\top \boldsymbol{\beta}}) \right]}
$$

This is the **final form of the log-likelihood** for logistic regression.

### Matrix Notation

For computational efficiency:

$$
\ell(\boldsymbol{\beta}) = \mathbf{y}^\top \mathbf{X} \boldsymbol{\beta} - \mathbf{1}^\top \log(\mathbf{1} + \exp(\mathbf{X}\boldsymbol{\beta}))
$$

where $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix, $\mathbf{y} \in \{0,1\}^n$ is the label vector, and operations are element-wise where appropriate.

---

## Connection to Binary Cross-Entropy

### From Log-Likelihood to Loss Function

In optimization, we minimize rather than maximize. The **negative log-likelihood (NLL)** becomes our loss:

$$
\text{NLL} = -\ell(\boldsymbol{\beta}) = -\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

Dividing by $n$ to get the average:

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

This is **Binary Cross-Entropy (BCE)** — the standard loss function for binary classification. Thus:

$$
\text{Minimizing BCE} = \text{Maximizing log-likelihood} = \text{Maximum Likelihood Estimation}
$$

### Information-Theoretic Interpretation

Cross-entropy between true distribution $q$ and predicted distribution $p$ is:

$$
H(q, p) = -\mathbb{E}_q[\log p] = -\sum_x q(x) \log p(x)
$$

For binary classification, with true label $y$ and predicted probability $\hat{y}$:

$$
H(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

BCE measures how well our predictions encode the true distribution. It is minimized when $\hat{y} = y$, and the gap between cross-entropy and entropy equals the KL divergence — the information lost by using our model's distribution instead of the true distribution.

---

## Properties of the Log-Likelihood

### Concavity

The logistic regression log-likelihood is **concave** (the negative log-likelihood is convex). This means:

1. **Any local maximum is a global maximum**
2. Gradient descent converges to the global optimum
3. No risk of getting stuck in local minima

The proof relies on showing the Hessian is negative semi-definite, which is developed in the [Gradient Computation](gradient.md) section.

### Scale of Log-Likelihood

The log-likelihood is typically negative (since we take the log of probabilities $< 1$). Better models have log-likelihood values closer to zero.

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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Part 1: Bernoulli Distribution Exploration
# ============================================================================

print("=" * 70)
print("BERNOULLI DISTRIBUTION AND GLM FOUNDATIONS")
print("=" * 70)

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
    skewness = (1 - 2 * p) / np.sqrt(p * (1 - p) + 1e-10)
    return {"mean": mean, "variance": variance, "skewness": skewness}


for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    props = bernoulli_properties(p)
    print(
        f"p = {p:.2f}: Mean = {props['mean']:.3f}, "
        f"Variance = {props['variance']:.3f}, "
        f"Skewness = {props['skewness']:+.3f}"
    )

# ============================================================================
# Part 2: GLM Components — Logistic Regression
# ============================================================================

print("\n" + "=" * 70)
print("GLM STRUCTURE: LINEAR PREDICTOR → SIGMOID → PROBABILITY")
print("=" * 70)


class LogisticRegressionGLM(nn.Module):
    """
    Logistic Regression as a Generalized Linear Model.

    Explicitly separates the GLM components:
    - Linear predictor (systematic component)
    - Inverse link function (sigmoid)

    The random component (Bernoulli) is implicit in the BCE loss.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def linear_predictor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the linear predictor η = Xβ (systematic component)."""
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse link function to get probabilities: p = σ(η)."""
        eta = self.linear_predictor(x)
        return torch.sigmoid(eta)

    def log_odds(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-odds (natural parameter), equal to linear predictor."""
        return self.linear_predictor(x)


# Initialize model with known weights for clear visualization
model = LogisticRegressionGLM(n_features=1)
model.linear.weight.data = torch.tensor([[2.0]])
model.linear.bias.data = torch.tensor([-1.0])

x_vals = torch.linspace(-3, 5, 200).reshape(-1, 1)

with torch.no_grad():
    eta = model.linear_predictor(x_vals)
    p = model(x_vals)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(x_vals.numpy(), eta.numpy(), "b-", linewidth=2)
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Feature x", fontsize=11)
axes[0].set_ylabel("Linear Predictor η", fontsize=11)
axes[0].set_title("Systematic Component\nη = β₀ + β₁x", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-8, 10])

z = torch.linspace(-6, 6, 200)
axes[1].plot(z.numpy(), torch.sigmoid(z).numpy(), "g-", linewidth=2)
axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Linear Predictor η", fontsize=11)
axes[1].set_ylabel("Probability p", fontsize=11)
axes[1].set_title(
    "Inverse Link Function\np = σ(η) = 1/(1+e⁻η)", fontsize=12, fontweight="bold"
)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_vals.numpy(), p.numpy(), "r-", linewidth=2)
axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[2].set_xlabel("Feature x", fontsize=11)
axes[2].set_ylabel("P(Y=1|x)", fontsize=11)
axes[2].set_title("Complete GLM\np = σ(β₀ + β₁x)", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig("glm_structure.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGLM Model Summary")
print(f"  Coefficient (β₁): {model.linear.weight.item():.3f}")
print(f"  Intercept (β₀): {model.linear.bias.item():.3f}")
print(
    f"  Decision boundary (where p=0.5): "
    f"x = {-model.linear.bias.item() / model.linear.weight.item():.3f}"
)

# ============================================================================
# Part 3: Manual Log-Likelihood Computation
# ============================================================================

print("\n" + "=" * 70)
print("LOG-LIKELIHOOD COMPUTATION")
print("=" * 70)


def compute_log_likelihood_manual(X, y, beta):
    """
    Compute log-likelihood using the derived formula.

    ℓ(β) = Σᵢ [yᵢ xᵢᵀβ - log(1 + exp(xᵢᵀβ))]
    """
    z = X @ beta
    ll = y * z - torch.logsumexp(torch.stack([torch.zeros_like(z), z], dim=0), dim=0)
    return ll.sum()


def compute_log_likelihood_direct(X, y, beta):
    """
    Compute log-likelihood using the direct formula.

    ℓ(β) = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
    """
    z = X @ beta
    p = torch.sigmoid(z)
    eps = 1e-10
    ll = y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)
    return ll.sum()


# Generate simple dataset
n_samples = 100
X = torch.randn(n_samples, 2)
true_beta = torch.tensor([0.5, 1.0, -0.5])  # Including bias
X_with_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)

z_true = X_with_bias @ true_beta
p_true = torch.sigmoid(z_true)
y = torch.bernoulli(p_true)

print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
print(f"Class distribution: {y.sum().item():.0f} positive, {(1-y).sum().item():.0f} negative")

# Compare both computation methods
test_beta = torch.tensor([0.3, 0.8, -0.3])
ll_manual = compute_log_likelihood_manual(X_with_bias, y, test_beta)
ll_direct = compute_log_likelihood_direct(X_with_bias, y, test_beta)

print(f"\nLog-likelihood comparison (β = {test_beta.tolist()}):")
print(f"  Simplified form (y·z - log(1+eᶻ)): {ll_manual.item():.6f}")
print(f"  Direct form (y·log p + (1-y)·log(1-p)): {ll_direct.item():.6f}")
print(f"  Difference: {abs(ll_manual - ll_direct).item():.2e}")

# ============================================================================
# Part 4: Log-Likelihood Landscape and MLE via Gradient Descent
# ============================================================================

print("\n" + "=" * 70)
print("MAXIMUM LIKELIHOOD ESTIMATION VIA GRADIENT DESCENT")
print("=" * 70)


class LogisticRegressionML(nn.Module):
    """Logistic Regression with explicit log-likelihood tracking."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def log_likelihood(self, x, y):
        """Compute log-likelihood under current parameters."""
        z = self.linear(x)
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
ml_model = LogisticRegressionML(2)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.5)

num_epochs = 200
history = {"log_likelihood": [], "nll_loss": [], "accuracy": []}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    predictions = ml_model(X_train)
    nll_loss = criterion(predictions, y_train)

    with torch.no_grad():
        ll = ml_model.log_likelihood(X_train, y_train)
        acc = ((predictions >= 0.5).float() == y_train).float().mean()

    optimizer.zero_grad()
    nll_loss.backward()
    optimizer.step()

    history["log_likelihood"].append(ll.item())
    history["nll_loss"].append(nll_loss.item())
    history["accuracy"].append(acc.item())

    if (epoch + 1) % 40 == 0:
        print(
            f"Epoch {epoch+1:3d}: Log-Likelihood = {ll.item():8.3f}, "
            f"NLL Loss = {nll_loss.item():.4f}, Accuracy = {acc.item():.4f}"
        )

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["log_likelihood"], "b-", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Log-Likelihood", fontsize=11)
axes[0].set_title("Log-Likelihood (Maximized)", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["nll_loss"], "r-", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("BCE Loss", fontsize=11)
axes[1].set_title("BCE Loss = Negative Log-Likelihood (Minimized)", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)

axes[2].plot(history["accuracy"], "g-", linewidth=2)
axes[2].set_xlabel("Epoch", fontsize=11)
axes[2].set_ylabel("Accuracy", fontsize=11)
axes[2].set_title("Training Accuracy", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("binary_classification_training.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Visualization saved!")
```

---

## Why the GLM Framework Matters

1. **Unified Theory**: GLMs provide a common language for regression (Gaussian), classification (Bernoulli), and count data (Poisson) problems

2. **Natural Constraints**: The link function automatically maps unbounded linear predictors to the appropriate range for each distribution

3. **Interpretation**: The logit link provides interpretable coefficients in terms of log-odds ratios (see [Sigmoid Function](sigmoid.md))

4. **Principled Inference**: GLM theory provides asymptotic results for confidence intervals and hypothesis tests

5. **Connection to Neural Networks**: Logistic regression is the simplest neural network — a single linear layer followed by sigmoid activation, trained with BCE loss. This perspective bridges classical statistics and deep learning

---

## Exercises

### Conceptual

1. **Prove** that the Bernoulli variance $p(1-p)$ is maximized at $p=0.5$. What is the significance of this result for classification?

2. **Explain** why the logit function is called the "canonical" link for the Bernoulli distribution. What happens if we use a different link function (e.g., probit)?

3. **Derive** the expected log-likelihood under the true data distribution and show that it equals the negative cross-entropy between the true and predicted distributions.

4. **Prove** that the MLE for logistic regression exists and is unique when the data is not perfectly separable.

### Computational

5. Implement a probit regression model (using the normal CDF as the inverse link) and compare it to logistic regression on synthetic data.

6. Create a visualization showing how the per-sample log-likelihood varies across the feature space.

---

## Summary

| Concept | Formula | Key Insight |
|---------|---------|-------------|
| Bernoulli PMF | $p^y(1-p)^{1-y}$ | Probability of single observation |
| Natural parameter | $\eta = \log\frac{p}{1-p}$ | Log-odds as canonical parameter |
| GLM link | $g(p) = \text{logit}(p)$ | Connects mean to linear predictor |
| Likelihood | $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$ | Probability of all data |
| Log-likelihood | $\sum_i [y_i z_i - \log(1+e^{z_i})]$ | Efficient computation form |
| BCE Loss | $-\frac{1}{n}\ell(\boldsymbol{\beta})$ | Negative average log-likelihood |

The GLM framework provides the theoretical foundation for understanding logistic regression not as an arbitrary choice, but as the natural model for binary classification when we assume a Bernoulli response and linear effects of features on the log-odds scale.

---

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- Agresti, A. (2015). *Foundations of Linear and Generalized Linear Models*. Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.
