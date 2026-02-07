# Probabilistic Interpretation

## Introduction

This section presents the **unified probabilistic view** of deep learning: every loss function is a negative log-likelihood, every regularizer is a prior, and training is Bayesian inference in disguise. Understanding this framework guides loss function selection, regularization design, and model interpretation.

!!! abstract "The Central Thesis"
    | Training Objective | Probabilistic Interpretation |
    |:-------------------|:-----------------------------|
    | Minimize MSE | MLE with Gaussian noise |
    | Minimize Cross-Entropy | MLE with categorical likelihood |
    | Add L2 Regularization | MAP with Gaussian prior |
    | Add L1 Regularization | MAP with Laplace prior |
    | Add Dropout | Approximate Bayesian inference |
    | Predict mean and variance | Heteroscedastic Gaussian MLE |

## The Fundamental Relationship

Given data $\{(x_i, y_i)\}_{i=1}^n$ and a probabilistic model $p(y|x, \theta)$:

**Maximum Likelihood**: $\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} p(y_i | x_i, \theta)$

**Equivalent Loss Minimization**: $\hat{\theta} = \arg\min_\theta \left[-\sum_{i=1}^{n} \log p(y_i | x_i, \theta)\right]$

The **loss function** is the average negative log-likelihood:

$$
\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i | x_i, \theta)
$$

Every choice of $p(y|x, \theta)$ yields a different loss function. Conversely, every loss function implicitly assumes a distributional model for the targets.

### Complete Loss–Distribution Mapping

| Loss Function | Distribution | Model Output | Use Case |
|:--------------|:-------------|:-------------|:---------|
| MSE | $\mathcal{N}(\mu, \sigma^2)$ | Mean $\mu$ | Regression |
| MAE | Laplace$(m, b)$ | Median $m$ | Robust regression |
| BCE | Bernoulli$(p)$ | Probability $p$ | Binary classification |
| Cross-Entropy | Categorical$(\mathbf{p})$ | Probabilities $\mathbf{p}$ | Multi-class classification |
| Poisson Loss | Poisson$(\lambda)$ | Rate $\lambda$ | Count data |
| Huber Loss | Gaussian–Laplace mixture | Robust mean | Outlier-robust regression |

## KL Divergence and Loss Functions

### Relationship to Cross-Entropy

Cross-entropy decomposes as:

$$
H(p, q) = H(p) + D_{\text{KL}}(p \| q)
$$

where $H(p, q) = -\mathbb{E}_p[\log q]$ is cross-entropy, $H(p) = -\mathbb{E}_p[\log p]$ is entropy of the true distribution, and $D_{\text{KL}}(p \| q) = \mathbb{E}_p[\log \frac{p}{q}]$ is KL divergence.

Since $H(p)$ is constant with respect to model parameters:

$$
\text{Minimizing Cross-Entropy} \equiv \text{Minimizing KL Divergence}
$$

### MLE Minimizes KL Divergence

The MLE objective:

$$
\hat{\theta} = \arg\min_\theta \left[-\frac{1}{n}\sum_{i=1}^n \log p(y_i | x_i, \theta)\right]
$$

converges (as $n \to \infty$) to minimizing:

$$
D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{p_{\text{data}}}\left[\log \frac{p_{\text{data}}}{p_\theta}\right]
$$

This provides a deep justification for MLE: it finds the model distribution closest to the true data distribution in the KL sense.

## Regularization as Prior: MAP Estimation

### From MLE to MAP

Maximum A Posteriori (MAP) estimation adds a prior $p(\theta)$ to MLE via Bayes' theorem:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, p(\theta | \text{data}) = \arg\max_\theta \left[\log p(\text{data} | \theta) + \log p(\theta)\right]
$$

The first term is the log-likelihood (data fit); the second is the log-prior (regularization). MAP estimation is the bridge between frequentist MLE and Bayesian inference.

### L2 Regularization = Gaussian Prior

If $\theta \sim \mathcal{N}(0, \tau^2 I)$:

$$
\log p(\theta) = -\frac{\|\theta\|^2}{2\tau^2} + \text{const}
$$

The MAP objective becomes:

$$
\mathcal{L}_{\text{MAP}} = \underbrace{-\log p(\text{data} | \theta)}_{\text{NLL (data fit)}} + \underbrace{\frac{\lambda}{2}\|\theta\|^2}_{\text{L2 penalty}}
$$

where $\lambda = 1/\tau^2$. Larger $\lambda$ (smaller $\tau^2$) corresponds to a tighter prior — stronger regularization that pulls parameters toward zero.

### L1 Regularization = Laplace Prior

If $\theta \sim \text{Laplace}(0, b)$:

$$
\log p(\theta) = -\frac{\|\theta\|_1}{b} + \text{const}
$$

The MAP objective becomes:

$$
\mathcal{L}_{\text{MAP}} = -\log p(\text{data} | \theta) + \lambda \|\theta\|_1
$$

The Laplace prior has a sharp peak at zero (unlike the smooth Gaussian), which encourages **sparsity** — many parameters are driven exactly to zero.

### Geometric Intuition

The L1 prior (diamond-shaped contours) tends to place the MAP estimate at a vertex where some coordinates are zero. The L2 prior (circular contours) shrinks all parameters uniformly but rarely produces exact zeros. This is why L1 (Lasso) is preferred for feature selection.

## Designing Custom Loss Functions

### The Recipe

To design a loss function for a specific problem:

1. **Specify the distributional model**: What distribution does $y|x$ follow?
2. **Write the log-likelihood**: $\log p(y|x, \theta)$
3. **Negate and average**: $\mathcal{L} = -\frac{1}{n}\sum \log p(y_i | x_i, \theta)$
4. **Add regularization** (optional): Choose a prior $p(\theta)$

This principled approach ensures the loss function is theoretically grounded and avoids ad hoc choices.

### Example: Poisson Regression

For count data $y \in \{0, 1, 2, \ldots\}$, model $y | x \sim \text{Poisson}(\exp(f_\theta(x)))$:

$$
\log p(y|x, \theta) = y \cdot f_\theta(x) - \exp(f_\theta(x)) - \log(y!)
$$

The loss (dropping the constant $\log(y!)$):

$$
\mathcal{L}_{\text{Poisson}} = \frac{1}{n}\sum_{i=1}^{n}\left[\exp(f_\theta(x_i)) - y_i \cdot f_\theta(x_i)\right]
$$

## PyTorch Implementation

### Regularization as Prior

```python
import torch
import torch.nn as nn
import numpy as np

def regularization_as_prior_demo():
    """Demonstrate that L1/L2 regularization corresponds to Bayesian priors."""
    torch.manual_seed(42)
    
    # Sparse true parameters (only 3 of 10 are nonzero)
    true_w = torch.tensor([3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
    
    n = 50
    X = torch.randn(n, 10)
    y = X @ true_w + torch.randn(n) * 0.5
    
    results = {}
    
    for reg_type, reg_strength in [('none', 0), ('L2', 0.1), ('L1', 0.1)]:
        w = torch.randn(10, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.1)
        
        for _ in range(2000):
            pred = X @ w
            mse = torch.mean((y - pred)**2)
            
            if reg_type == 'L2':
                loss = mse + reg_strength * torch.sum(w**2)
            elif reg_type == 'L1':
                loss = mse + reg_strength * torch.sum(torch.abs(w))
            else:
                loss = mse
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        results[reg_type] = w.detach().clone()
    
    print("Regularization as Bayesian Prior")
    print("=" * 60)
    print(f"{'Index':>6} {'True':>10} {'No Reg':>10} {'L2 (Gauss)':>10} {'L1 (Lapl)':>10}")
    print("-" * 60)
    
    for i in range(10):
        print(f"{i:>6} {true_w[i]:>10.4f} {results['none'][i]:>10.4f} "
              f"{results['L2'][i]:>10.4f} {results['L1'][i]:>10.4f}")
    
    print("-" * 60)
    print("L1 encourages sparsity (zeros), L2 encourages small values")
```

### Custom Loss: Poisson Regression

```python
def poisson_regression_demo():
    """Implement Poisson regression as MLE for count data."""
    torch.manual_seed(42)
    
    # Generate count data
    n = 500
    X = torch.randn(n, 3)
    true_w = torch.tensor([0.5, -0.3, 0.8])
    log_rate = X @ true_w + 1.0
    y = torch.poisson(torch.exp(log_rate))
    
    # Model
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        log_lambda = model(X).squeeze()
        # Poisson NLL: exp(f(x)) - y * f(x)
        loss = torch.mean(torch.exp(log_lambda) - y * log_lambda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: Poisson NLL = {loss.item():.4f}")
    
    print(f"\nTrue weights:      {true_w.tolist()}")
    print(f"Estimated weights: {model.weight.data.squeeze().tolist()}")
```

### Comparing Loss Functions on Synthetic Data

```python
def loss_function_comparison():
    """
    Compare MSE, MAE, and Huber loss on regression with outliers.
    Demonstrates that loss choice = distributional assumption.
    """
    torch.manual_seed(42)
    
    n = 200
    x = torch.rand(n, 1) * 10
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    
    # Add 10% outliers
    n_outliers = n // 10
    outlier_idx = torch.randperm(n)[:n_outliers]
    y = y_clean.clone()
    y[outlier_idx] += torch.randn(n_outliers, 1) * 10
    
    losses = {
        'MSE (Gaussian)': nn.MSELoss(),
        'MAE (Laplace)': nn.L1Loss(),
        'Huber (Mixture)': nn.HuberLoss(delta=1.0),
    }
    
    print("Loss Function Comparison (true w=2.0, b=1.0, 10% outliers)")
    print("-" * 55)
    
    for name, criterion in losses.items():
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        for _ in range(2000):
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        w = model.weight.item()
        b = model.bias.item()
        print(f"{name:>20}: w={w:.4f}, b={b:.4f}")
```

## Key Takeaways

The probabilistic interpretation of deep learning provides several practical benefits:

1. **Loss function selection**: Choose the loss by asking "what distribution does my target follow?" rather than guessing.

2. **Regularization design**: Choose regularization by asking "what prior beliefs do I have about the parameters?"

3. **Uncertainty quantification**: The probabilistic model gives not just point predictions but distributional predictions (means, variances, confidence intervals).

4. **Custom loss design**: Any new problem can be addressed by specifying a distributional model and deriving the corresponding NLL.

5. **Principled model comparison**: Different models can be compared via their likelihoods on held-out data.

## Exercises

1. **Derive** the loss function for a model that predicts counts with overdispersion using the Negative Binomial distribution.

2. **Implement** a multi-task learning loss where one output is regression (Gaussian) and another is classification (categorical), weighted by task-specific uncertainty following Kendall & Gal (2018).

3. **Show** that dropout training approximates variational inference and explain what posterior distribution it approximates.

4. **Design** a custom loss for predicting circular data (e.g., wind direction) using the von Mises distribution.

5. **Compare** MLE, MAP with L2, and MAP with L1 on a high-dimensional sparse regression problem ($p = 1000$, $n = 100$, 10 true nonzero features). Evaluate parameter recovery and prediction error.

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 4
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 5
- Kendall, A. & Gal, Y. (2018). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 5
