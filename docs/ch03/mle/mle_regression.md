# MLE for Regression

## Introduction

Regression loss functions are negative log-likelihoods under specific noise assumptions. This section makes the connection precise: **MSE assumes Gaussian noise**, **MAE assumes Laplace noise**, and **heteroscedastic models learn the noise itself**.

!!! success "Key Insight"
    When you train a neural network with MSE loss, you are performing MLE under the assumption that targets follow a Gaussian distribution centered on the model's prediction.

## MSE as Gaussian Negative Log-Likelihood

### The Probabilistic Model

Assume the target follows a Gaussian distribution:

$$
y | x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

where $f_\theta(x)$ is the model's prediction (e.g., a neural network output) and $\sigma^2$ is a fixed noise variance.

### Derivation

**Likelihood** of a single observation:

$$
p(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f_\theta(x))^2}{2\sigma^2}\right)
$$

**Log-likelihood**:

$$
\log p(y | x, \theta) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f_\theta(x))^2}{2\sigma^2}
$$

**Negative log-likelihood** over $n$ observations:

$$
-\sum_{i=1}^{n} \log p(y_i | x_i, \theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2
$$

Since $\sigma$ is fixed, the first term is constant with respect to $\theta$. Minimizing NLL is therefore equivalent to minimizing:

$$
\boxed{\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2}
$$

### Implicit Assumptions of MSE

MSE loss implicitly assumes:

- **Homoscedastic Gaussian noise**: Constant variance $\sigma^2$ for all inputs
- **Independent errors**: Each observation's noise is independent
- **Mean prediction**: The model predicts the **conditional mean** $\mathbb{E}[Y|X]$

When these assumptions are violated (e.g., heteroscedastic data, heavy-tailed noise, outliers), MSE may not be the best choice.

### Gradient Equivalence

The gradients of MSE and Gaussian NLL with respect to $\theta$ are proportional:

$$
\nabla_\theta \mathcal{L}_{\text{MSE}} = \frac{2}{n}\sum_{i=1}^{n}(f_\theta(x_i) - y_i)\nabla_\theta f_\theta(x_i)
$$

$$
\nabla_\theta \mathcal{L}_{\text{NLL}} = \frac{1}{n\sigma^2}\sum_{i=1}^{n}(f_\theta(x_i) - y_i)\nabla_\theta f_\theta(x_i)
$$

They differ only by a constant factor $2\sigma^2$, so optimization follows the same trajectory.

## MAE as Laplace Negative Log-Likelihood

### The Probabilistic Model

Assume the target follows a Laplace distribution:

$$
y | x \sim \text{Laplace}(f_\theta(x), b)
$$

**PDF**:

$$
p(y | x, \theta) = \frac{1}{2b}\exp\left(-\frac{|y - f_\theta(x)|}{b}\right)
$$

### Derivation

**Negative log-likelihood**:

$$
-\log p(y | x, \theta) = \log(2b) + \frac{|y - f_\theta(x)|}{b}
$$

For fixed $b$, minimizing NLL gives the **Mean Absolute Error**:

$$
\boxed{\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - f_\theta(x_i)|}
$$

### MAE vs. MSE: Distributional Perspective

| Property | MSE | MAE |
|----------|-----|-----|
| Noise model | Gaussian | Laplace |
| Optimal prediction | Conditional mean $\mathbb{E}[Y\|X]$ | Conditional median |
| Tail behavior | Light tails | Heavier tails |
| Outlier robustness | Sensitive (squared penalty) | Robust (linear penalty) |
| Gradient at zero | Smooth | Non-differentiable |

!!! note "Robustness Intuition"
    The Laplace distribution has heavier tails than the Gaussian, meaning it assigns higher probability to extreme observations. This makes MAE naturally more tolerant of outliers — the linear penalty doesn't amplify large errors the way squaring does.

## Heteroscedastic Regression: Learning Variance

### Motivation

Standard regression with MSE predicts only the conditional mean. But real data often has **input-dependent noise** — for example, predicting stock returns is harder for volatile stocks. A heteroscedastic model predicts both mean and variance.

### Model

$$
y | x \sim \mathcal{N}(\mu_\theta(x), \sigma_\theta(x)^2)
$$

The network has two output heads: one for $\mu_\theta(x)$ and one for $\log \sigma_\theta(x)^2$ (log-variance for numerical stability).

### Loss Function

The **negative log-likelihood** for heteroscedastic Gaussian regression is:

$$
\boxed{\mathcal{L}_{\text{hetero}} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{1}{2}\log \sigma_\theta(x_i)^2 + \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma_\theta(x_i)^2}\right]}
$$

The first term penalizes large predicted variance (prevents trivially setting $\sigma \to \infty$). The second term weights the squared error by the inverse predicted variance — regions where the model is confident are penalized more harshly for errors.

!!! tip "Practical Benefits"
    Heteroscedastic regression naturally handles: **uncertainty estimation** (output predictive variance), **heteroscedastic noise** (different variance for different inputs), and **confidence-aware predictions** (the model knows where it's uncertain).

## PyTorch Implementation

### MSE vs. Gaussian NLL Equivalence

```python
import torch
import torch.nn as nn
import numpy as np

def demonstrate_mse_nll_equivalence():
    """Show that MSE and Gaussian NLL give proportional gradients."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.rand(n, 1) * 10
    true_w, true_b = 3.0, 2.0
    y = true_w * x + true_b + torch.randn(n, 1) * 1.0
    
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # MSE gradient
    pred = w * x + b
    mse_loss = torch.mean((y - pred)**2)
    mse_loss.backward()
    mse_grad_w = w.grad.clone()
    w.grad.zero_(); b.grad.zero_()
    
    # Gaussian NLL gradient (sigma=1)
    sigma = 1.0
    pred = w * x + b
    nll_loss = torch.mean(0.5 * torch.log(torch.tensor(2 * np.pi * sigma**2)) + 
                          (y - pred)**2 / (2 * sigma**2))
    nll_loss.backward()
    nll_grad_w = w.grad.clone()
    
    print("MSE vs Gaussian NLL Equivalence")
    print(f"MSE gradient w.r.t w: {mse_grad_w.item():.6f}")
    print(f"NLL gradient w.r.t w: {nll_grad_w.item():.6f}")
    print(f"Ratio (should be 2σ² = 2): {mse_grad_w.item() / nll_grad_w.item():.4f}")
```

### MAE vs. MSE Robustness

```python
def mae_vs_mse_robustness():
    """Demonstrate robustness difference between MAE and MSE with outliers."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.linspace(0, 10, n).reshape(-1, 1)
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    
    # Add large outliers
    y = y_clean.clone()
    outlier_idx = [10, 30, 50, 70, 90]
    y[outlier_idx] += 15
    
    # Train with MSE
    model_mse = nn.Linear(1, 1)
    optimizer_mse = torch.optim.Adam(model_mse.parameters(), lr=0.1)
    for _ in range(1000):
        loss = nn.MSELoss()(model_mse(x), y)
        optimizer_mse.zero_grad(); loss.backward(); optimizer_mse.step()
    
    # Train with MAE
    model_mae = nn.Linear(1, 1)
    optimizer_mae = torch.optim.Adam(model_mae.parameters(), lr=0.1)
    for _ in range(1000):
        loss = nn.L1Loss()(model_mae(x), y)
        optimizer_mae.zero_grad(); loss.backward(); optimizer_mae.step()
    
    print("MAE vs MSE with Outliers (true w=2.0, b=1.0)")
    print(f"MSE fit: w={model_mse.weight.item():.4f}, b={model_mse.bias.item():.4f}")
    print(f"MAE fit: w={model_mae.weight.item():.4f}, b={model_mae.bias.item():.4f}")
```

### Heteroscedastic Regression

```python
class HeteroscedasticNet(nn.Module):
    """Neural network that predicts both mean and log-variance."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var

def heteroscedastic_nll(y: torch.Tensor, 
                        mean: torch.Tensor, 
                        log_var: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for heteroscedastic Gaussian.
    
    NLL = 0.5 * [log(2π) + log_var + (y - mean)² / exp(log_var)]
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (y - mean)**2 / torch.exp(log_var))

def train_heteroscedastic():
    """Train heteroscedastic model on data with varying noise."""
    torch.manual_seed(42)
    
    # Generate data with increasing noise
    n = 500
    x = torch.rand(n, 1) * 10
    noise_std = 0.5 + 0.3 * x  # Noise increases with x
    y = 2 * x + 1 + torch.randn(n, 1) * noise_std
    
    model = HeteroscedasticNet(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        mean, log_var = model(x)
        loss = heteroscedastic_nll(y, mean, log_var).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, NLL: {loss.item():.4f}")
    
    return model
```

### Complete Regression Training with MLE Perspective

```python
def train_regression_mle_perspective():
    """Complete example showing MLE interpretation of regression training."""
    torch.manual_seed(42)
    
    # Generate data
    n = 200
    X = torch.randn(n, 5)
    true_w = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5])
    true_b = 2.0
    sigma = 0.5
    y = X @ true_w + true_b + torch.randn(n) * sigma
    
    # Model
    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # MSE Loss = Gaussian NLL (up to constant)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Neural Network (MLE with Gaussian likelihood)")
    print("-" * 50)
    
    for epoch in range(500):
        pred = model(X).squeeze()
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                residuals = y - model(X).squeeze()
                estimated_sigma = residuals.std().item()
            
            print(f"Epoch {epoch+1}: MSE = {loss.item():.4f}, "
                  f"Est. σ = {estimated_sigma:.4f} (true: {sigma:.4f})")
```

## Exercises

1. **Huber Loss**: Show that Huber loss corresponds to MLE for a specific mixture of Gaussian and Laplace distributions. What are the mixture proportions as a function of the Huber $\delta$ parameter?

2. **Poisson Regression**: Derive the loss function for Poisson regression where $y|x \sim \text{Poisson}(\exp(f_\theta(x)))$ and implement it in PyTorch.

3. **Multivariate Heteroscedastic**: Extend the heteroscedastic regression to predict a full covariance matrix for multivariate outputs $\mathbf{y} \in \mathbb{R}^d$.

4. **Beta Regression**: Design a loss function for bounded outputs in $[0, 1]$ using the Beta distribution.

5. **Robustness Comparison**: Implement a simulation comparing MSE, MAE, and Huber loss on data with varying outlier fractions (0%, 5%, 10%, 20%) and measure parameter recovery.

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 3
- Nix, D. A. & Weigend, A. S. (1994). "Estimating the mean and variance of the target probability distribution." *ICNN*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 5
