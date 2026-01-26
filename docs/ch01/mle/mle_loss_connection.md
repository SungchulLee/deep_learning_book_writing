# Connection Between MLE and Loss Functions

## Introduction

One of the most important insights in machine learning is that **standard loss functions are negative log-likelihoods under specific probabilistic assumptions**. This connection provides deep theoretical justification for commonly used loss functions and guides the design of new ones.

!!! success "Key Insight"
    $$\text{Minimizing Loss} \equiv \text{Maximizing Likelihood}$$
    
    When you train a neural network with cross-entropy or MSE loss, you are performing maximum likelihood estimation.

## The Fundamental Relationship

### From MLE to Loss Minimization

Given data $\{(x_i, y_i)\}_{i=1}^n$ and a model $p(y|x, \theta)$:

**Maximum Likelihood**:
$$
\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} p(y_i | x_i, \theta)
$$

**Equivalent Loss Minimization**:
$$
\hat{\theta} = \arg\min_\theta \left[-\sum_{i=1}^{n} \log p(y_i | x_i, \theta)\right]
$$

The **loss function** is simply:
$$
\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i | x_i, \theta)
$$

This is the **negative log-likelihood (NLL)** averaged over the dataset.

## Mean Squared Error and Gaussian Likelihood

### The Probabilistic Model

Assume the target follows a Gaussian distribution:

$$
y | x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

where $f_\theta(x)$ is your model's prediction (e.g., neural network output).

### Deriving MSE from MLE

**Likelihood**:
$$
p(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f_\theta(x))^2}{2\sigma^2}\right)
$$

**Log-Likelihood**:
$$
\log p(y | x, \theta) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f_\theta(x))^2}{2\sigma^2}
$$

**Negative Log-Likelihood**:
$$
-\log p(y | x, \theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - f_\theta(x))^2}{2\sigma^2}
$$

For fixed $\sigma$, minimizing NLL is equivalent to minimizing:
$$
\boxed{\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2}
$$

!!! info "Interpretation"
    MSE loss implicitly assumes:
    
    - Targets have **homoscedastic Gaussian noise** (constant variance)
    - Errors are **independent and identically distributed**
    - The model predicts the **conditional mean** $\mathbb{E}[Y|X]$

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_mse_nll_equivalence():
    """Show that MSE and Gaussian NLL give same gradients."""
    torch.manual_seed(42)
    
    # Generate data
    n = 100
    x = torch.rand(n, 1) * 10
    true_w, true_b = 3.0, 2.0
    y = true_w * x + true_b + torch.randn(n, 1) * 1.0
    
    # Simple linear model
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # MSE gradient
    pred = w * x + b
    mse_loss = torch.mean((y - pred)**2)
    mse_loss.backward()
    mse_grad_w = w.grad.clone()
    w.grad.zero_()
    b.grad.zero_()
    
    # Gaussian NLL gradient (assuming sigma=1)
    sigma = 1.0
    pred = w * x + b
    nll_loss = torch.mean(0.5 * torch.log(torch.tensor(2 * np.pi * sigma**2)) + 
                          (y - pred)**2 / (2 * sigma**2))
    nll_loss.backward()
    nll_grad_w = w.grad.clone()
    
    print("MSE vs Gaussian NLL Equivalence")
    print("-" * 40)
    print(f"MSE gradient w.r.t w: {mse_grad_w.item():.6f}")
    print(f"NLL gradient w.r.t w: {nll_grad_w.item():.6f}")
    print(f"Ratio (should be 2σ² = 2): {mse_grad_w.item() / nll_grad_w.item():.4f}")
```

## Cross-Entropy and Categorical Likelihood

### Binary Classification

**Probabilistic Model**: Bernoulli distribution
$$
y | x \sim \text{Bernoulli}(\sigma(f_\theta(x)))
$$

where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function.

**Likelihood**:
$$
p(y | x, \theta) = \sigma(f_\theta(x))^y \cdot (1 - \sigma(f_\theta(x)))^{1-y}
$$

**Negative Log-Likelihood**:
$$
-\log p(y | x, \theta) = -y \log \sigma(f_\theta(x)) - (1-y) \log(1 - \sigma(f_\theta(x)))
$$

This is the **Binary Cross-Entropy (BCE)** loss:
$$
\boxed{\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]}
$$

```python
def bce_from_nll():
    """Derive BCE from Bernoulli negative log-likelihood."""
    torch.manual_seed(42)
    
    # Binary classification setup
    n = 100
    x = torch.randn(n, 2)
    true_w = torch.tensor([2.0, -1.0])
    y = (x @ true_w > 0).float()
    
    # Model prediction
    w = torch.randn(2, requires_grad=True)
    logits = x @ w
    probs = torch.sigmoid(logits)
    
    # Manual BCE computation
    eps = 1e-7
    manual_bce = -torch.mean(y * torch.log(probs + eps) + 
                              (1 - y) * torch.log(1 - probs + eps))
    
    # PyTorch BCE
    pytorch_bce = nn.BCELoss()(probs, y)
    
    # PyTorch BCE with logits (numerically stable)
    pytorch_bce_logits = nn.BCEWithLogitsLoss()(logits, y)
    
    print("Binary Cross-Entropy Verification")
    print("-" * 40)
    print(f"Manual BCE: {manual_bce.item():.6f}")
    print(f"PyTorch BCE: {pytorch_bce.item():.6f}")
    print(f"BCE with Logits: {pytorch_bce_logits.item():.6f}")
```

### Multi-class Classification

**Probabilistic Model**: Categorical distribution
$$
y | x \sim \text{Categorical}(\text{softmax}(f_\theta(x)))
$$

**Softmax Function**:
$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Negative Log-Likelihood** (for one-hot encoded $y$):
$$
-\log p(y | x, \theta) = -\sum_{k=1}^{K} y_k \log \text{softmax}(f_\theta(x))_k
$$

This is the **Categorical Cross-Entropy** loss:
$$
\boxed{\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}}
$$

For hard labels (single correct class $c_i$):
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{i, c_i}
$$

```python
def cross_entropy_from_nll():
    """Derive cross-entropy from categorical NLL."""
    torch.manual_seed(42)
    
    # Multi-class setup
    n = 100
    K = 5  # Number of classes
    x = torch.randn(n, 10)
    y = torch.randint(0, K, (n,))  # Class indices
    
    # Model (simple linear)
    W = torch.randn(10, K, requires_grad=True)
    logits = x @ W  # Shape: (n, K)
    
    # Manual softmax and cross-entropy
    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])  # Numerical stability
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    
    # NLL: -log(prob of correct class)
    manual_ce = -torch.mean(torch.log(probs[range(n), y] + 1e-7))
    
    # PyTorch CrossEntropyLoss (combines LogSoftmax + NLLLoss)
    pytorch_ce = nn.CrossEntropyLoss()(logits, y)
    
    print("Cross-Entropy Verification")
    print("-" * 40)
    print(f"Manual CE: {manual_ce.item():.6f}")
    print(f"PyTorch CE: {pytorch_ce.item():.6f}")
```

## Complete Mapping: Loss Functions and Distributions

| Loss Function | Distribution | Model Output | Use Case |
|---------------|--------------|--------------|----------|
| MSE | $\mathcal{N}(\mu, \sigma^2)$ | Mean $\mu$ | Regression |
| MAE | Laplace$(m, b)$ | Median $m$ | Robust regression |
| BCE | Bernoulli$(p)$ | Probability $p$ | Binary classification |
| Cross-Entropy | Categorical$(\mathbf{p})$ | Probabilities $\mathbf{p}$ | Multi-class classification |
| Poisson Loss | Poisson$(\lambda)$ | Rate $\lambda$ | Count data |
| Huber Loss | Mixture model | Robust mean | Outlier-robust regression |

## Mean Absolute Error and Laplace Distribution

### The Probabilistic Model

Assume the target follows a Laplace distribution:
$$
y | x \sim \text{Laplace}(f_\theta(x), b)
$$

**PDF**:
$$
p(y | x, \theta) = \frac{1}{2b}\exp\left(-\frac{|y - f_\theta(x)|}{b}\right)
$$

### Deriving MAE from MLE

**Negative Log-Likelihood**:
$$
-\log p(y | x, \theta) = \log(2b) + \frac{|y - f_\theta(x)|}{b}
$$

For fixed $b$, minimizing NLL gives the **Mean Absolute Error**:
$$
\boxed{\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - f_\theta(x_i)|}
$$

!!! note "MAE vs MSE"
    - **MSE** assumes Gaussian noise → minimizes $\mathbb{E}[Y|X]$ (mean)
    - **MAE** assumes Laplace noise → minimizes median of $Y|X$
    - **MAE is more robust** to outliers (heavier tails in Laplace)

```python
def mae_vs_mse_robustness():
    """Demonstrate robustness difference between MAE and MSE."""
    torch.manual_seed(42)
    
    # Clean data with outliers
    n = 100
    x = torch.linspace(0, 10, n).reshape(-1, 1)
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    
    # Add outliers
    y = y_clean.clone()
    outlier_idx = [10, 30, 50, 70, 90]
    y[outlier_idx] += 15  # Large outliers
    
    # Train with MSE
    model_mse = nn.Linear(1, 1)
    optimizer_mse = torch.optim.Adam(model_mse.parameters(), lr=0.1)
    
    for _ in range(1000):
        pred = model_mse(x)
        loss = nn.MSELoss()(pred, y)
        optimizer_mse.zero_grad()
        loss.backward()
        optimizer_mse.step()
    
    # Train with MAE
    model_mae = nn.Linear(1, 1)
    optimizer_mae = torch.optim.Adam(model_mae.parameters(), lr=0.1)
    
    for _ in range(1000):
        pred = model_mae(x)
        loss = nn.L1Loss()(pred, y)
        optimizer_mae.zero_grad()
        loss.backward()
        optimizer_mae.step()
    
    print("\nMAE vs MSE with Outliers")
    print("-" * 40)
    print(f"True parameters: w=2.0, b=1.0")
    print(f"MSE fit: w={model_mse.weight.item():.4f}, b={model_mse.bias.item():.4f}")
    print(f"MAE fit: w={model_mae.weight.item():.4f}, b={model_mae.bias.item():.4f}")
    print("MAE is more robust to outliers!")
```

## Heteroscedastic Regression: Learning Variance

### Model with Learned Variance

Instead of predicting just the mean, predict both mean and variance:
$$
y | x \sim \mathcal{N}(\mu_\theta(x), \sigma_\theta(x)^2)
$$

**Negative Log-Likelihood**:
$$
-\log p(y | x, \theta) = \frac{1}{2}\log(2\pi\sigma_\theta(x)^2) + \frac{(y - \mu_\theta(x))^2}{2\sigma_\theta(x)^2}
$$

This loss naturally handles:
- **Uncertainty estimation**: Outputs predictive variance
- **Heteroscedastic noise**: Different variance for different inputs

```python
class HeteroscedasticNet(nn.Module):
    """Neural network that predicts both mean and variance."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)  # Log variance for stability
    
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
    
    NLL = 0.5 * log(2π) + 0.5 * log_var + 0.5 * (y - mean)² / exp(log_var)
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (y - mean)**2 / torch.exp(log_var))

def train_heteroscedastic():
    """Train heteroscedastic model."""
    torch.manual_seed(42)
    
    # Generate data with varying noise
    n = 500
    x = torch.rand(n, 1) * 10
    noise_std = 0.5 + 0.3 * x  # Increasing noise with x
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

## KL Divergence and Loss Functions

### Relationship to Cross-Entropy

Cross-entropy can be decomposed as:
$$
H(p, q) = H(p) + D_{\text{KL}}(p \| q)
$$

Where:
- $H(p, q) = -\mathbb{E}_p[\log q]$ is cross-entropy
- $H(p) = -\mathbb{E}_p[\log p]$ is entropy of true distribution
- $D_{\text{KL}}(p \| q) = \mathbb{E}_p[\log \frac{p}{q}]$ is KL divergence

Since $H(p)$ is constant w.r.t. model parameters:
$$
\text{Minimizing Cross-Entropy} \equiv \text{Minimizing KL Divergence}
$$

### MLE Minimizes KL Divergence

The MLE objective:
$$
\hat{\theta} = \arg\min_\theta \left[-\frac{1}{n}\sum_{i=1}^n \log p(y_i | x_i, \theta)\right]
$$

Converges to minimizing:
$$
D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{p_{\text{data}}}\left[\log \frac{p_{\text{data}}}{p_\theta}\right]
$$

## Regularization as Prior: MAP Estimation

### Maximum A Posteriori (MAP)

Adding a prior $p(\theta)$ to MLE gives MAP estimation:
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[p(\theta | \text{data})\right] = \arg\max_\theta \left[\log p(\text{data} | \theta) + \log p(\theta)\right]
$$

### L2 Regularization = Gaussian Prior

If $\theta \sim \mathcal{N}(0, \tau^2 I)$:
$$
\log p(\theta) = -\frac{\|\theta\|^2}{2\tau^2} + \text{const}
$$

**MAP Loss**:
$$
\mathcal{L}_{\text{MAP}} = -\log p(\text{data} | \theta) + \frac{\lambda}{2}\|\theta\|^2
$$

where $\lambda = 1/\tau^2$.

### L1 Regularization = Laplace Prior

If $\theta \sim \text{Laplace}(0, b)$:
$$
\log p(\theta) = -\frac{\|\theta\|_1}{b} + \text{const}
$$

**MAP Loss**:
$$
\mathcal{L}_{\text{MAP}} = -\log p(\text{data} | \theta) + \lambda \|\theta\|_1
$$

This encourages **sparsity** (many parameters near zero).

```python
def regularization_as_prior_demo():
    """Demonstrate that regularization corresponds to Bayesian priors."""
    torch.manual_seed(42)
    
    # Sparse true parameters
    true_w = torch.tensor([3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
    
    # Generate data
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
    
    # Compare results
    print("\nRegularization as Bayesian Prior")
    print("=" * 60)
    print(f"{'Index':>6} {'True':>10} {'No Reg':>10} {'L2 (Ridge)':>10} {'L1 (Lasso)':>10}")
    print("-" * 60)
    
    for i in range(10):
        print(f"{i:>6} {true_w[i]:>10.4f} {results['none'][i]:>10.4f} "
              f"{results['L2'][i]:>10.4f} {results['L1'][i]:>10.4f}")
    
    print("-" * 60)
    print("Note: L1 encourages sparsity (zeros), L2 encourages small values")
```

## Summary: The Probabilistic View of Deep Learning

| Training Objective | Probabilistic Interpretation |
|-------------------|------------------------------|
| Minimize MSE | MLE with Gaussian noise |
| Minimize Cross-Entropy | MLE with categorical likelihood |
| Add L2 Regularization | MAP with Gaussian prior |
| Add L1 Regularization | MAP with Laplace prior |
| Add Dropout | Approximate Bayesian inference |
| Predict mean and variance | Heteroscedastic Gaussian MLE |

!!! summary "Key Takeaways"
    1. **Every loss function has a probabilistic interpretation**
    2. **MLE justifies standard loss functions** under distributional assumptions
    3. **Regularization corresponds to priors** in Bayesian framework
    4. **Understanding the probabilistic model helps** choose appropriate losses
    5. **Custom losses can be designed** by specifying desired distributions

## Practical PyTorch Examples

### Complete Training Loop with MLE Perspective

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
        loss = criterion(pred, y)  # This is NLL (up to constant factor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            # Convert MSE to estimated sigma (MLE for noise)
            with torch.no_grad():
                residuals = y - model(X).squeeze()
                estimated_sigma = residuals.std().item()
            
            print(f"Epoch {epoch+1}: MSE = {loss.item():.4f}, "
                  f"Est. σ = {estimated_sigma:.4f} (true: {sigma:.4f})")

def train_classification_mle_perspective():
    """Complete example showing MLE interpretation of classification training."""
    torch.manual_seed(42)
    
    # Generate 3-class data
    n_per_class = 100
    X0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, 0.0])
    X1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, 0.0])
    X2 = torch.randn(n_per_class, 2) + torch.tensor([0.0, 3.0])
    
    X = torch.cat([X0, X1, X2])
    y = torch.cat([torch.zeros(n_per_class), 
                   torch.ones(n_per_class), 
                   2*torch.ones(n_per_class)]).long()
    
    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    
    # Model
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 3)  # Output: logits for 3 classes
    )
    
    # CrossEntropy = Categorical NLL
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\nTraining Classifier (MLE with Categorical likelihood)")
    print("-" * 50)
    
    for epoch in range(300):
        logits = model(X)
        loss = criterion(logits, y)  # This is NLL
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 60 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == y).float().mean().item()
            
            # NLL in nats, convert to bits
            nll_bits = loss.item() / np.log(2)
            
            print(f"Epoch {epoch+1}: NLL = {loss.item():.4f} nats "
                  f"({nll_bits:.4f} bits), Accuracy = {accuracy:.2%}")
```

## Exercises

1. **Derive** the loss function for Poisson regression (count data prediction)

2. **Implement** a multi-task learning loss where one output is regression (Gaussian) and another is classification (categorical)

3. **Show** that Huber loss corresponds to MLE for a specific mixture distribution

4. **Design** a loss function for bounded outputs in $[0, 1]$ using the Beta distribution

5. **Extend** the heteroscedastic regression to predict a full covariance matrix for multivariate outputs

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapter 5
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 5
