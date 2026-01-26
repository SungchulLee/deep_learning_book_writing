# MSE as Negative Log-Likelihood

## Overview

The Mean Squared Error (MSE) loss function, ubiquitous in regression tasks, is not an arbitrary choice but emerges directly from the principle of Maximum Likelihood Estimation under Gaussian noise assumptions. This connection provides both theoretical justification and practical insights.

## The Fundamental Connection

### From Likelihood to Loss

Starting from the Gaussian likelihood for a single observation:

$$p(y | \mathbf{x}; \mathbf{w}, b, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mathbf{w}^T\mathbf{x} - b)^2}{2\sigma^2}\right)$$

The negative log-likelihood (NLL) for $n$ observations is:

$$-\ell = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

For fixed $\sigma^2$, minimizing NLL is equivalent to minimizing:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Why This Matters

| Aspect | Implication |
|--------|-------------|
| **Theoretical Grounding** | MSE isn't arbitrary—it's the MLE under Gaussian noise |
| **Optimal Properties** | MSE estimator inherits MLE's asymptotic efficiency |
| **Uncertainty** | Enables principled uncertainty estimation |
| **Model Selection** | Connects to information criteria (AIC, BIC) |

## Mathematical Derivation

### Step-by-Step

**Step 1: Write the likelihood for all observations**

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)$$

**Step 2: Take the log**

$$\log \mathcal{L}(\theta) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Step 3: Identify the objective**

The terms $-\frac{n}{2}\log(2\pi)$ and $-\frac{n}{2}\log(\sigma^2)$ are constants w.r.t. model parameters $(\mathbf{w}, b)$.

Therefore:

$$\arg\max_{\mathbf{w}, b} \log \mathcal{L} = \arg\min_{\mathbf{w}, b} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Step 4: Scale to get MSE**

Dividing by $n$ gives MSE without changing the optimal parameters:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

## PyTorch Implementation

### MSE Loss: Three Equivalent Ways

```python
import torch
import torch.nn as nn

def demonstrate_mse_equivalence():
    """Show three equivalent ways to compute MSE in PyTorch"""
    torch.manual_seed(42)
    
    # Sample data
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.2, 2.8, 4.1, 4.9])
    
    # Method 1: Manual computation
    mse_manual = torch.mean((y_true - y_pred) ** 2)
    
    # Method 2: Using nn.MSELoss
    criterion = nn.MSELoss(reduction='mean')
    mse_pytorch = criterion(y_pred, y_true)
    
    # Method 3: Using functional API
    mse_functional = torch.nn.functional.mse_loss(y_pred, y_true)
    
    print("MSE Computation Methods:")
    print(f"  Manual:     {mse_manual.item():.6f}")
    print(f"  nn.MSELoss: {mse_pytorch.item():.6f}")
    print(f"  Functional: {mse_functional.item():.6f}")
    print(f"  All equal:  {torch.allclose(mse_manual, mse_pytorch) and torch.allclose(mse_pytorch, mse_functional)}")

demonstrate_mse_equivalence()
```

### Explicit NLL Computation

```python
import torch
import numpy as np

def negative_log_likelihood(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute negative log-likelihood under Gaussian noise
    
    NLL = (n/2)*log(2*pi*sigma^2) + (1/(2*sigma^2)) * sum((y - y_pred)^2)
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        sigma: Standard deviation of noise
    
    Returns:
        Negative log-likelihood value
    """
    n = y_true.shape[0]
    sigma_sq = sigma ** 2
    
    # Constant term
    constant_term = (n / 2) * np.log(2 * np.pi * sigma_sq)
    
    # Data-dependent term (SSE scaled)
    sse = torch.sum((y_true - y_pred) ** 2)
    data_term = sse / (2 * sigma_sq)
    
    return constant_term + data_term

def show_nll_mse_relationship():
    """Demonstrate that minimizing NLL = minimizing MSE"""
    torch.manual_seed(42)
    
    # Generate data
    n = 100
    true_w, true_b = 2.0, 1.0
    sigma = 0.5
    
    X = torch.randn(n, 1)
    y = true_w * X + true_b + sigma * torch.randn(n, 1)
    
    # Test different parameter values
    w_values = torch.linspace(0, 4, 50)
    nll_values = []
    mse_values = []
    
    for w in w_values:
        y_pred = w * X + true_b  # Fix b at true value
        nll_values.append(negative_log_likelihood(y, y_pred, sigma).item())
        mse_values.append(torch.mean((y - y_pred) ** 2).item())
    
    # Find minimizers
    nll_argmin = w_values[np.argmin(nll_values)]
    mse_argmin = w_values[np.argmin(mse_values)]
    
    print(f"NLL minimizer:  w = {nll_argmin.item():.4f}")
    print(f"MSE minimizer:  w = {mse_argmin.item():.4f}")
    print(f"True value:     w = {true_w}")
    print(f"Same minimizer: {torch.isclose(nll_argmin, mse_argmin)}")
    
    return w_values, nll_values, mse_values

w_vals, nll_vals, mse_vals = show_nll_mse_relationship()
```

### Visualizing the Relationship

```python
import matplotlib.pyplot as plt

def plot_nll_mse_comparison(w_values, nll_values, mse_values, true_w=2.0):
    """Plot NLL and MSE as functions of parameter w"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Both on same axes (different scales)
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    line1, = ax1.plot(w_values.numpy(), nll_values, 'b-', linewidth=2, label='NLL')
    line2, = ax1_twin.plot(w_values.numpy(), mse_values, 'r-', linewidth=2, label='MSE')
    
    ax1.axvline(x=true_w, color='green', linestyle='--', label=f'True w={true_w}')
    ax1.set_xlabel('w (weight)')
    ax1.set_ylabel('Negative Log-Likelihood', color='blue')
    ax1_twin.set_ylabel('Mean Squared Error', color='red')
    ax1.set_title('NLL and MSE Have Same Minimizer')
    ax1.legend(handles=[line1, line2], loc='upper right')
    
    # Plot 2: Normalized to show identical shape
    ax2 = axes[1]
    nll_norm = (np.array(nll_values) - min(nll_values)) / (max(nll_values) - min(nll_values))
    mse_norm = (np.array(mse_values) - min(mse_values)) / (max(mse_values) - min(mse_values))
    
    ax2.plot(w_values.numpy(), nll_norm, 'b-', linewidth=2, label='NLL (normalized)')
    ax2.plot(w_values.numpy(), mse_norm, 'r--', linewidth=2, label='MSE (normalized)')
    ax2.axvline(x=true_w, color='green', linestyle='--', label=f'True w={true_w}')
    ax2.set_xlabel('w (weight)')
    ax2.set_ylabel('Normalized Loss')
    ax2.set_title('Identical Shape (Up to Scaling)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Run: plot_nll_mse_comparison(w_vals, nll_vals, mse_vals)
```

## Loss Function Details in PyTorch

### nn.MSELoss Options

```python
import torch
import torch.nn as nn

def demonstrate_mse_reductions():
    """Show different reduction modes for MSELoss"""
    y_true = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_pred = torch.tensor([[1.5], [2.0], [2.5], [4.5]])
    
    # Reduction modes
    mse_none = nn.MSELoss(reduction='none')
    mse_mean = nn.MSELoss(reduction='mean')
    mse_sum = nn.MSELoss(reduction='sum')
    
    print("MSELoss Reduction Modes:")
    print(f"  reduction='none': {mse_none(y_pred, y_true).squeeze().tolist()}")
    print(f"  reduction='mean': {mse_mean(y_pred, y_true).item():.4f}")
    print(f"  reduction='sum':  {mse_sum(y_pred, y_true).item():.4f}")
    
    # Verify relationships
    individual_losses = mse_none(y_pred, y_true)
    print(f"\n  Manual mean: {individual_losses.mean().item():.4f}")
    print(f"  Manual sum:  {individual_losses.sum().item():.4f}")

demonstrate_mse_reductions()
```

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_with_mse_loss():
    """Complete training loop demonstrating MSE as NLL"""
    torch.manual_seed(42)
    
    # Generate synthetic data
    n_samples = 500
    n_features = 5
    true_weights = torch.randn(n_features, 1)
    true_bias = 0.5
    noise_std = 0.3
    
    X = torch.randn(n_samples, n_features)
    y = X @ true_weights + true_bias + noise_std * torch.randn(n_samples, 1)
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = nn.Linear(n_features, 1)
    
    # Loss function (MSE = proportional to NLL under Gaussian noise)
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    n_epochs = 100
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            # Forward pass
            y_pred = model(batch_X)
            
            # Compute MSE loss (equivalent to maximizing Gaussian likelihood)
            loss = criterion(y_pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: MSE = {epoch_loss:.6f}")
    
    # Evaluate
    with torch.no_grad():
        y_pred_all = model(X)
        final_mse = criterion(y_pred_all, y).item()
        
        # Estimate noise variance from residuals
        residuals = y - y_pred_all
        estimated_sigma = torch.sqrt(torch.mean(residuals ** 2)).item()
        
        # Compute NLL with estimated sigma
        nll = negative_log_likelihood(y, y_pred_all, estimated_sigma).item()
    
    print(f"\nFinal Results:")
    print(f"  Final MSE: {final_mse:.6f}")
    print(f"  Estimated σ: {estimated_sigma:.4f} (True: {noise_std})")
    print(f"  Final NLL: {nll:.4f}")
    
    return model, losses

model, loss_history = train_with_mse_loss()
```

## When MSE/Gaussian Assumption Breaks Down

### Heavy-Tailed Distributions

When errors have heavier tails than Gaussian, use robust alternatives:

```python
def robust_loss_functions():
    """Demonstrate robust alternatives to MSE"""
    torch.manual_seed(42)
    
    # Data with outliers
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 20.0])  # 20.0 is an outlier
    y_pred = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.0])
    
    # MSE (sensitive to outliers)
    mse = torch.mean((y_true - y_pred) ** 2)
    
    # MAE (Mean Absolute Error) - more robust
    mae = torch.mean(torch.abs(y_true - y_pred))
    
    # Huber loss - smooth transition from L2 to L1
    huber = nn.HuberLoss(delta=1.0)
    huber_loss = huber(y_pred, y_true)
    
    print("Robust Alternatives to MSE:")
    print(f"  MSE:   {mse.item():.4f} (sensitive to outlier)")
    print(f"  MAE:   {mae.item():.4f} (more robust)")
    print(f"  Huber: {huber_loss.item():.4f} (balanced)")

robust_loss_functions()
```

### Non-Constant Variance (Heteroscedasticity)

```python
def heteroscedastic_loss():
    """
    When variance depends on x, use weighted loss or 
    model the variance explicitly
    """
    torch.manual_seed(42)
    
    # Data with heteroscedastic noise (variance increases with X)
    n = 100
    X = torch.linspace(0, 5, n).reshape(-1, 1)
    noise_std = 0.1 + 0.3 * X.squeeze()  # Increasing variance
    y = 2 * X.squeeze() + noise_std * torch.randn(n)
    
    # Standard MSE (ignores heteroscedasticity)
    model_simple = nn.Linear(1, 1)
    y_pred_simple = model_simple(X).squeeze()
    mse_simple = torch.mean((y - y_pred_simple) ** 2)
    
    # Weighted MSE (inverse variance weighting)
    weights = 1.0 / (noise_std ** 2)
    weights = weights / weights.sum() * len(weights)  # Normalize
    weighted_mse = torch.mean(weights * (y - y_pred_simple) ** 2)
    
    print("Heteroscedastic Data:")
    print(f"  Standard MSE:  {mse_simple.item():.4f}")
    print(f"  Weighted MSE:  {weighted_mse.item():.4f}")
    print("\n  Note: Proper handling requires modeling σ(x)")

heteroscedastic_loss()
```

## Implications for Deep Learning

### Why MSE for Regression

The MSE/NLL connection justifies using MSE for regression networks:

```python
class RegressionNetwork(nn.Module):
    """Deep network for regression with MSE loss"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training uses MSE loss
model = RegressionNetwork(input_dim=10, hidden_dims=[64, 32])
criterion = nn.MSELoss()  # Justified by Gaussian likelihood
optimizer = torch.optim.Adam(model.parameters())
```

### Connection to Regression Metrics

| Metric | Formula | Relation to NLL |
|--------|---------|-----------------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Proportional to NLL |
| RMSE | $\sqrt{\text{MSE}}$ | Scale of predictions |
| R² | $1 - \frac{\text{SSE}}{\text{SST}}$ | Explained variance |

## Summary

### Key Insights

1. **MSE = NLL (up to constants)** under Gaussian noise assumption
2. **Minimizing MSE = Maximizing likelihood** for linear regression
3. **Gradient descent on MSE** is statistically justified
4. **MLE properties** (consistency, efficiency) transfer to MSE estimator
5. **Violations** (heavy tails, heteroscedasticity) suggest alternative losses

### Practical Guidelines

| Scenario | Recommendation |
|----------|----------------|
| Standard regression | Use `nn.MSELoss()` |
| Outliers present | Consider `nn.HuberLoss()` or `nn.L1Loss()` |
| Heteroscedastic data | Model variance explicitly or use weighted loss |
| Uncertainty needed | Model both mean and variance |

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 1.2.5
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapter 4
- PyTorch Documentation: [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
