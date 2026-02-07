# Factor Discovery

Using autoencoders for unsupervised extraction of latent risk factors from high-dimensional market data.

---

## Overview

**What you'll learn:**

- Autoencoder-based factor models as nonlinear extensions of PCA
- Extracting latent factors from cross-sectional asset returns
- Comparing autoencoder factors with traditional factor models (PCA, Fama-French)
- Interpreting learned factors via decoder weight analysis
- Time-varying factor structure and regime-dependent factor models

---

## Mathematical Foundation

### Linear Factor Models

Traditional factor models assume a linear relationship between asset returns and latent factors:

$$r_t = B f_t + \epsilon_t$$

where $r_t \in \mathbb{R}^N$ is the vector of $N$ asset returns, $B \in \mathbb{R}^{N \times K}$ is the factor loading matrix, $f_t \in \mathbb{R}^K$ is the vector of $K$ latent factors, and $\epsilon_t$ is idiosyncratic noise.

PCA extracts factors by solving:

$$\min_B \sum_t \|r_t - B B^\top r_t\|^2$$

### Nonlinear Factor Models via Autoencoders

Autoencoders generalize this to nonlinear factor models:

$$f_t = f_\theta(r_t), \quad \hat{r}_t = g_\phi(f_t)$$

where $f_\theta$ is a nonlinear encoder (factor extraction) and $g_\phi$ is a nonlinear decoder (return reconstruction). This captures:

- Nonlinear factor loadings that vary with market conditions
- Interaction effects between factors
- Regime-dependent factor structures

---

## Part 1: Factor Extraction from Asset Returns

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FactorAutoencoder(nn.Module):
    """
    Autoencoder for extracting latent risk factors from
    cross-sectional asset returns.
    
    The encoder maps N-dimensional return vectors to K-dimensional
    factor vectors. The decoder reconstructs returns from factors.
    
    Parameters:
        n_assets: Number of assets (input dimension)
        n_factors: Number of latent factors (bottleneck dimension)
        hidden_dim: Hidden layer width
    """
    
    def __init__(self, n_assets, n_factors=5, hidden_dim=64):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_factors = n_factors
        
        # Encoder: Returns → Factors
        self.encoder = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_factors),
        )
        
        # Decoder: Factors → Returns
        self.decoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )
    
    def encode(self, r):
        """Extract factors from returns."""
        return self.encoder(r)
    
    def decode(self, f):
        """Reconstruct returns from factors."""
        return self.decoder(f)
    
    def forward(self, r):
        factors = self.encode(r)
        reconstructed = self.decode(factors)
        return reconstructed, factors


def generate_synthetic_returns(n_assets=50, n_days=2000, n_true_factors=3,
                                seed=42):
    """
    Generate synthetic asset returns with known factor structure
    for controlled evaluation.
    
    Returns = B × Factors + ε
    where factors have time-varying volatility (GARCH-like).
    """
    np.random.seed(seed)
    
    # True factor loadings
    B = np.random.randn(n_assets, n_true_factors) * 0.5
    
    # True factors with time-varying volatility
    factors = np.zeros((n_days, n_true_factors))
    vol = np.ones(n_true_factors) * 0.01
    
    for t in range(1, n_days):
        vol = 0.9 * vol + 0.1 * factors[t-1] ** 2 + 0.001
        factors[t] = np.random.randn(n_true_factors) * np.sqrt(vol)
    
    # Idiosyncratic noise
    epsilon = np.random.randn(n_days, n_assets) * 0.005
    
    # Returns
    returns = factors @ B.T + epsilon
    
    return returns, factors, B


def train_factor_model(returns, n_factors=5, num_epochs=100, 
                       learning_rate=0.001, batch_size=64):
    """
    Train autoencoder factor model on historical returns.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    
    # Create dataset
    returns_tensor = torch.FloatTensor(returns_scaled)
    dataset = torch.utils.data.TensorDataset(returns_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                          shuffle=True)
    
    # Initialize model
    n_assets = returns.shape[1]
    model = FactorAutoencoder(n_assets, n_factors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for (batch,) in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            recon, factors = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    return model, scaler, history
```

---

## Part 2: Comparison with PCA Factors

```python
def compare_with_pca(returns, model, scaler, device, n_factors=5):
    """
    Compare autoencoder factors with PCA factors.
    
    Metrics:
    - Reconstruction R² (explained variance)
    - Factor stability (rolling correlation)
    - Out-of-sample performance
    """
    returns_scaled = scaler.transform(returns)
    
    # PCA factors
    pca = PCA(n_components=n_factors)
    pca_factors = pca.fit_transform(returns_scaled)
    pca_recon = pca.inverse_transform(pca_factors)
    
    # Autoencoder factors
    model.eval()
    with torch.no_grad():
        returns_tensor = torch.FloatTensor(returns_scaled).to(device)
        ae_recon, ae_factors = model(returns_tensor)
        ae_recon = ae_recon.cpu().numpy()
        ae_factors = ae_factors.cpu().numpy()
    
    # Explained variance (R²)
    total_var = np.var(returns_scaled, axis=0).sum()
    
    pca_residual_var = np.var(returns_scaled - pca_recon, axis=0).sum()
    ae_residual_var = np.var(returns_scaled - ae_recon, axis=0).sum()
    
    pca_r2 = 1 - pca_residual_var / total_var
    ae_r2 = 1 - ae_residual_var / total_var
    
    print(f"Explained Variance (R²):")
    print(f"  PCA ({n_factors} factors): {pca_r2:.4f}")
    print(f"  Autoencoder ({n_factors} factors): {ae_r2:.4f}")
    
    # Per-asset reconstruction error
    pca_asset_mse = np.mean((returns_scaled - pca_recon) ** 2, axis=0)
    ae_asset_mse = np.mean((returns_scaled - ae_recon) ** 2, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(pca_asset_mse, ae_asset_mse, alpha=0.7)
    max_val = max(pca_asset_mse.max(), ae_asset_mse.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Equal')
    axes[0].set_xlabel('PCA Asset MSE')
    axes[0].set_ylabel('AE Asset MSE')
    axes[0].set_title('Per-Asset Reconstruction Error')
    axes[0].legend()
    
    axes[1].bar(range(n_factors), pca.explained_variance_ratio_, 
                alpha=0.7, label='PCA')
    axes[1].set_xlabel('Factor')
    axes[1].set_ylabel('Variance Explained')
    axes[1].set_title('PCA Factor Importance')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('factor_comparison.png', dpi=150)
    plt.show()
    
    return pca_factors, ae_factors
```

---

## Part 3: Factor Interpretation

```python
def interpret_factors(model, asset_names=None, n_top=10):
    """
    Interpret learned factors by examining decoder weights.
    
    The first layer of the decoder approximates the factor loading
    matrix: how each factor contributes to each asset's return.
    """
    # Extract decoder first layer weights
    # Shape: (hidden_dim, n_factors) — we want the effective loading
    first_layer = list(model.decoder.children())[0]
    weights = first_layer.weight.detach().cpu().numpy()  # (hidden, n_factors)
    
    # For deeper decoders, use the full Jacobian at the mean
    # Here we use a linear approximation via the first layer
    
    n_factors = model.n_factors
    
    if asset_names is None:
        asset_names = [f'Asset_{i}' for i in range(model.n_assets)]
    
    print("Factor Loadings (top assets per factor):")
    print("=" * 50)
    
    for f in range(n_factors):
        # Approximate loadings via decoder sensitivity
        z = torch.zeros(1, n_factors)
        z[0, f] = 1.0
        
        model.eval()
        with torch.no_grad():
            sensitivity = model.decode(z).squeeze().numpy()
        
        # Top positive and negative loadings
        sorted_idx = np.argsort(sensitivity)
        
        print(f"\nFactor {f+1}:")
        print(f"  Top positive: ", end="")
        for idx in sorted_idx[-n_top:][::-1]:
            print(f"{asset_names[idx]}({sensitivity[idx]:.3f})", end=" ")
        print()
        print(f"  Top negative: ", end="")
        for idx in sorted_idx[:n_top]:
            print(f"{asset_names[idx]}({sensitivity[idx]:.3f})", end=" ")
        print()


def visualize_factor_dynamics(factors, window=60):
    """
    Visualize time-varying factor dynamics:
    - Factor time series
    - Rolling volatility
    - Factor correlations over time
    """
    n_factors = factors.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Factor time series
    for f in range(n_factors):
        axes[0].plot(factors[:, f], alpha=0.7, label=f'Factor {f+1}')
    axes[0].set_title('Extracted Factor Time Series')
    axes[0].legend()
    
    # Rolling volatility
    for f in range(n_factors):
        rolling_vol = np.array([
            np.std(factors[max(0, t-window):t, f])
            for t in range(window, len(factors))
        ])
        axes[1].plot(rolling_vol, alpha=0.7, label=f'Factor {f+1}')
    axes[1].set_title(f'Rolling {window}-day Factor Volatility')
    axes[1].legend()
    
    # Factor correlation matrix
    corr = np.corrcoef(factors.T)
    im = axes[2].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title('Factor Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('factor_dynamics.png', dpi=150)
    plt.show()
```

---

## Part 4: Optimal Number of Factors

```python
def select_n_factors(returns, factor_range=range(1, 16), 
                     n_splits=5, **train_kwargs):
    """
    Select optimal number of factors via cross-validation.
    
    Uses time-series cross-validation (expanding window)
    to avoid look-ahead bias.
    """
    n_days = returns.shape[0]
    split_size = n_days // (n_splits + 1)
    
    results = {}
    
    for n_factors in factor_range:
        oos_errors = []
        
        for fold in range(n_splits):
            train_end = split_size * (fold + 1)
            test_end = min(train_end + split_size, n_days)
            
            train_returns = returns[:train_end]
            test_returns = returns[train_end:test_end]
            
            # Train model
            model, scaler, _ = train_factor_model(
                train_returns, n_factors=n_factors, 
                num_epochs=50, **train_kwargs
            )
            
            # Out-of-sample evaluation
            test_scaled = scaler.transform(test_returns)
            model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_scaled)
                recon, _ = model(test_tensor)
                oos_mse = nn.MSELoss()(recon, test_tensor).item()
            
            oos_errors.append(oos_mse)
        
        results[n_factors] = {
            'mean_oos_mse': np.mean(oos_errors),
            'std_oos_mse': np.std(oos_errors)
        }
        
        print(f"K={n_factors}: OOS MSE = {np.mean(oos_errors):.6f} "
              f"± {np.std(oos_errors):.6f}")
    
    return results
```

---

## Exercises

### Exercise 1: Synthetic Factor Recovery
Generate synthetic returns with 3 known factors. Train autoencoders with K = 1, 2, 3, 4, 5 factors. Measure how well the autoencoder recovers the true factor structure using canonical correlation analysis.

### Exercise 2: PCA vs Autoencoder Factors
Using real or simulated market data, compare PCA and autoencoder factors on explained variance (in-sample and out-of-sample), factor stability over rolling windows, and portfolio construction quality.

### Exercise 3: Nonlinear Factor Loadings
Train autoencoders with varying hidden layer sizes. Do wider/deeper networks capture meaningful nonlinear factor loadings, or does linear (shallow) perform comparably?

### Exercise 4: Factor Number Selection
Implement the cross-validation approach and plot the elbow curve of out-of-sample reconstruction error vs number of factors. Compare with the scree plot from PCA.

---

## Summary

| Approach | Linearity | Factor Loadings | Scalability |
|----------|-----------|-----------------|-------------|
| **PCA** | Linear | Fixed | Closed-form solution |
| **Shallow AE** | Approximately linear | Learned, approximately fixed | Gradient-based |
| **Deep AE** | Nonlinear | Learned, state-dependent | Gradient-based |
| **Sparse AE** | Nonlinear | Sparse, interpretable | Gradient-based |

**Key Insight:** Autoencoders extend traditional factor models by learning nonlinear factor structures from data. The encoder replaces the linear projection in PCA, while the decoder replaces the linear loading matrix. For financial data with regime changes, nonlinear interactions, and fat-tailed distributions, this flexibility can capture factor structures that linear methods miss.
