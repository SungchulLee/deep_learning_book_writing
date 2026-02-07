# Synthetic Data Generation

Using VAEs to generate realistic synthetic financial data for augmentation, privacy, and testing.

---

## Learning Objectives

By the end of this section, you will be able to:

- Apply VAEs to generate synthetic financial time series and cross-sectional data
- Validate that synthetic data preserves key statistical properties
- Use synthetic data for data augmentation and privacy-preserving analysis
- Implement a complete synthetic data generation pipeline

---

## Motivation

### Why Synthetic Financial Data?

Quantitative finance faces several data challenges where synthetic generation helps. Historical data is limited — markets have finite history, and regime changes are rare. Extreme events are scarce — tail events critical for risk management appear infrequently. Privacy and regulation restrict sharing of proprietary data. Model validation requires diverse test scenarios beyond historical experience.

VAEs address these by learning the joint distribution of financial variables and generating new samples that preserve realistic statistical properties.

---

## Generating Synthetic Returns

### Architecture for Financial Data

Financial returns have specific properties that inform VAE design: heavy tails, volatility clustering, cross-asset correlations, and non-stationarity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialVAE(nn.Module):
    """
    VAE for generating synthetic financial return data.
    
    Uses Gaussian decoder (MSE) since returns are continuous.
    """
    
    def __init__(self, num_assets=50, hidden_dim=128, latent_dim=16):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (Gaussian output — no sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)  # no activation
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate(self, num_samples, device='cpu'):
        """Generate synthetic return vectors."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z)
```

### Training

```python
def train_financial_vae(returns_data, num_epochs=100, batch_size=64):
    """
    Train VAE on historical return data.
    
    Args:
        returns_data: numpy array [num_days, num_assets]
    """
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(returns_data)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FinancialVAE(num_assets=returns_data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            
            # MSE reconstruction (Gaussian decoder)
            recon_loss = F.mse_loss(recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model
```

---

## Validation: Statistical Fidelity

Generated data must preserve key statistical properties:

```python
import numpy as np

def validate_synthetic_data(real_data, synthetic_data):
    """
    Compare statistical properties of real and synthetic data.
    
    Args:
        real_data: [N_real, num_assets]
        synthetic_data: [N_synth, num_assets]
    """
    results = {}
    
    # Marginal statistics
    results['mean_diff'] = np.abs(real_data.mean(axis=0) - synthetic_data.mean(axis=0)).mean()
    results['std_diff'] = np.abs(real_data.std(axis=0) - synthetic_data.std(axis=0)).mean()
    results['skew_diff'] = np.abs(
        scipy.stats.skew(real_data, axis=0) - scipy.stats.skew(synthetic_data, axis=0)
    ).mean()
    results['kurt_diff'] = np.abs(
        scipy.stats.kurtosis(real_data, axis=0) - scipy.stats.kurtosis(synthetic_data, axis=0)
    ).mean()
    
    # Correlation structure
    real_corr = np.corrcoef(real_data.T)
    synth_corr = np.corrcoef(synthetic_data.T)
    results['corr_frobenius'] = np.linalg.norm(real_corr - synth_corr, 'fro')
    
    return results
```

### Key Properties to Validate

| Property | Test | Acceptable Threshold |
|----------|------|---------------------|
| **Means** | Absolute difference | < 0.1× real mean |
| **Volatilities** | Ratio comparison | 0.8–1.2× real std |
| **Correlations** | Frobenius norm of difference | Problem-dependent |
| **Tail behavior** | QQ-plot, kurtosis | Qualitative match |
| **Cross-asset dependence** | Copula comparison | Visual + statistical |

---

## Applications

### Data Augmentation

Augment limited historical datasets for more robust model training:

```python
def augment_dataset(model, real_data, augmentation_factor=5):
    """Generate synthetic data to augment real dataset."""
    n_synthetic = len(real_data) * augmentation_factor
    synthetic = model.generate(n_synthetic).numpy()
    return np.concatenate([real_data, synthetic], axis=0)
```

### Privacy-Preserving Data Sharing

Share synthetic data that captures statistical properties without exposing actual trading records or proprietary positions.

### Backtesting with Diverse Scenarios

Generate thousands of plausible market scenarios to backtest strategies more thoroughly than historical data alone permits.

---

## Summary

| Application | Method | Key Validation |
|-------------|--------|---------------|
| **Augmentation** | Generate + concatenate | Marginal statistics match |
| **Privacy** | Share synthetic only | No real data leakage |
| **Backtesting** | Generate diverse scenarios | Correlation structure preserved |

---

## Exercises

### Exercise 1

Train a VAE on S&P 500 sector ETF daily returns. Generate 10,000 synthetic return days and compare correlation matrices.

### Exercise 2

Use synthetic data augmentation to improve a portfolio optimization model. Compare Sharpe ratios with and without augmentation.

---

## What's Next

The next section covers [Missing Data Imputation](imputation.md) using VAEs.
