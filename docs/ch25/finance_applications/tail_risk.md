# Tail Risk Modeling with GANs

GANs provide powerful tools for modeling extreme events and tail risks in financial markets, where traditional parametric models often fail to capture the complexity of crisis dynamics.

## Risk Modeling

### VaR Scenario Generator

```python
class VaRScenarioGenerator(nn.Module):
    """
    Generate scenarios for Value-at-Risk estimation.
    
    Focuses on tail risk by conditioning on extreme scenarios.
    """
    
    def __init__(self, latent_dim=100, num_factors=10, num_assets=100):
        super().__init__()
        
        self.num_factors = num_factors
        self.num_assets = num_assets
        
        # Factor model: returns = factor_loadings @ factors + idiosyncratic
        # Generate factor returns
        self.factor_gen = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_factors),
            nn.Tanh(),
        )
        
        # Generate idiosyncratic returns
        self.idio_gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_assets),
            nn.Tanh(),
        )
        
        # Learnable factor loadings
        self.factor_loadings = nn.Parameter(torch.randn(num_assets, num_factors) * 0.1)
    
    def forward(self, z, stress_level=0.0):
        """
        Args:
            z: Latent noise
            stress_level: 0 = normal conditions, 1 = extreme stress
        
        Returns:
            Asset returns
        """
        # Generate factor returns
        factor_returns = self.factor_gen(z)
        
        # Apply stress amplification
        factor_returns = factor_returns * (1 + stress_level)
        
        # Systematic component
        systematic = torch.matmul(factor_returns, self.factor_loadings.t())
        
        # Idiosyncratic component
        idiosyncratic = 0.01 * self.idio_gen(z)
        
        # Increase idiosyncratic vol under stress
        idiosyncratic = idiosyncratic * (1 + 0.5 * stress_level)
        
        returns = systematic + idiosyncratic
        
        return returns, factor_returns


def estimate_var_from_gan(generator, confidence_level=0.99, num_scenarios=10000,
                           portfolio_weights=None, latent_dim=100, device='cpu'):
    """
    Estimate VaR using GAN-generated scenarios.
    
    Args:
        generator: Trained scenario generator
        confidence_level: VaR confidence (e.g., 0.99 for 99% VaR)
        num_scenarios: Number of scenarios to generate
        portfolio_weights: Portfolio weights
        latent_dim: Latent dimension
        device: Device
    
    Returns:
        VaR estimate
    """
    generator.eval()
    
    if portfolio_weights is None:
        portfolio_weights = torch.ones(generator.num_assets, device=device)
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
    
    all_portfolio_returns = []
    
    batch_size = 1000
    for _ in range(num_scenarios // batch_size):
        z = torch.randn(batch_size, latent_dim, device=device)
        
        with torch.no_grad():
            returns, _ = generator(z, stress_level=0.0)
        
        portfolio_returns = (returns * portfolio_weights).sum(dim=1)
        all_portfolio_returns.append(portfolio_returns.cpu())
    
    all_portfolio_returns = torch.cat(all_portfolio_returns)
    
    # VaR is the quantile at (1 - confidence_level)
    var = torch.quantile(all_portfolio_returns, 1 - confidence_level)
    
    # ES (Expected Shortfall) is the mean of losses beyond VaR
    es = all_portfolio_returns[all_portfolio_returns <= var].mean()
    
    return {
        'VaR': -var.item(),  # Negate to express as positive loss
        'ES': -es.item(),
        'scenarios': all_portfolio_returns.numpy(),
    }
```



## Fraud Detection

### GAN for Anomaly Detection

```python
class FraudDetectorGAN:
    """
    Use GAN for fraud detection via reconstruction error.
    
    Train on legitimate transactions; fraudulent ones will have
    high reconstruction error.
    """
    
    def __init__(self, feature_dim=50, latent_dim=20, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        # Encoder: transaction -> latent
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim),
        ).to(device)
        
        # Decoder/Generator: latent -> transaction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim),
        ).to(device)
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        ).to(device)
    
    def compute_anomaly_score(self, transactions):
        """
        Compute anomaly score based on reconstruction error.
        
        Higher score = more likely to be fraudulent.
        """
        transactions = transactions.to(self.device)
        
        # Encode and decode
        latent = self.encoder(transactions)
        reconstructed = self.decoder(latent)
        
        # Reconstruction error
        recon_error = ((transactions - reconstructed) ** 2).sum(dim=1)
        
        # Discriminator score
        disc_score = torch.sigmoid(self.discriminator(transactions)).squeeze()
        
        # Combined anomaly score
        anomaly_score = recon_error * (1 - disc_score)
        
        return anomaly_score
    
    def classify(self, transactions, threshold=None):
        """
        Classify transactions as fraudulent or legitimate.
        """
        scores = self.compute_anomaly_score(transactions)
        
        if threshold is None:
            # Use 95th percentile as default threshold
            threshold = torch.quantile(scores, 0.95)
        
        is_fraud = scores > threshold
        
        return is_fraud, scores
```



## Stress Testing with GANs

### Conditional Stress Scenario Generation

GANs can be conditioned on stress indicators to generate plausible crisis scenarios:

```python
class StressConditionedGenerator(nn.Module):
    """Generate market scenarios conditioned on stress level."""
    
    def __init__(self, latent_dim=100, num_assets=50, stress_features=5):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim + stress_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, num_assets),
            nn.Tanh()
        )
    
    def forward(self, z, stress_indicators):
        # stress_indicators: VIX level, credit spreads, liquidity, etc.
        combined = torch.cat([z, stress_indicators], dim=1)
        returns = 0.05 * self.main(combined)  # Scale for crisis-level returns
        return returns
```

### Expected Shortfall Estimation

```python
def estimate_es_from_gan(generator, confidence=0.975, n_scenarios=50000,
                          portfolio_weights=None, latent_dim=100, device='cpu'):
    """
    Estimate Expected Shortfall using GAN-generated tail scenarios.
    
    ES provides a coherent risk measure that captures the average loss
    in the worst (1-α)% of scenarios.
    """
    generator.eval()
    
    all_losses = []
    batch_size = 1000
    
    for _ in range(n_scenarios // batch_size):
        z = torch.randn(batch_size, latent_dim, device=device)
        
        with torch.no_grad():
            returns, _ = generator(z, stress_level=0.0)
        
        if portfolio_weights is not None:
            portfolio_pnl = (returns * portfolio_weights).sum(dim=1)
        else:
            portfolio_pnl = returns.mean(dim=1)
        
        all_losses.append(-portfolio_pnl.cpu())  # Negate: positive = loss
    
    all_losses = torch.cat(all_losses)
    
    # VaR at confidence level
    var = torch.quantile(all_losses, confidence)
    
    # ES = mean of losses exceeding VaR
    tail_losses = all_losses[all_losses >= var]
    es = tail_losses.mean() if len(tail_losses) > 0 else var
    
    return {
        'VaR': var.item(),
        'ES': es.item(),
        'tail_fraction': len(tail_losses) / len(all_losses),
    }
```

## Validation Framework

GAN-generated financial data must pass rigorous statistical validation:

```python
def validate_financial_gan(real_returns, generated_returns):
    """Validate that generated returns match real data properties."""
    from scipy import stats
    
    results = {}
    
    # 1. Distribution shape
    for asset_idx in range(real_returns.shape[1]):
        ks_stat, ks_pval = stats.ks_2samp(
            real_returns[:, asset_idx], generated_returns[:, asset_idx]
        )
        results[f'ks_asset_{asset_idx}'] = {'stat': ks_stat, 'pval': ks_pval}
    
    # 2. Correlation structure
    real_corr = np.corrcoef(real_returns.T)
    gen_corr = np.corrcoef(generated_returns.T)
    corr_error = np.mean(np.abs(real_corr - gen_corr))
    results['correlation_mae'] = corr_error
    
    # 3. Tail behavior (Hill estimator for tail index)
    for asset_idx in range(min(5, real_returns.shape[1])):
        real_tail = np.sort(np.abs(real_returns[:, asset_idx]))[-100:]
        gen_tail = np.sort(np.abs(generated_returns[:, asset_idx]))[-100:]
        
        real_hill = len(real_tail) / np.sum(np.log(real_tail / real_tail[0]))
        gen_hill = len(gen_tail) / np.sum(np.log(gen_tail / gen_tail[0]))
        results[f'hill_diff_asset_{asset_idx}'] = abs(real_hill - gen_hill)
    
    # 4. Volatility clustering (ARCH effect)
    for asset_idx in range(min(5, real_returns.shape[1])):
        real_sq = real_returns[:, asset_idx] ** 2
        gen_sq = generated_returns[:, asset_idx] ** 2
        
        real_acf = np.corrcoef(real_sq[:-1], real_sq[1:])[0, 1]
        gen_acf = np.corrcoef(gen_sq[:-1], gen_sq[1:])[0, 1]
        results[f'vol_clustering_diff_{asset_idx}'] = abs(real_acf - gen_acf)
    
    return results
```

## Summary

| Application | Use Case | Key Consideration |
|-------------|----------|-------------------|
| **VaR/ES Estimation** | Regulatory capital | Tail accuracy |
| **Stress Testing** | Scenario analysis | Conditional generation |
| **Fraud Detection** | Anomaly scoring | Reconstruction error |
| **Portfolio Stress** | Risk management | Factor model structure |
| **Validation** | Model governance | Statistical testing |
