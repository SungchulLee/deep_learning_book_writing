# Finance Applications of GANs

GANs are increasingly valuable in quantitative finance for scenario generation, market simulation, risk modeling, and synthetic data generation. This section covers practical applications for financial institutions.

## Market Scenario Generation

### Stock Return Generator

```python
import torch
import torch.nn as nn
import numpy as np

class MarketScenarioGenerator(nn.Module):
    """
    Generate realistic multi-asset return scenarios.
    
    Captures:
    - Fat-tailed distributions
    - Volatility clustering
    - Cross-asset correlations
    """
    
    def __init__(self, latent_dim=100, num_assets=50, sequence_length=252):
        super().__init__()
        
        self.num_assets = num_assets
        self.sequence_length = sequence_length
        
        # Initial noise transformation
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Temporal generation with 1D convolutions
        self.temporal = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, num_assets, 4, 2, 1),
        )
        
        # Adaptive layer to match sequence length
        self.adapt = nn.Linear(128, sequence_length)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        x = self.fc(z)
        x = x.view(batch_size, 256, 8)
        x = self.temporal(x)  # (batch, num_assets, ~128)
        
        # Reshape and adapt to target length
        x = x.transpose(1, 2)  # (batch, time, assets)
        x = self.adapt(x.reshape(batch_size, -1, x.size(-1)))
        
        # Returns are typically small, so we scale appropriately
        returns = 0.02 * torch.tanh(x)  # Daily returns in reasonable range
        
        return returns


class MarketDiscriminator(nn.Module):
    """
    Discriminate real vs generated market scenarios.
    
    Uses TCN (Temporal Convolutional Network) architecture.
    """
    
    def __init__(self, num_assets=50, sequence_length=252):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv1d(num_assets, 64, 8, 2, 3),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, 8, 2, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, 8, 2, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, 8, 2, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )
    
    def forward(self, returns):
        # returns: (batch, sequence_length, num_assets)
        x = returns.transpose(1, 2)  # (batch, num_assets, sequence_length)
        features = self.main(x)
        return self.classifier(features)


def stylized_facts_loss(returns):
    """
    Encourage generated returns to match stylized facts of financial markets.
    
    Stylized facts:
    1. Fat tails (excess kurtosis)
    2. Volatility clustering
    3. Negative skewness
    4. Leverage effect (negative correlation between returns and volatility)
    """
    # Kurtosis (should be > 3 for fat tails)
    mean = returns.mean(dim=1, keepdim=True)
    std = returns.std(dim=1, keepdim=True) + 1e-8
    standardized = (returns - mean) / std
    kurtosis = (standardized ** 4).mean(dim=1) - 3
    kurtosis_loss = torch.relu(3 - kurtosis).mean()  # Penalize kurtosis < 3
    
    # Volatility clustering (squared returns should be autocorrelated)
    squared_returns = returns ** 2
    vol_t = squared_returns[:, :-1, :]
    vol_t1 = squared_returns[:, 1:, :]
    vol_autocorr = ((vol_t - vol_t.mean()) * (vol_t1 - vol_t1.mean())).mean()
    vol_clustering_loss = torch.relu(0.1 - vol_autocorr)  # Should be positive
    
    return kurtosis_loss + vol_clustering_loss
```

## Synthetic Financial Data

### Transaction Data Generator

```python
class TransactionGenerator(nn.Module):
    """
    Generate synthetic transaction data.
    
    Features: amount, timestamp, merchant category, location, etc.
    """
    
    def __init__(self, latent_dim=100, num_merchant_cats=100, num_locations=500):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Amount head (log-normal distribution)
        self.amount_head = nn.Linear(256, 2)  # mean and log_std
        
        # Merchant category head
        self.merchant_head = nn.Linear(256, num_merchant_cats)
        
        # Location head
        self.location_head = nn.Linear(256, num_locations)
        
        # Time features head (hour of day, day of week)
        self.time_head = nn.Linear(256, 24 + 7)  # 24 hours + 7 days
    
    def forward(self, z):
        features = self.main(z)
        
        # Amount (sample from log-normal)
        amount_params = self.amount_head(features)
        amount_mean = amount_params[:, 0]
        amount_logstd = amount_params[:, 1]
        amount = torch.exp(amount_mean + torch.exp(amount_logstd) * torch.randn_like(amount_mean))
        
        # Categorical distributions
        merchant_logits = self.merchant_head(features)
        location_logits = self.location_head(features)
        time_logits = self.time_head(features)
        
        return {
            'amount': amount,
            'merchant_logits': merchant_logits,
            'location_logits': location_logits,
            'hour_logits': time_logits[:, :24],
            'day_logits': time_logits[:, 24:],
        }
```

## Orderbook Simulation

### Limit Order Book Generator

```python
class OrderbookGenerator(nn.Module):
    """
    Generate realistic limit order book states.
    
    Output: Price levels and quantities on bid/ask sides.
    """
    
    def __init__(self, latent_dim=100, num_levels=10):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Condition on market state (volatility, trend, etc.)
        self.condition_embed = nn.Linear(5, latent_dim // 2)
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        
        # Bid prices (should be decreasing from best bid)
        self.bid_price_head = nn.Linear(256, num_levels)
        
        # Ask prices (should be increasing from best ask)
        self.ask_price_head = nn.Linear(256, num_levels)
        
        # Quantities (positive)
        self.bid_qty_head = nn.Linear(256, num_levels)
        self.ask_qty_head = nn.Linear(256, num_levels)
    
    def forward(self, z, market_conditions):
        cond_emb = self.condition_embed(market_conditions)
        x = torch.cat([z, cond_emb], dim=1)
        
        features = self.main(x)
        
        # Generate price levels
        # Use cumulative sum to ensure ordering
        bid_deltas = torch.softplus(self.bid_price_head(features))
        ask_deltas = torch.softplus(self.ask_price_head(features))
        
        # Best bid/ask centered at 0
        bid_prices = -torch.cumsum(bid_deltas, dim=1)  # Decreasing
        ask_prices = torch.cumsum(ask_deltas, dim=1)   # Increasing
        
        # Quantities
        bid_qtys = torch.softplus(self.bid_qty_head(features))
        ask_qtys = torch.softplus(self.ask_qty_head(features))
        
        return {
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_quantities': bid_qtys,
            'ask_quantities': ask_qtys,
        }
```



## Quantitative Finance Considerations

Financial GANs for market data generation must capture several stylized facts:

| Stylized Fact | Description | Implementation |
|---------------|-------------|----------------|
| **Fat tails** | Excess kurtosis > 3 | Kurtosis penalty in loss |
| **Volatility clustering** | ARCH effects | Temporal convolutions |
| **Leverage effect** | Negative return-vol correlation | Conditional generation |
| **Cross-asset correlation** | Time-varying dependencies | Multi-asset architecture |
| **Mean reversion** | Some assets exhibit reversion | Factor model structure |

Proper validation against historical data using statistical tests (Kolmogorov-Smirnov, autocorrelation matching, cross-correlation structure) is essential before any GAN-generated data is used in production systems.
