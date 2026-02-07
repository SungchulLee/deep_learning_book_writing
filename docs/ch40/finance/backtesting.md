# Backtesting Infrastructure

## Overview

Backtesting infrastructure validates model performance on historical data before live deployment. For quantitative finance, rigorous backtesting requires careful handling of look-ahead bias, survivorship bias, and realistic transaction cost modeling.

## Backtesting Pipeline

```
Historical Data → Feature Engineering → Model Inference → Strategy Simulation → Performance Analysis
      ↓                   ↓                    ↓                    ↓                     ↓
  Point-in-Time     No Future Info      Batch Inference     Realistic Costs        Risk Metrics
  Data Snapshots    Leakage Check       Over History        Slippage Model         Sharpe, DD
```

## Implementation

```python
import torch
import numpy as np
import pandas as pd
from typing import Dict, List

class Backtester:
    """Walk-forward backtesting with point-in-time data."""
    
    def __init__(self, model, feature_pipeline, transaction_cost_bps=5):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.transaction_cost = transaction_cost_bps / 10000
    
    def run(self, data: pd.DataFrame, train_window: int = 252,
            rebalance_freq: int = 21) -> pd.DataFrame:
        """Walk-forward backtest."""
        results = []
        
        for t in range(train_window, len(data), rebalance_freq):
            # Point-in-time features (no look-ahead)
            features = self.feature_pipeline.transform(
                data.iloc[:t]  # Only past data
            )
            
            # Model prediction
            with torch.no_grad():
                signal = self.model(
                    torch.tensor(features[-1:], dtype=torch.float32)
                ).item()
            
            # Forward returns (what we're trying to predict)
            end_t = min(t + rebalance_freq, len(data))
            forward_return = data['returns'].iloc[t:end_t].sum()
            
            # Apply transaction costs
            net_return = signal * forward_return - abs(signal) * self.transaction_cost
            
            results.append({
                'date': data.index[t],
                'signal': signal,
                'gross_return': signal * forward_return,
                'net_return': net_return,
            })
        
        return pd.DataFrame(results)
    
    def compute_metrics(self, results: pd.DataFrame) -> Dict:
        """Compute strategy performance metrics."""
        returns = results['net_return']
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
        win_rate = (returns > 0).mean()
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': returns.mean() * 252,
            'annual_vol': returns.std() * np.sqrt(252),
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(results),
        }
```

## Best Practices

- **Use walk-forward validation** — never train on future data
- **Model transaction costs** realistically including slippage and market impact
- **Test multiple time periods** including crisis periods (2008, 2020)
- **Beware of overfitting** — out-of-sample Sharpe rarely exceeds 50% of in-sample
- **Version everything** — data, features, model, and strategy parameters

## References

1. De Prado, M. López. "Advances in Financial Machine Learning." Wiley, 2018.
2. Bailey, D., et al. "The Deflated Sharpe Ratio." Journal of Portfolio Management, 2014.
