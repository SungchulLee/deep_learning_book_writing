# Order Flow Modeling

## Overview

Order flow — the sequence of buy and sell orders arriving at an exchange — is inherently sequential and well-suited to autoregressive modeling. Modeling order flow enables market microstructure analysis, optimal execution, and market making.

## Order Representation

Each order can be represented as a tuple of discrete and continuous features:

$$o_t = (\text{side}_t, \text{type}_t, \text{price}_t, \text{size}_t, \Delta t_t)$$

- **Side**: buy or sell
- **Type**: market, limit, cancel
- **Price**: relative to mid-price (discretized into ticks)
- **Size**: order quantity (discretized)
- **Inter-arrival time**: time since last order

## Autoregressive Order Flow Model

```python
class OrderFlowModel(nn.Module):
    def __init__(self, n_sides=2, n_types=3, n_prices=100, n_sizes=50):
        super().__init__()
        d = 128
        self.side_embed = nn.Embedding(n_sides, d)
        self.type_embed = nn.Embedding(n_types, d)
        self.price_embed = nn.Embedding(n_prices, d)
        self.size_embed = nn.Embedding(n_sizes, d)
        self.time_proj = nn.Linear(1, d)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 512), num_layers=6
        )
        
        # Factored output: side → type → price → size → time
        self.side_head = nn.Linear(d, n_sides)
        self.type_head = nn.Linear(d, n_types)
        self.price_head = nn.Linear(d, n_prices)
        self.size_head = nn.Linear(d, n_sizes)
```

## Applications

### Optimal Execution
Generate future order flow scenarios to evaluate execution strategies: predict market impact and optimal scheduling of large orders.

### Market Making
Predict the arrival of future orders to set bid-ask spreads: if aggressive buy orders are predicted, widen the ask spread.

### Price Impact Modeling
Learn how order flow drives price changes:

$$\Delta p_t = f(o_t, o_{t-1}, \ldots, o_{t-L})$$

## Challenges

- **High frequency**: order arrivals can exceed 10,000 per second, requiring efficient models
- **Non-stationarity**: market microstructure changes with time of day, news events, and market conditions
- **Latency**: real-time applications require sub-millisecond inference
