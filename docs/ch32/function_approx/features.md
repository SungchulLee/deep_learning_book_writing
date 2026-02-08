# 32.8.2 Feature Engineering for RL

## Overview

The choice of features $\mathbf{x}(s)$ critically determines the quality of function approximation. Good features capture the aspects of the state that are relevant for predicting value.

## Common Feature Types

### Polynomial Features

For state $s = (s_1, s_2)$: $\mathbf{x}(s) = (1, s_1, s_2, s_1 s_2, s_1^2, s_2^2, \ldots)$

Simple but suffers from the curse of dimensionality for high-order polynomials.

### Tile Coding

Partition the state space into overlapping grids (tilings). Each tiling activates one tile per state. Multiple overlapping tilings provide generalization.

Properties:
- Binary features (0 or 1)
- Constant number of active features per state
- Resolution controlled by tile size and number of tilings
- Efficient: O(number of tilings) per update

### Radial Basis Functions (RBFs)

$$x_j(s) = \exp\left(-\frac{\|s - c_j\|^2}{2\sigma_j^2}\right)$$

Gaussian bumps centered at $c_j$ with width $\sigma_j$. Provide smooth generalization.

### Fourier Basis

$$x_j(s) = \cos(\pi \mathbf{c}_j^\top s)$$

where $\mathbf{c}_j$ are integer coefficient vectors. Good for periodic or smooth value functions.

### Coarse Coding

Binary features that are active when the state falls within a receptive field. Larger receptive fields → more generalization but less discrimination.

## Comparison

| Feature Type | Generalization | Resolution | Complexity | Best For |
|-------------|---------------|-----------|-----------|----------|
| Polynomial | Smooth | Global | O(d^k) | Low-dim, smooth |
| Tile coding | Local | Adjustable | O(n_tilings) | General purpose |
| RBF | Smooth, local | Adjustable | O(n_centers) | Smooth functions |
| Fourier | Global | Adjustable | O(order^d) | Periodic/smooth |
| Neural network | Learned | Adaptive | O(parameters) | High-dimensional |

## Feature Engineering for Finance

### Technical Features

- Multi-scale returns: $r_t^{(k)} = (P_t - P_{t-k})/P_{t-k}$ for $k = 1, 5, 20, 60$
- Volatility estimates at multiple horizons
- Normalized volume, bid-ask spread
- RSI, MACD, Bollinger band position

### Fundamental Features

- Price-to-earnings ratio, book-to-market
- Earnings surprise, analyst revision
- Sector membership (one-hot)

### Portfolio Features

- Current position, unrealized P&L
- Time since last trade, holding period
- Distance to risk limits

## Feature Normalization

Critical for function approximation:
- **Z-score**: $(x - \mu) / \sigma$ — standard for most features
- **Min-max**: $(x - x_{\min}) / (x_{\max} - x_{\min})$ — bounded [0,1]
- **Rank transform**: Replace values with ranks — robust to outliers

## Summary

Feature engineering transforms raw states into representations suitable for function approximation. The choice of features determines the bias-variance trade-off: rich features reduce bias but increase variance and computation. In finance, combining multi-scale technical indicators with portfolio state features provides effective representations for RL agents.
