# 33.2.5 Distributional RL

## From Expected Values to Distributions

Standard DQN estimates the *expected* return $Q(s, a) = \mathbb{E}[Z(s, a)]$. **Distributional RL** instead models the full *distribution* of returns $Z(s, a)$, capturing not just the mean but also variance, skewness, and multimodality.

## C51: Categorical Distribution

The **C51 algorithm** (Bellemare et al., 2017) represents the return distribution as a categorical distribution over $N$ fixed atoms:

$$Z(s, a) \approx \sum_{i=0}^{N-1} p_i(s, a) \cdot \delta_{z_i}$$

where:
- $z_i = V_\text{min} + i \cdot \Delta z$ are $N$ equally spaced atoms in $[V_\text{min}, V_\text{max}]$
- $\Delta z = (V_\text{max} - V_\text{min}) / (N - 1)$
- $p_i(s, a)$ are the probabilities output by the network (via softmax)
- $N = 51$ atoms (hence "C51")

### Distributional Bellman Equation

The distributional Bellman operator projects the shifted target distribution back onto the fixed atom support:

$$T Z(s, a) \stackrel{D}{=} R + \gamma Z(s', a^*)$$

For each atom $z_j$ of the target distribution:
1. Compute the projected atom: $\hat{z}_j = r + \gamma z_j$
2. Clip to $[V_\text{min}, V_\text{max}]$
3. Distribute probability to neighboring atoms proportionally

### Loss Function

C51 minimizes the **cross-entropy** between the projected target distribution and the predicted distribution:

$$\mathcal{L} = -\sum_{i} m_i \log p_i(s, a)$$

where $m_i$ are the projected target probabilities.

## QR-DQN: Quantile Regression

**Quantile Regression DQN** (Dabney et al., 2018) represents the distribution using $N$ quantiles instead of fixed atoms:

$$Z(s, a) \approx \frac{1}{N}\sum_{i=1}^{N} \delta_{\theta_i(s, a)}$$

where $\theta_i$ are learned quantile values at fixed quantile fractions $\tau_i = \frac{2i - 1}{2N}$.

### Quantile Huber Loss

$$\rho_\tau^\kappa(\delta) = |\tau - \mathbf{1}(\delta < 0)| \cdot \mathcal{L}_\kappa(\delta)$$

where $\mathcal{L}_\kappa$ is the Huber loss with threshold $\kappa$.

## Advantages of Distributional RL

1. **Richer learning signal**: Cross-entropy loss provides more gradient information than MSE
2. **Risk-sensitive policies**: Access to the full distribution enables risk-aware decision making
3. **Better representations**: Distributional predictions force the network to learn more structured features
4. **State-of-the-art performance**: C51 and QR-DQN significantly outperform DQN on Atari benchmarks

## Finance Motivation

Distributional RL is particularly valuable in finance:
- **Risk management**: Directly model the distribution of portfolio returns, not just the expected return
- **VaR/CVaR optimization**: Quantile-based methods (QR-DQN) naturally estimate Value-at-Risk
- **Fat tails**: Capture non-Gaussian return distributions common in financial markets
- **Risk-return trade-off**: Agents can be trained to maximize expected return subject to distributional constraints

## Hyperparameters

| Parameter | C51 | QR-DQN |
|-----------|-----|--------|
| Number of atoms/quantiles $N$ | 51 | 200 |
| $V_\text{min}$ | -10 | N/A |
| $V_\text{max}$ | 10 | N/A |
| Huber threshold $\kappa$ | N/A | 1.0 |
