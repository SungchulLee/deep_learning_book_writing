# Tail Risk Modeling

## Overview

Normalizing flows can model the tails of financial return distributions more accurately than Gaussian-based models, enabling better estimation of extreme loss probabilities.

## Importance of Tails

Financial risk management is primarily concerned with tail events: the probability and severity of large losses. Standard models underestimate tail risk because Gaussian distributions decay as $e^{-x^2/2}$ while empirical returns have power-law tails decaying as $x^{-\alpha}$.

## Flow-Based Tail Modeling

### Heavy-Tailed Base Distributions
Instead of a Gaussian base, use a Student-t distribution:

$$z \sim t_\nu$$

This gives the flow a natural ability to model heavy tails, with the neural network transformations capturing asymmetry and dependence.

### Tail-Weighted Training
Emphasize tail observations during training:

$$\mathcal{L} = \frac{1}{N} \sum_i w_i \log p_\theta(x_i), \quad w_i \propto \mathbb{1}[|x_i| > q_{0.95}]$$

## Tail Dependence

Flows can capture asymmetric tail dependence — the phenomenon that assets become more correlated during market crashes than during rallies. This is critical for portfolio risk and cannot be captured by linear correlation alone.

## Backtesting

Evaluate tail risk models using:

- **Coverage test**: does the VaR violation rate match the predicted level?
- **Independence test**: are violations clustered or independent?
- **Conditional coverage**: combines both tests

## Stress Testing

Generate extreme scenarios by sampling from the tail of the learned distribution:

```python
def generate_stress_scenarios(flow, n_scenarios, tail_quantile=0.01):
    samples = flow.sample(n_scenarios * 100)
    portfolio_loss = -(samples * weights).sum(dim=-1)
    threshold = torch.quantile(portfolio_loss, 1 - tail_quantile)
    stress_scenarios = samples[portfolio_loss >= threshold]
    return stress_scenarios[:n_scenarios]
```
