# Hutchinson Trace Estimator

## Overview

The Hutchinson trace estimator is a stochastic method for estimating the trace of a matrix using random vector probes. It is the key computational trick that makes FFJORD and other continuous normalizing flows practical.

## Mathematical Foundation

For any square matrix $A \in \mathbb{R}^{d \times d}$ and random vector $\epsilon$ with $\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}[\epsilon\epsilon^T] = I$:

$$\text{tr}(A) = \mathbb{E}[\epsilon^T A \epsilon]$$

**Proof**: $\mathbb{E}[\epsilon^T A \epsilon] = \mathbb{E}[\text{tr}(\epsilon^T A \epsilon)] = \text{tr}(A \mathbb{E}[\epsilon\epsilon^T]) = \text{tr}(A)$

## Probe Distributions

| Distribution | $\epsilon_i \sim$ | Variance | Properties |
|-------------|------------|----------|-----------|
| Gaussian | $\mathcal{N}(0, 1)$ | $2\|A\|_F^2$ | Smooth, differentiable |
| Rademacher | $\{-1, +1\}$ uniform | $\sum_{i \neq j} A_{ij}^2$ | Lower variance, not differentiable |

Rademacher probes have strictly lower variance than Gaussian probes, but Gaussian probes are smoother for gradient-based optimization.

## Variance Reduction

The single-sample estimator has high variance. Strategies to reduce it:

### Multiple Probes
Average over $K$ random vectors: variance reduces by $1/K$, cost increases by $K$.

### Russian Roulette Estimator
Unbiased estimator using a random number of terms from the power series expansion of $\log\det(I + A)$.

## Computational Cost

| Method | Cost | Memory |
|--------|------|--------|
| Exact trace | $O(d)$ calls to $f$ | $O(d^2)$ for Jacobian |
| Hutchinson (1 probe) | $O(1)$ VJP | $O(d)$ |
| Hutchinson ($K$ probes) | $O(K)$ VJPs | $O(Kd)$ |

The VJP (vector-Jacobian product) is computed via reverse-mode autodiff, which has the same cost as a single backward pass.

## Usage in CNFs

In the FFJORD setting, the trace is integrated over time:

$$\Delta \log p = -\int_0^1 \text{tr}\left(\frac{\partial f}{\partial z}\right) dt \approx -\int_0^1 \epsilon^T \frac{\partial f}{\partial z} \epsilon \, dt$$

A single $\epsilon$ is drawn and reused across all ODE solver steps. The stochastic trace estimate introduces variance in the log-likelihood estimate but remains unbiased.
