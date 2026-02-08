# 32.9.3 Boltzmann (Softmax) Exploration

## Overview

**Boltzmann exploration** (softmax action selection) selects actions with probability proportional to their exponentiated Q-values:

$$\pi(a|s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

where $\tau > 0$ is the **temperature** parameter.

## Temperature Parameter

| $\tau$ | Behavior |
|--------|----------|
| $\tau \to \infty$ | Uniform random (maximum exploration) |
| $\tau = 1$ | Standard softmax |
| $\tau \to 0$ | Greedy (maximum exploitation) |

## Properties

1. **Graded exploration**: Better actions are selected more often (unlike ε-greedy which explores uniformly)
2. **Smooth**: Small changes in Q-values lead to small changes in action probabilities
3. **Differentiable**: Enables policy gradient methods
4. **No hard threshold**: All actions always have positive probability

## Comparison with ε-Greedy

Consider Q-values: $Q(s, a_1) = 10$, $Q(s, a_2) = 9$, $Q(s, a_3) = 1$

| Method | P(a₁) | P(a₂) | P(a₃) |
|--------|--------|--------|--------|
| ε-greedy (ε=0.1) | 0.933 | 0.033 | 0.033 |
| Boltzmann (τ=1) | 0.731 | 0.269 | 0.000 |
| Boltzmann (τ=2) | 0.577 | 0.377 | 0.047 |
| Boltzmann (τ=5) | 0.422 | 0.380 | 0.198 |

Boltzmann naturally allocates more exploration to competitive alternatives and less to clearly inferior actions.

## Temperature Annealing

Decrease temperature over time: $\tau_t = \tau_0 / (1 + c \cdot t)$

This provides heavy exploration early and converges toward greedy behavior.

## Challenges

1. **Scale sensitivity**: Q-value magnitudes affect action probabilities
2. **Temperature tuning**: Too high → random, too low → greedy too fast
3. **Numerical stability**: Large Q-values cause overflow → use log-sum-exp trick

## Financial Application

Boltzmann exploration in portfolio allocation naturally assigns higher probability to promising allocations while still exploring alternatives. The temperature can be adapted to market regime: higher temperature during regime changes (explore more) and lower during stable periods (exploit known strategies).
