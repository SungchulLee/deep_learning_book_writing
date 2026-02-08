# 34.1.2 Policy Gradient Theorem

## Introduction

The policy gradient theorem is the foundational result that makes direct policy optimization tractable. It provides an analytical expression for the gradient of the expected return with respect to policy parameters, without requiring differentiation through the environment dynamics.

## The Objective Function

The goal of policy optimization is to maximize the expected cumulative discounted reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory sampled under policy $\pi_\theta$.

Equivalently, using the state distribution:

$$J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}} \left[\mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [Q^{\pi_\theta}(s, a)]\right]$$

where $d^{\pi_\theta}(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi_\theta)$ is the discounted state visitation distribution.

## The Policy Gradient Theorem

**Theorem (Sutton et al., 1999)**. The gradient of $J(\theta)$ with respect to $\theta$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s, a)\right]$$

### Proof Sketch

Starting from the performance objective:

$$J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a|s) Q^{\pi_\theta}(s, a)$$

Taking the gradient:

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \left[\nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a) + \pi_\theta(a|s) \nabla_\theta Q^{\pi_\theta}(s, a)\right]$$

The key insight is that the gradient of $Q^{\pi_\theta}$ with respect to $\theta$ can be recursively expanded using the Bellman equation, and the resulting infinite sum telescopes to give:

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a)$$

Using the log-derivative trick $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \, Q^{\pi_\theta}(s, a)\right]$$

## The Log-Derivative Trick

The identity $\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$ transforms the gradient computation:

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

This is essential because it converts the gradient into an expectation under the policy, enabling Monte Carlo estimation from sampled trajectories.

## Score Function Estimator

The policy gradient can be estimated from trajectories using the score function estimator:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \, G_t^{(i)}$$

where $G_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}$ is the return-to-go from time $t$ in trajectory $i$.

### Causality Refinement

The return-to-go formulation already incorporates the causality principle: actions at time $t$ can only affect future rewards. This eliminates past rewards from the gradient estimate, reducing variance.

## Equivalent Forms

The policy gradient theorem can be written in several equivalent forms:

| Form | Expression | Notes |
|------|-----------|-------|
| Q-function | $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot Q^{\pi}(s,a)]$ | Original form |
| Advantage | $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot A^{\pi}(s,a)]$ | Baseline subtraction |
| TD residual | $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot \delta_t]$ | One-step estimate |
| Return-to-go | $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot G_t]$ | Monte Carlo estimate |

where $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ and $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

## Connection to Likelihood Ratio Methods

The policy gradient is a special case of the likelihood ratio gradient estimator used in variational inference and black-box optimization. The general form:

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[\nabla_\theta \log p_\theta(x) \cdot f(x)]$$

In RL, $p_\theta$ is the trajectory distribution under policy $\pi_\theta$, and $f$ is the cumulative reward.

## Variance and Bias Trade-offs

A fundamental tension in policy gradient estimation:

- **Monte Carlo returns** ($G_t$): Unbiased but high variance
- **Value function estimates** ($Q_\theta$): Lower variance but potentially biased
- **Advantage estimates** ($A^\pi$): Reduces variance without adding bias (with perfect baseline)

The choice of return estimate determines the bias-variance trade-off and is explored further in subsequent sections on baselines and GAE.

## Computational Considerations

For practical implementation:

1. **Batch estimation**: Collect $N$ trajectories and average gradients
2. **Auto-differentiation**: PyTorch computes $\nabla_\theta \log \pi_\theta$ automatically
3. **Loss construction**: Define the surrogate loss $L(\theta) = -\frac{1}{N}\sum \log \pi_\theta(a|s) \cdot \hat{A}$
4. **Gradient ascent**: Standard optimizers minimize $-J(\theta)$

## Summary

The policy gradient theorem provides a model-free, gradient-based approach to policy optimization. Its power lies in requiring only samples from the environment—no knowledge of transition dynamics is needed. The log-derivative trick converts the gradient into a tractable expectation, while the choice of return estimator trades off bias and variance.
