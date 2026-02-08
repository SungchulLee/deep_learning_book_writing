# 34.1.1 Policy Parameterization

## Introduction

Policy parameterization defines how a neural network maps observations to action distributions. The choice of parameterization profoundly impacts learning dynamics, exploration behavior, and the types of problems an agent can solve. This section covers the major parameterization strategies for both discrete and continuous action spaces.

## Discrete Action Spaces

For environments with a finite set of actions $\mathcal{A} = \{a_1, a_2, \ldots, a_K\}$, the policy outputs a categorical distribution over actions.

### Softmax Policy

The most common parameterization applies softmax to network logits:

$$\pi_\theta(a_i | s) = \frac{\exp(h_i(s; \theta))}{\sum_{j=1}^{K} \exp(h_j(s; \theta))}$$

where $h_i(s; \theta)$ are the logits produced by the neural network. Key properties:

- **Differentiable**: Enables gradient-based optimization
- **Strictly positive**: All actions maintain non-zero probability, ensuring exploration
- **Temperature control**: Scaling logits by $1/\tau$ adjusts exploration (lower $\tau$ → more exploitation)

### Log-Softmax Stability

Direct softmax computation is numerically unstable. The log-softmax trick provides stable log-probabilities:

$$\log \pi_\theta(a_i | s) = h_i(s; \theta) - \log \sum_{j=1}^{K} \exp(h_j(s; \theta))$$

PyTorch's `F.log_softmax` implements this with the max-subtraction trick internally.

## Continuous Action Spaces

Continuous control tasks require policies that output probability densities over continuous actions $a \in \mathbb{R}^d$.

### Gaussian Policy

The most widely used continuous parameterization assumes diagonal Gaussian:

$$\pi_\theta(a | s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta^2(s))$$

$$= \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}\sigma_i(s)} \exp\left(-\frac{(a_i - \mu_i(s))^2}{2\sigma_i^2(s)}\right)$$

The network outputs mean $\mu_\theta(s)$ and either:
- **State-dependent variance**: $\sigma_\theta(s) = \text{softplus}(f_\theta(s))$
- **State-independent log-std**: Learnable parameter $\log \sigma$ (common in PPO)

The log-probability used for policy gradients:

$$\log \pi_\theta(a | s) = -\frac{1}{2}\sum_{i=1}^{d}\left[\frac{(a_i - \mu_i)^2}{\sigma_i^2} + \log(2\pi\sigma_i^2)\right]$$

### Squashed Gaussian (SAC)

For bounded action spaces $a \in [-1, 1]^d$, SAC applies a tanh squashing function:

$$a = \tanh(u), \quad u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$

The log-probability requires a change-of-variables correction:

$$\log \pi_\theta(a | s) = \log \mathcal{N}(u; \mu_\theta(s), \sigma_\theta^2(s)) - \sum_{i=1}^{d} \log(1 - \tanh^2(u_i))$$

### Beta Distribution Policy

An alternative for bounded actions uses the Beta distribution:

$$\pi_\theta(a | s) = \text{Beta}(a; \alpha_\theta(s), \beta_\theta(s))$$

where $\alpha, \beta > 0$ are parameterized via softplus. Advantages include natural bounded support without squashing corrections.

## Network Architectures

### Shared vs. Separate Networks

Two design patterns for actor-critic methods:

1. **Shared backbone**: Feature extractor shared between policy and value heads. Efficient but can cause gradient interference.
2. **Separate networks**: Independent networks for policy and value. More stable but doubles parameters.

### Output Layer Design

| Action Space | Output | Activation | Distribution |
|-------------|--------|------------|--------------|
| Discrete | Logits $\in \mathbb{R}^K$ | None (raw logits) | `Categorical` |
| Continuous (unbounded) | $\mu, \log\sigma \in \mathbb{R}^d$ | None / None | `Normal` |
| Continuous (bounded) | $\mu, \log\sigma \in \mathbb{R}^d$ | None / Clamp | `Normal` + `tanh` |

### Weight Initialization

Policy networks benefit from specific initialization:

- **Hidden layers**: Orthogonal initialization with gain $\sqrt{2}$
- **Policy output**: Small scale (gain $0.01$) for near-uniform initial policy
- **Value output**: Standard scale (gain $1.0$)

## Entropy and Exploration

The entropy of the policy provides a measure of exploration:

$$\mathcal{H}(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

For Gaussian policies:

$$\mathcal{H}(\pi_\theta(\cdot|s)) = \frac{d}{2}\log(2\pi e) + \sum_{i=1}^{d} \log \sigma_i(s)$$

Adding an entropy bonus $\beta \mathcal{H}(\pi)$ to the objective prevents premature convergence to deterministic policies.

## Summary

Policy parameterization determines the expressiveness and behavior of the learned policy. Softmax policies handle discrete actions naturally, Gaussian policies dominate continuous control, and squashed Gaussians address bounded action spaces. Proper initialization and entropy regularization are critical for stable training.
