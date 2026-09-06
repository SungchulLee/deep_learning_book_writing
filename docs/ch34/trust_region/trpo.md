# 34.3.1 Trust Region Policy Optimization (TRPO)
## Introduction

TRPO (Schulman et al., 2015) addresses a fundamental challenge in policy gradient methods: how large should each update step be? Too large a step can catastrophically degrade the policy; too small a step wastes computation. TRPO solves this by formulating policy optimization as a constrained optimization problem where updates are bounded by a KL divergence trust region.

## Motivation

Standard policy gradient updates $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ have no guarantees on the magnitude of policy change. A small change in parameters $\theta$ can cause a large change in the policy $\pi_\theta$, especially in regions of high sensitivity.

TRPO provides monotonic improvement guarantees by constraining how much the policy distribution can change per update.

## Theoretical Foundation

### Surrogate Objective

TRPO maximizes a surrogate objective that lower-bounds the true improvement:

$$L_{\pi_{\theta_\text{old}}}(\theta) = \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}, a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} A^{\pi_{\theta_\text{old}}}(s, a)\right]$$

This importance-sampled objective allows evaluating the new policy $\pi_\theta$ using data from the old policy $\pi_{\theta_\text{old}}$.

### Monotonic Improvement Theorem

Kakade & Langford (2002) showed that the true performance improvement satisfies:

$$J(\pi_\theta) \geq L_{\pi_{\theta_\text{old}}}(\theta) - C \cdot D_\text{KL}^{\max}(\pi_{\theta_\text{old}} \| \pi_\theta)$$

where $C = \frac{2\gamma \epsilon}{(1-\gamma)^2}$ and $\epsilon = \max_s |A^{\pi}(s, a)|$.

### TRPO Optimization Problem

TRPO replaces the penalty with a hard constraint:

$$\max_\theta \quad L_{\pi_{\theta_\text{old}}}(\theta)$$

$$\text{s.t.} \quad \bar{D}_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

where $\bar{D}_\text{KL}$ is the average KL divergence over states and $\delta$ is the trust region radius (typically $0.01$).

## Algorithm

### Conjugate Gradient Method

TRPO uses the conjugate gradient algorithm to approximately solve:

$$\theta_{k+1} = \theta_k + \alpha \hat{F}^{-1} \hat{g}$$

where:

- $\hat{g} = \nabla_\theta L(\theta)|_{\theta_k}$ is the policy gradient
- $\hat{F}$ is the Fisher information matrix (Hessian of KL divergence)
- $\alpha$ is determined by backtracking line search to satisfy the KL constraint

### Fisher-Vector Product

Computing $\hat{F}^{-1} g$ directly is intractable for large networks. Instead, the conjugate gradient method only requires Fisher-vector products $\hat{F}v$, computed efficiently via:

$$\hat{F}v = \nabla_\theta \left[(\nabla_\theta \bar{D}_\text{KL})^\top v\right]$$

This requires only two backpropagation passes, avoiding explicit matrix construction.

### Backtracking Line Search

After computing the search direction $s = \hat{F}^{-1}g$, TRPO determines the step size:

1. Compute maximum step: $\alpha_\text{max} = \sqrt{\frac{2\delta}{s^\top \hat{F} s}}$
2. Try $\alpha = \alpha_\text{max} \cdot \beta^k$ for $k = 0, 1, 2, \ldots$
3. Accept first $\alpha$ satisfying: (a) KL constraint, (b) positive surrogate improvement

## TRPO Pseudocode

```
For each iteration:
    Collect trajectories using π_θ
    Compute advantages (GAE)
    
    Compute policy gradient g = ∇_θ L(θ)
    Compute search direction s = F⁻¹g via conjugate gradient
    Compute max step size: α_max = sqrt(2δ / s^T F s)
    
    Backtracking line search:
        For k = 0, 1, 2, ..., K:
            θ_new = θ + α_max · β^k · s
            If KL(π_old || π_new) ≤ δ and L(θ_new) > L(θ):
                Accept θ_new
                Break
    
    Update value function (separate optimizer)
```

## Properties

### Advantages
- Monotonic improvement guarantee (under exact optimization)
- Robust to hyperparameter choices
- Stable training across diverse environments
- Does not require learning rate tuning

### Limitations
- Complex implementation (conjugate gradient, line search)
- Computationally expensive per iteration
- Does not scale well with very large networks
- Second-order computation overhead

## Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| $\delta$ | 0.01 | KL constraint (trust region size) |
| CG iterations | 10 | Conjugate gradient steps |
| CG damping | 0.1 | Fisher matrix damping |
| Line search steps | 10 | Backtracking attempts |
| Backtrack ratio $\beta$ | 0.5 | Step size decay |

## Summary

TRPO provides a principled approach to stable policy optimization through constrained updates. Its theoretical monotonic improvement guarantee and robustness to hyperparameters made it a significant advance. While PPO has largely replaced TRPO in practice due to simpler implementation, TRPO's theoretical framework remains foundational.

## Exercises

**Exercise 1.**
Derive the policy gradient for the method described in this section. Clearly state which terms require estimation and which can be computed exactly.

??? success "Solution to Exercise 1"
    The policy gradient takes the form $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ where $\hat{A}_t$ is the advantage estimate. The log-probability gradient $\nabla_\theta \log \pi_\theta$ can be computed exactly via automatic differentiation. The advantage $\hat{A}_t$ must be estimated from sampled trajectories, introducing variance. The expectation is approximated by averaging over a batch of trajectories. Variance reduction via baselines preserves unbiasedness while reducing the estimation noise. $\square$

---

**Exercise 2.**
Compare the sample efficiency of this method with a value-based approach (e.g., DQN) on a continuous control task. Explain the theoretical reasons for any observed differences.

??? success "Solution to Exercise 2"
    Policy-based methods are generally less sample-efficient than value-based methods because they use on-policy data (each trajectory is used once). DQN reuses data via experience replay, achieving better sample efficiency. However, policy methods handle continuous actions naturally (no argmax over action space needed), converge to stochastic policies when optimal, and provide monotonic improvement guarantees under trust regions. Off-policy actor-critic methods (DDPG, SAC) bridge this gap by combining policy optimization with experience replay. $\square$

---

**Exercise 3.**
Implement this method for a simple continuous control task (e.g., Pendulum-v1). Report hyperparameter sensitivity with respect to the learning rate and the key method-specific parameter.

??? success "Solution to Exercise 3"
    For Pendulum-v1 with a Gaussian policy, typical performance: learning rate $3 \times 10^{-4}$ achieves convergence in $\sim$500 episodes; $10^{-3}$ causes oscillation; $10^{-5}$ converges too slowly. The method-specific parameter (e.g., clipping range for PPO, KL constraint for TRPO) controls the trade-off between update aggressiveness and stability. Too aggressive leads to performance collapse; too conservative wastes samples. The optimal operating point balances these, typically found via grid search over a small range. $\square$

---

**Exercise 4.**
Discuss how this method could be applied to portfolio optimization where the action space is a simplex (portfolio weights summing to 1) and the reward is risk-adjusted return.

??? success "Solution to Exercise 4"
    The action space is the $(n-1)$-dimensional simplex $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$. The policy can use a Dirichlet distribution or softmax-transformed Gaussian. The reward is the Sharpe ratio or differential Sharpe ratio of the resulting portfolio. Challenges include: high-dimensional action space (many assets), transaction costs penalizing frequent rebalancing, and non-stationarity of market returns. The method from this section addresses these through its specific mechanism for stable policy updates. $\square$
