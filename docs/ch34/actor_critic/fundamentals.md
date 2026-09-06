# 34.2.1 Actor-Critic Fundamentals
## Introduction

Actor-critic methods combine the strengths of policy-based and value-based approaches. The **actor** maintains a parameterized policy $\pi_\theta(a|s)$ and the **critic** learns a value function $V_\phi(s)$ (or $Q_\phi(s,a)$) that evaluates the actor's performance. This combination enables lower-variance gradient estimates than pure REINFORCE while maintaining the advantages of direct policy optimization.

## Architecture

### Two-Component Design

The actor-critic framework consists of:

1. **Actor** $\pi_\theta(a|s)$: Policy network that selects actions
2. **Critic** $V_\phi(s)$: Value network that estimates expected returns

The critic serves as a learned baseline, providing advantage estimates:

$$A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### Update Rules

**Actor update** (policy gradient with advantage):

$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t$$

**Critic update** (value function regression):

$$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi (V_\phi(s_t) - G_t)^2$$

where $G_t$ is the target return (Monte Carlo, TD, or n-step).

## Advantage of Actor-Critic over REINFORCE

| Aspect | REINFORCE | Actor-Critic |
|--------|-----------|--------------|
| Variance | High (Monte Carlo) | Low (bootstrapping) |
| Bias | Unbiased | Biased (value function error) |
| Update frequency | End of episode | Every step (online) |
| Sample efficiency | Low | Higher |
| Convergence speed | Slow | Faster |

The key trade-off: actor-critic introduces bias through the critic's imperfect value estimates but dramatically reduces variance, leading to faster practical convergence.

## Online Actor-Critic

The simplest actor-critic performs updates at every timestep:

```
For each step t:
    Take action a_t ~ π_θ(·|s_t)
    Observe r_t, s_{t+1}
    δ_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)       # TD error
    θ ← θ + α_θ · δ_t · ∇_θ log π_θ(a_t|s_t)     # Actor update
    φ ← φ + α_φ · δ_t · ∇_φ V_φ(s_t)              # Critic update
```

The TD error $\delta_t$ serves as a one-step advantage estimate.

## Batch Actor-Critic

In practice, batch updates are more stable:

1. Collect a batch of transitions using $\pi_\theta$
2. Compute advantage estimates for all transitions
3. Update actor using policy gradient with advantages
4. Update critic using value regression

### Critic Targets

The choice of critic target affects bias-variance:

- **TD(0)**: $y_t = r_t + \gamma V_\phi(s_{t+1})$ — low variance, high bias
- **N-step**: $y_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})$ — intermediate
- **Monte Carlo**: $y_t = G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ — no bias, high variance
- **GAE**: Exponentially-weighted mixture of n-step returns (Section 34.2.4)

## Design Choices

### Shared vs. Separate Networks

**Shared backbone** (common in A2C/PPO):

- Feature extractor shared between actor and critic heads
- Reduces parameters and computation
- Risk: gradient interference between policy and value objectives
- Mitigation: use different loss coefficients (e.g., 0.5 for value loss)

**Separate networks** (common in SAC/TD3):

- Independent architectures for actor and critic
- More stable training
- Better for off-policy methods where actor and critic may need different learning rates

### Entropy Regularization

Adding entropy bonus prevents premature convergence:

$$L_\text{total} = L_\text{policy} + c_v L_\text{value} - c_e \mathcal{H}(\pi_\theta)$$

Typical coefficients: $c_v = 0.5$, $c_e = 0.01$.

### Gradient Flow

For the shared network case, gradients from three sources flow through the network:

1. Policy gradient (actor): increases probability of advantageous actions
2. Value gradient (critic): improves value estimates
3. Entropy gradient: maintains exploration

These must be balanced to avoid one objective dominating.

## Theoretical Properties

### Convergence

Actor-critic methods with compatible function approximation converge to a local optimum under standard conditions. The critic must be a compatible function approximator:

$$Q_w(s, a) = \nabla_\theta \log \pi_\theta(a|s)^\top w$$

In practice, neural network critics violate this condition, but learning remains effective with sufficient capacity.

### The Deadly Triad

Actor-critic with function approximation combines three elements that can cause instability:

1. **Function approximation** (neural networks)
2. **Bootstrapping** (TD learning for critic)
3. **Off-policy learning** (if data reuse is employed)

On-policy actor-critic methods avoid the third element, providing better stability guarantees.

## Summary

Actor-critic methods bridge policy gradient and value-based learning. The critic provides low-variance advantage estimates, while the actor directly optimizes the policy. This combination forms the foundation for modern algorithms like A2C, A3C, PPO, and SAC.

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
