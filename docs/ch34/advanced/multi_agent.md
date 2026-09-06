# 34.5.3 Multi-Agent Reinforcement Learning
## Introduction

Multi-Agent RL (MARL) extends single-agent policy optimization to settings with multiple interacting agents. In finance, this models competitive markets, cooperative portfolio management, and adversarial scenarios like market making against informed traders.

## Problem Formulation

A **Markov Game** (Stochastic Game) extends the MDP to $N$ agents:

- **State**: $s \in \mathcal{S}$ (global or partially observed)
- **Actions**: $a = (a_1, \ldots, a_N)$, joint action of all agents
- **Transitions**: $P(s'|s, a_1, \ldots, a_N)$
- **Rewards**: $r_i(s, a_1, \ldots, a_N)$ per agent $i$

## Paradigms

### Independent Learning
Each agent learns independently, treating other agents as part of the environment. Simple but suffers from non-stationarity as all agents change simultaneously.

### Centralized Training, Decentralized Execution (CTDE)
During training, agents share information (observations, actions). During execution, each agent acts based only on its local observations. This is the dominant paradigm.

### Fully Centralized
A single policy controls all agents. Scales poorly with agent count but provides optimal coordination.

## Key Algorithms

### MADDPG (Multi-Agent DDPG)
Extends DDPG to multi-agent settings with centralized critics:

- Each agent $i$ has actor $\mu_{\theta_i}(o_i)$ and critic $Q_{\phi_i}(s, a_1, \ldots, a_N)$
- Critics see all agents' observations and actions (centralized)
- Actors only see local observations (decentralized)

### MAPPO (Multi-Agent PPO)
Applies PPO independently to each agent with a shared or agent-specific value function that conditions on global state. Surprisingly competitive with more complex methods.

### QMIX
For cooperative tasks, decomposes the joint Q-function into agent-specific utilities:

$$Q_\text{tot}(s, \mathbf{a}) = f(Q_1(o_1, a_1), \ldots, Q_N(o_N, a_N); s)$$

where $f$ is a monotonic mixing function ensuring consistent greedy action selection.

## Challenges

1. **Non-stationarity**: Each agent's environment changes as others learn
2. **Credit assignment**: Attributing team reward to individual agents
3. **Scalability**: Joint action space grows exponentially with agents
4. **Partial observability**: Agents typically have limited views
5. **Equilibrium selection**: Multiple Nash equilibria may exist

## Finance Applications

- **Market simulation**: Multiple trading agents creating realistic order flow
- **Multi-asset management**: Cooperative agents managing portfolio sectors
- **Adversarial trading**: Market makers vs. informed traders
- **Auction mechanisms**: Bidding strategies in financial markets

## Summary

MARL extends policy-based methods to multi-agent settings, with CTDE being the dominant paradigm. Applications in finance leverage both cooperative (portfolio management) and competitive (market making) formulations.

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
