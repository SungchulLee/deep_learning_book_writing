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
