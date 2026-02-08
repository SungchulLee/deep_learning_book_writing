# 33.5.1 Offline RL Fundamentals

## Motivation

Standard RL requires online interaction with the environment. In many domains—healthcare, finance, autonomous driving—online exploration is expensive, risky, or impossible. **Offline RL** (also called **batch RL**) learns policies entirely from a fixed dataset of previously collected transitions, without any environment interaction.

## Problem Formulation

Given a fixed dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ collected by one or more behavior policies $\mu$, find a policy $\pi$ that maximizes expected returns:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^T \gamma^t r_t\right]$$

The key constraint: we cannot collect new data. The policy must be evaluated and improved using only $\mathcal{D}$.

## The Distribution Shift Problem

The fundamental challenge of offline RL is **distributional shift**. When the learned policy $\pi$ selects actions not well-represented in $\mathcal{D}$, the Q-function has no data to accurately evaluate those actions, leading to **extrapolation error**.

### Why Standard DQN Fails Offline

1. The Q-network is trained on state-action pairs $(s, a) \in \mathcal{D}$
2. The $\max_{a'} Q(s', a')$ in the Bellman target may query unseen actions
3. Q-values for unseen actions are unreliable (often overestimated)
4. The policy exploits these overestimated Q-values, choosing bad actions
5. This compounds through bootstrapping → divergence

## Categories of Offline RL Methods

### 1. Policy Constraint Methods
Restrict $\pi$ to stay close to the behavior policy $\mu$:
- **BCQ**: Only consider actions that $\mu$ would take
- **BEAR**: Constrain via MMD distance to $\mu$

### 2. Value Pessimism Methods
Penalize Q-values for unseen state-action pairs:
- **CQL**: Add a regularizer that lowers Q-values for OOD actions
- Ensures the learned Q-function is a lower bound

### 3. Importance Sampling Methods
Re-weight transitions by $\pi/\mu$ ratios. High variance but theoretically principled.

### 4. Model-Based Methods
Learn a dynamics model from $\mathcal{D}$, then plan or generate synthetic data with uncertainty penalties.

### 5. In-Sample Methods
Only evaluate actions seen in the dataset:
- **IQL**: Learns from dataset Q-values without querying OOD actions

## Dataset Quality Matters

| Dataset Type | Description | Difficulty |
|-------------|-------------|------------|
| Expert | Collected by optimal policy | Easy (behavior cloning often suffices) |
| Medium | Collected by partially trained policy | Moderate |
| Random | Collected by random policy | Hard (poor coverage) |
| Mixed | Mix of policies (e.g., replay buffer) | Moderate |
| Medium-replay | Replay buffer of medium policy | Moderate-Hard |

## Evaluation Protocol

Offline RL evaluation typically:
1. Train on a fixed dataset (no interaction)
2. Deploy the learned policy in the environment
3. Report normalized scores: $\text{score} = \frac{\text{policy return} - \text{random return}}{\text{expert return} - \text{random return}} \times 100$

## Finance Relevance

Offline RL is arguably the most important RL paradigm for finance:
- **Historical data is abundant**: Years of market data are available
- **Online exploration is costly**: Trading with an untrained agent loses real money
- **Safety**: Cannot afford to explore risky strategies in live markets
- **Regulation**: Some trading strategies must be validated before deployment
