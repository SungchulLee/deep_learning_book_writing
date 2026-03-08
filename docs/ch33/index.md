# Chapter 33: Value-Based Deep RL

This chapter covers **Competition Programming**.

Value-based deep RL replaces tabular Q-functions with neural networks, enabling reinforcement learning in environments with high-dimensional state spaces. This chapter covers the Deep Q-Network (DQN) and its many improvements, multi-step learning corrections for off-policy data, extensions to continuous actions, offline RL methods that learn from fixed datasets, and evaluation best practices. Financial applications including order execution, market making, and discrete trading strategies are developed throughout.

## Contents

### 33.1 Deep Q-Networks

- DQN Fundamentals -- From tabular Q-learning to neural network function approximation with the TD loss formulation
- Experience Replay -- Breaking temporal correlations and improving data efficiency by storing and sampling transitions from a replay buffer
- Target Networks -- Stabilizing training by separating the online network from a periodically updated target network
- DQN Implementation -- Complete DQN algorithm combining Q-learning, experience replay, and target networks with pseudocode
- [DQN Hyperparameters](dqn/hyperparameters.md) -- Comprehensive guide to learning rate, discount factor, buffer size, and exploration schedule tuning

### 33.2 DQN Improvements

- Double DQN -- Addressing Q-value overestimation by decoupling action selection from action evaluation
- Dueling DQN -- Decomposing Q-values into state value and advantage streams for more efficient learning
- Prioritized Experience Replay -- Sampling transitions proportional to TD error magnitude with importance sampling correction
- Noisy Networks -- Learned state-dependent exploration through parametric noise in network weights
- Distributional RL -- Modeling the full return distribution with C51 categorical atoms instead of expected values
- [Rainbow](improvements/rainbow.md) -- Integrating six orthogonal DQN improvements into a single high-performance agent

### 33.3 Multi-Step Learning

- [N-Step Returns](multi_step/n_step_returns.md) -- Using multiple actual rewards before bootstrapping for better credit assignment with bias-variance trade-offs
- Retrace(lambda) -- Safe off-policy correction for multi-step targets using truncated importance sampling ratios
- V-Trace -- Correcting for policy lag in distributed actor-learner architectures with truncated importance weights

### 33.4 Continuous Action Spaces

- [Normalized Advantage Functions (NAF)](continuous/naf.md) -- Quadratic Q-function decomposition enabling analytical argmax for continuous action DQN
- QT-Opt -- Cross-entropy method for approximate action optimization with arbitrary Q-network architectures

### 33.5 Offline RL

- Offline RL Fundamentals -- Learning from fixed datasets without environment interaction, the distribution shift problem, and why standard DQN fails
- [Conservative Q-Learning (CQL)](offline/cql.md) -- Pessimistic Q-value estimation that lower-bounds true values for out-of-distribution actions
- Batch-Constrained Q-Learning (BCQ) -- Constraining the policy to only select actions similar to those in the offline dataset
- [Implicit Q-Learning (IQL)](offline/iql.md) -- Avoiding out-of-distribution action queries entirely using expectile regression

### 33.6 Evaluation

- Training Curves -- Monitoring episode returns, diagnostic metrics, and interpreting healthy vs unhealthy training behavior
- Benchmarks -- Standard RL evaluation environments from Classic Control and Atari to continuous control suites
- Statistical Testing -- Welch's t-test, Mann-Whitney U, and confidence intervals for reliable algorithm comparison

### 33.7 Financial Applications

- Order Execution -- Optimal liquidation of large positions using DQN to minimize market impact costs
- [Market Making](finance/market_making.md) -- Learning optimal bid-ask quoting strategies with inventory risk management via DQN
- [Discrete Trading](finance/discrete_trading.md) -- Buy/sell/hold decision-making with DQN using technical features and transaction cost penalties
