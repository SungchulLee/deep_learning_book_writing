<<<<<<< HEAD
# Chapter Overview
=======
# Chapter 33: Value-Based Deep RL
>>>>>>> 96f31bd (...)

This chapter covers **Competition Programming**.

<<<<<<< HEAD
# Reference

[Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
=======
Value-based deep RL replaces tabular Q-functions with neural networks, enabling reinforcement learning in environments with high-dimensional state spaces. This chapter covers the Deep Q-Network (DQN) and its many improvements, multi-step learning corrections for off-policy data, extensions to continuous actions, offline RL methods that learn from fixed datasets, and evaluation best practices. Financial applications including order execution, market making, and discrete trading strategies are developed throughout.

## Contents

### 33.1 Deep Q-Networks

- [DQN Fundamentals](dqn/fundamentals.md) -- From tabular Q-learning to neural network function approximation with the TD loss formulation
- [Experience Replay](dqn/experience_replay.md) -- Breaking temporal correlations and improving data efficiency by storing and sampling transitions from a replay buffer
- [Target Networks](dqn/target_networks.md) -- Stabilizing training by separating the online network from a periodically updated target network
- [DQN Implementation](dqn/implementation.md) -- Complete DQN algorithm combining Q-learning, experience replay, and target networks with pseudocode
- [DQN Hyperparameters](dqn/hyperparameters.md) -- Comprehensive guide to learning rate, discount factor, buffer size, and exploration schedule tuning

### 33.2 DQN Improvements

- [Double DQN](improvements/double_dqn.md) -- Addressing Q-value overestimation by decoupling action selection from action evaluation
- [Dueling DQN](improvements/dueling_dqn.md) -- Decomposing Q-values into state value and advantage streams for more efficient learning
- [Prioritized Experience Replay](improvements/prioritized_replay.md) -- Sampling transitions proportional to TD error magnitude with importance sampling correction
- [Noisy Networks](improvements/noisy_networks.md) -- Learned state-dependent exploration through parametric noise in network weights
- [Distributional RL](improvements/distributional.md) -- Modeling the full return distribution with C51 categorical atoms instead of expected values
- [Rainbow](improvements/rainbow.md) -- Integrating six orthogonal DQN improvements into a single high-performance agent

### 33.3 Multi-Step Learning

- [N-Step Returns](multi_step/n_step_returns.md) -- Using multiple actual rewards before bootstrapping for better credit assignment with bias-variance trade-offs
- [Retrace(lambda)](multi_step/retrace.md) -- Safe off-policy correction for multi-step targets using truncated importance sampling ratios
- [V-Trace](multi_step/v_trace.md) -- Correcting for policy lag in distributed actor-learner architectures with truncated importance weights

### 33.4 Continuous Action Spaces

- [Normalized Advantage Functions (NAF)](continuous/naf.md) -- Quadratic Q-function decomposition enabling analytical argmax for continuous action DQN
- [QT-Opt](continuous/qt_opt.md) -- Cross-entropy method for approximate action optimization with arbitrary Q-network architectures

### 33.5 Offline RL

- [Offline RL Fundamentals](offline/fundamentals.md) -- Learning from fixed datasets without environment interaction, the distribution shift problem, and why standard DQN fails
- [Conservative Q-Learning (CQL)](offline/cql.md) -- Pessimistic Q-value estimation that lower-bounds true values for out-of-distribution actions
- [Batch-Constrained Q-Learning (BCQ)](offline/bcq.md) -- Constraining the policy to only select actions similar to those in the offline dataset
- [Implicit Q-Learning (IQL)](offline/iql.md) -- Avoiding out-of-distribution action queries entirely using expectile regression

### 33.6 Evaluation

- [Training Curves](evaluation/training_curves.md) -- Monitoring episode returns, diagnostic metrics, and interpreting healthy vs unhealthy training behavior
- [Benchmarks](evaluation/benchmarks.md) -- Standard RL evaluation environments from Classic Control and Atari to continuous control suites
- [Statistical Testing](evaluation/statistical.md) -- Welch's t-test, Mann-Whitney U, and confidence intervals for reliable algorithm comparison

### 33.7 Financial Applications

- [Order Execution](finance/order_execution.md) -- Optimal liquidation of large positions using DQN to minimize market impact costs
- [Market Making](finance/market_making.md) -- Learning optimal bid-ask quoting strategies with inventory risk management via DQN
- [Discrete Trading](finance/discrete_trading.md) -- Buy/sell/hold decision-making with DQN using technical features and transaction cost penalties
>>>>>>> 96f31bd (...)
