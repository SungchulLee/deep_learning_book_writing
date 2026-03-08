# Chapter 34: Policy-Based Deep RL

This chapter covers **Advanced Data Structures**.

Policy-based methods directly optimize a parameterized policy to maximize expected returns, enabling learning in continuous action spaces and stochastic policy representations. This chapter covers the policy gradient theorem and REINFORCE, actor-critic architectures, trust region and proximal optimization methods, off-policy continuous control algorithms, and advanced topics including maximum entropy RL, hierarchical RL, and multi-agent settings. Financial applications in portfolio optimization, continuous trading, and risk-sensitive objectives are developed throughout.

## Contents

### 34.1 Foundations

- Policy Parameterization -- Mapping observations to action distributions with softmax (discrete) and Gaussian (continuous) parameterizations
- Policy Gradient Theorem -- The foundational result providing an analytical gradient of expected return without differentiating through environment dynamics
- REINFORCE -- The Monte Carlo policy gradient algorithm using sampled episode returns for gradient estimation
- Baseline Methods -- Variance reduction through state-dependent baselines that preserve gradient unbiasedness

### 34.2 Actor-Critic Methods

- [Actor-Critic Fundamentals](actor_critic/fundamentals.md) -- Combining policy-based actors with value-based critics for lower-variance gradient estimates
- Advantage Actor-Critic (A2C) -- Synchronous parallel environments with advantage-based policy updates and shared actor-critic networks
- Asynchronous Advantage Actor-Critic (A3C) -- Distributed asynchronous training with multiple worker threads and a shared global network
- [Generalized Advantage Estimation (GAE)](actor_critic/gae.md) -- Exponentially-weighted multi-step advantage estimates with tunable bias-variance trade-off via lambda

### 34.3 Trust Region Methods

- [Trust Region Policy Optimization (TRPO)](trust_region/trpo.md) -- Constrained policy optimization with KL divergence trust regions for monotonic improvement guarantees
- Natural Policy Gradient -- Fisher information-based gradient steps that are invariant to policy parameterization
- Proximal Policy Optimization (PPO) -- Clipped surrogate objective achieving TRPO-like stability with simple first-order optimization
- [PPO Implementation](trust_region/ppo_implementation.md) -- Complete production-quality PPO with vectorized environments, advantage normalization, and engineering best practices

### 34.4 Off-Policy Actor-Critic

- [Deep Deterministic Policy Gradient (DDPG)](off_policy/ddpg.md) -- DQN-style off-policy learning with a deterministic actor for continuous control
- [Twin Delayed DDPG (TD3)](off_policy/td3.md) -- Addressing DDPG overestimation with twin critics, delayed policy updates, and target policy smoothing
- [Soft Actor-Critic (SAC)](off_policy/sac.md) -- Maximum entropy off-policy learning with stochastic policies and automatic temperature tuning
- SAC Implementation -- Complete SAC with squashed Gaussian policy, twin Q-networks, and automatic entropy coefficient adjustment

### 34.5 Advanced Topics

- [Maximum Entropy RL](advanced/max_entropy.md) -- Entropy-augmented objectives, soft Bellman equations, and the theoretical foundation for exploration-exploitation balance
- Hierarchical RL -- Options framework, feudal networks, and goal-conditioned policies for temporal abstraction and long-horizon tasks
- [Multi-Agent RL](advanced/multi_agent.md) -- Markov games, independent learning, centralized training with decentralized execution, and competitive market modeling
- [Model-Based RL](advanced/model_based.md) -- Learning environment dynamics for planning, Dyna-style methods, and model predictive control

### 34.6 Practical Considerations

- Reward Shaping -- Potential-based reward modification for faster learning while preserving optimal policy guarantees
- Action Spaces -- Design choices for discrete vs continuous actions, bounded parameterizations, and multi-dimensional action structures
- Observation Normalization -- Running mean-variance normalization for observations, rewards, and advantages to stabilize training
- Debugging RL -- Systematic strategies for diagnosing policy collapse, value divergence, reward hacking, and other common failure modes

### 34.7 Financial Applications

- [Portfolio Optimization](finance/portfolio.md) -- Dynamic asset allocation with policy-based RL, continuous portfolio weight outputs, and risk-adjusted reward objectives
- [Continuous Trading](finance/continuous_trading.md) -- Real-time execution and position sizing at intraday timescales using continuous action policies
- [Risk-Sensitive RL](finance/risk_sensitive.md) -- Incorporating CVaR, Sharpe ratio, and variance penalties directly into the RL optimization objective
