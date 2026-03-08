<<<<<<< HEAD
# Chapter Overview
=======
# Chapter 32: RL Foundations
>>>>>>> 96f31bd (...)

This chapter covers **algorithm design techniques** -- the meta-skills that allow you to approach unfamiliar problems systematically. Rather than memorizing solutions, we focus on building a toolkit of strategies that transfer across problem domains.

<<<<<<< HEAD
The chapter is organized into three major areas:

1. **Problem Classification** -- learning to recognize what type of problem you are facing (pattern matching, greedy vs. dynamic programming, transformation).
2. **Design Process** -- a disciplined approach to solving problems (understanding constraints, finding invariants, simplifying, working backwards, incremental construction).
3. **Optimization Techniques** -- powerful algorithmic patterns that reduce time complexity (two pointers, sliding window, prefix sums, monotonic structures, sparse tables, sqrt decomposition, Mo's algorithm).

Mastering these techniques transforms problem solving from guesswork into a structured discipline.

| Section | Key Topics |
|---|---|
| Classification | Patterns, Greedy vs DP, Selection, Transformation |
| Design Process | Understanding, Invariants, Simplification, Backwards, Incremental |
| Optimization | Two Pointers, Sliding Window, Prefix Sums, Monotonic Stack/Queue, Sparse Table, Sqrt Decomposition, Mo's Algorithm |

# Reference

- Skiena, S. *The Algorithm Design Manual*, 3rd Edition, Springer, 2020.
- Halim, S. & Halim, F. *Competitive Programming 4*, lulu.com, 2020.
- Cormen, T. et al. *Introduction to Algorithms (CLRS)*, 4th Edition, MIT Press, 2022.
=======
Reinforcement learning (RL) is the study of how agents learn to make sequential decisions by interacting with an environment and receiving reward signals. This chapter builds the mathematical foundations of RL from the ground up, covering Markov decision processes, value functions, dynamic programming, Monte Carlo methods, temporal difference learning, and exploration strategies. These fundamentals provide the essential building blocks for the deep RL methods in subsequent chapters.

## Contents

### 32.1 Introduction

- [Reinforcement Learning Overview](introduction/overview.md) -- RL as the third learning paradigm, distinguishing features, and comparison with supervised and unsupervised learning
- [Agent-Environment Interface](introduction/agent_environment.md) -- The interaction loop, trajectories, states, actions, rewards, and the formal agent-environment abstraction
- [The Reward Hypothesis](introduction/reward_hypothesis.md) -- The foundational assumption that all goals can be expressed as cumulative reward maximization

### 32.2 Markov Decision Processes

- [MDP Fundamentals](mdp/fundamentals.md) -- Formal MDP definition, the Markov property, and the mathematical framework for sequential decision-making
- [States and Actions](mdp/states_actions.md) -- Discrete and continuous state and action spaces with design principles for state representations
- [Transition Dynamics](mdp/transitions.md) -- Transition functions, matrix representations, and model-based vs model-free settings
- [Reward Functions](mdp/rewards.md) -- One, two, and three-argument reward formulations with design considerations
- [Discount Factor](mdp/discount.md) -- Discounting for convergence, economic interpretation, and the effect on agent behavior

### 32.3 Value Functions

- [State Value Function](value_functions/state_value.md) -- Expected return under a policy, properties, and computation methods for finite MDPs
- [Action Value Function](value_functions/action_value.md) -- Q-function definition, relationship to state value, and its role in model-free control
- [Bellman Equations](value_functions/bellman.md) -- Recursive value function decomposition as the mathematical foundation for all RL algorithms
- [Bellman Optimality Equations](value_functions/bellman_optimality.md) -- Optimal value functions and the max operator replacing policy averaging

### 32.4 Dynamic Programming

- [Policy Evaluation](dynamic_programming/policy_evaluation.md) -- Iterative computation of the state value function for a given policy using Bellman updates
- [Policy Improvement](dynamic_programming/policy_improvement.md) -- Constructing a better policy from a value function using greedy action selection
- [Policy Iteration](dynamic_programming/policy_iteration.md) -- Alternating evaluation and improvement to converge to the optimal policy
- [Value Iteration](dynamic_programming/value_iteration.md) -- Directly computing the optimal value function by applying the Bellman optimality operator

### 32.5 Monte Carlo Methods

- [Monte Carlo Prediction](monte_carlo/prediction.md) -- Model-free value estimation by averaging returns from sampled episodes
- [Monte Carlo Control](monte_carlo/control.md) -- Finding optimal policies using Q-value estimation with exploring starts and epsilon-soft policies
- [Off-Policy Monte Carlo](monte_carlo/off_policy.md) -- Learning about a target policy from data generated by a different behavior policy
- [Importance Sampling](monte_carlo/importance_sampling.md) -- Reweighting returns for off-policy estimation with ordinary and weighted variants

### 32.6 Temporal Difference Learning

- [TD Prediction](temporal_difference/td_prediction.md) -- Combining MC experience-based learning with DP bootstrapping for step-by-step value updates
- [TD(0)](temporal_difference/td0.md) -- The simplest one-step TD algorithm for estimating state values with the TD error signal
- [SARSA](temporal_difference/sarsa.md) -- On-policy TD control using state-action-reward-state-action quintuples for Q-value learning
- [Q-Learning](temporal_difference/q_learning.md) -- Off-policy TD control that directly learns the optimal action-value function regardless of exploration policy
- [Expected SARSA](temporal_difference/expected_sarsa.md) -- Variance-reduced TD control using expected Q-values over the next action distribution

### 32.7 N-Step Methods

- [N-Step TD Prediction](n_step/n_step_td.md) -- Generalizing TD(0) and Monte Carlo by bootstrapping after n actual reward steps
- [N-Step SARSA](n_step/n_step_sarsa.md) -- Extending SARSA with n-step returns for control with tunable bias-variance trade-off
- [TD(lambda)](n_step/td_lambda.md) -- Exponentially weighted combination of all n-step returns using the lambda-return
- [Eligibility Traces](n_step/eligibility_traces.md) -- Short-term memory mechanism for efficient backward-view implementation of TD(lambda)

### 32.8 Function Approximation

- [Linear Function Approximation](function_approx/linear.md) -- Parameterized value functions with linear weights and semi-gradient TD updates for large state spaces
- [Feature Engineering for RL](function_approx/features.md) -- Polynomial, tile coding, and radial basis function features for effective state representation
- [Convergence Issues](function_approx/convergence.md) -- The deadly triad of function approximation, bootstrapping, and off-policy training

### 32.9 Exploration Strategies

- [Epsilon-Greedy](exploration/epsilon_greedy.md) -- Simple random exploration with probability epsilon and common decay schedules
- [Upper Confidence Bound (UCB)](exploration/ucb.md) -- Optimism in the face of uncertainty using exploration bonuses based on visit counts
- [Boltzmann Exploration](exploration/boltzmann.md) -- Softmax action selection with temperature-controlled exploration-exploitation trade-off
- [Exploration Bonuses](exploration/bonuses.md) -- Intrinsic motivation through count-based, density-based, and prediction-error-based novelty rewards
>>>>>>> 96f31bd (...)
