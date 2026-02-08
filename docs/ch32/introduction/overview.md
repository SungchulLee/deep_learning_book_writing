# 32.1.1 Reinforcement Learning Overview

## What Is Reinforcement Learning?

Reinforcement Learning (RL) is one of three fundamental paradigms of machine learning, alongside supervised and unsupervised learning. In RL, an **agent** learns to make decisions by interacting with an **environment**, receiving **rewards** (or penalties) as feedback, and adjusting its behavior to maximize long-term cumulative reward.

### The Three Paradigms Compared

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|--------------------|-----------------------|----------------------|
| **Feedback** | Labeled examples | No labels | Reward signals |
| **Goal** | Predict labels/outputs | Find structure/patterns | Maximize cumulative reward |
| **Data** | i.i.d. dataset | i.i.d. dataset | Sequential, correlated |
| **Decision** | Single prediction | Clustering/representation | Sequential decisions |
| **Key Challenge** | Generalization | Representation | Credit assignment + exploration |

### Distinguishing Features of RL

1. **No supervisor**: The agent receives only a scalar reward signal, not explicit instructions on what the correct action is.

2. **Delayed reward**: Actions may have consequences that only become apparent many steps later. This is the **temporal credit assignment** problem.

3. **Sequential decision-making**: Current actions affect not only the immediate reward but also the future state and, consequently, all future rewards.

4. **Exploration vs. exploitation**: The agent must balance **exploiting** known good actions with **exploring** potentially better alternatives.

5. **Non-stationarity**: The data distribution the agent experiences depends on its own behavior—actions determine which states are visited.

## Historical Context

RL has roots in multiple disciplines:

- **Psychology**: Trial-and-error learning in animal behavior (Thorndike's Law of Effect, 1911)
- **Control Theory**: Optimal control and dynamic programming (Bellman, 1957)
- **Computer Science**: Temporal difference learning and the computational approach (Samuel, 1959; Sutton, 1988)
- **Neuroscience**: Dopamine reward prediction error signals (Schultz et al., 1997)

The modern synthesis of these threads began in the 1980s–1990s, with the convergence of TD learning, dynamic programming, and Monte Carlo methods into a unified framework.

## Key Terminology

- **Agent**: The learner and decision-maker
- **Environment**: Everything outside the agent that it interacts with
- **State** ($s$): A representation of the current situation
- **Action** ($a$): A choice made by the agent
- **Reward** ($r$): A scalar feedback signal
- **Policy** ($\pi$): A mapping from states to actions (the agent's strategy)
- **Value Function** ($V$ or $Q$): Expected cumulative future reward from a state (or state-action pair)
- **Model**: The agent's internal representation of environment dynamics (optional)
- **Episode**: A complete sequence from start to terminal state
- **Return** ($G_t$): The cumulative (possibly discounted) reward from time $t$ onward

## Taxonomy of RL Methods

RL methods can be categorized along several dimensions:

### Model-Based vs. Model-Free

- **Model-based**: The agent learns or is given a model of the environment dynamics $P(s'|s,a)$ and reward function $R(s,a)$. It can then plan using this model (e.g., dynamic programming, Monte Carlo tree search).
- **Model-free**: The agent learns directly from experience without explicitly modeling the environment. Most practical RL algorithms are model-free.

### Value-Based vs. Policy-Based vs. Actor-Critic

- **Value-based**: Learn a value function and derive a policy from it (e.g., Q-learning, DQN)
- **Policy-based**: Directly parameterize and optimize the policy (e.g., REINFORCE, PPO)
- **Actor-Critic**: Combine both—a policy (actor) and a value function (critic)

### On-Policy vs. Off-Policy

- **On-policy**: Learn about the policy currently being used to make decisions (e.g., SARSA)
- **Off-policy**: Learn about a different (typically optimal) policy while following an exploratory policy (e.g., Q-learning)

## Applications Overview

RL has achieved remarkable successes across diverse domains:

- **Games**: Backgammon (TD-Gammon), Go (AlphaGo/AlphaZero), Atari (DQN), StarCraft II (AlphaStar)
- **Robotics**: Locomotion, manipulation, autonomous navigation
- **Natural Language Processing**: RLHF for language model alignment
- **Healthcare**: Treatment planning, clinical trial optimization
- **Quantitative Finance**: Portfolio management, order execution, market making

## Quantitative Finance Perspective

RL is particularly well-suited to financial problems because:

1. **Markets are sequential**: Investment decisions unfold over time with uncertain outcomes
2. **Delayed feedback**: The profitability of a trading strategy may only become clear over extended periods
3. **Partial observability**: Market participants don't have access to all relevant information
4. **Non-stationarity**: Market dynamics change over time (regime changes)
5. **Risk-reward tradeoffs**: Financial objectives naturally involve balancing exploration (trying new strategies) with exploitation (following known profitable approaches)

### Example: Portfolio Management as RL

| RL Component | Financial Interpretation |
|-------------|------------------------|
| State | Market features (prices, volumes, indicators, portfolio holdings) |
| Action | Portfolio weight adjustments (buy/sell/hold decisions) |
| Reward | Risk-adjusted return (e.g., Sharpe ratio, log returns minus transaction costs) |
| Policy | Trading strategy |
| Episode | Investment horizon (e.g., one quarter, one year) |
| Environment | Financial market |

## Summary

Reinforcement learning provides a principled mathematical framework for sequential decision-making under uncertainty. Its core challenge—balancing exploration and exploitation while learning from delayed, scalar feedback—distinguishes it from other ML paradigms and makes it uniquely suited to problems where an agent must learn through interaction. The remainder of this chapter formalizes these intuitions and develops the algorithmic tools needed to solve RL problems.
