# 32.2.4 Reward Functions

## Definition

The **reward function** specifies the immediate feedback the agent receives from the environment. It can be defined in several equivalent ways:

### Three-Argument Form

$$R(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$$

The expected reward depends on the current state, action, and resulting next state.

### Two-Argument Form

$$R(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{s'} P(s'|s,a) R(s,a,s')$$

The expected reward given state and action, marginalizing over next states.

### One-Argument Form

$$R(s) = \mathbb{E}[R_{t+1} \mid S_t = s] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')$$

The expected reward in a state under a specific policy (less common).

## Properties of Reward Functions

### Bounded Rewards

Most theoretical results assume bounded rewards: $|R(s,a,s')| \leq R_{\max}$ for all $(s,a,s')$. This ensures the discounted return is finite:

$$|G_t| \leq \frac{R_{\max}}{1 - \gamma}$$

### Reward Invariance

Adding a constant $c$ to all rewards does not change the optimal policy for discounted MDPs (it shifts all values uniformly). However, it **does** affect average-reward formulations.

### Potential-Based Shaping

The transformed reward $R'(s,a,s') = R(s,a,s') + \gamma \Phi(s') - \Phi(s)$ preserves the optimal policy for any potential function $\Phi: \mathcal{S} \to \mathbb{R}$. This is the only shaping that guarantees policy invariance (Ng et al., 1999).

## Matrix Representation

For finite MDPs, the expected reward can be represented as vectors:

$$\mathbf{r}_a \in \mathbb{R}^{|\mathcal{S}|}, \quad [\mathbf{r}_a]_i = R(s_i, a)$$

Or as a matrix under a fixed policy $\pi$:

$$[\mathbf{r}_\pi]_i = \sum_a \pi(a|s_i) R(s_i, a)$$

## Types of Reward Structures

| Type | Description | Learning Impact |
|------|-------------|-----------------|
| **Dense** | Non-zero reward at most steps | Easier to learn, may bias |
| **Sparse** | Non-zero only at goal/failure | Harder to learn, more general |
| **Shaped** | Engineered guidance signals | Faster learning if well-designed |
| **Intrinsic** | Self-generated (curiosity, novelty) | Encourages exploration |
| **Composite** | Weighted sum of objectives | Multi-objective trade-offs |

## Reward Design Challenges

### Reward Hacking

The agent finds unintended shortcuts to maximize reward:

- A cleaning robot that covers its camera sensor (can't see mess = no penalty)
- A trading agent that exploits simulation artifacts rather than learning genuine strategies

### Sparse Reward Problem

When rewards are sparse, the agent receives zero feedback for most actions, making credit assignment extremely difficult. Solutions include:

- Reward shaping to provide intermediate signals
- Curiosity-driven exploration (intrinsic motivation)
- Hindsight experience replay
- Curriculum learning (start with easier tasks)

### Multi-Objective Rewards

Real problems often have competing objectives. The scalar reward must encode trade-offs:

$$R = w_1 R_{\text{profit}} + w_2 R_{\text{risk}} + w_3 R_{\text{cost}} + w_4 R_{\text{constraint}}$$

Choosing weights is itself a design decision that can significantly affect the learned policy.

## Financial Reward Functions

### Profit-Based Rewards

$$R_{\text{PnL}}(s, a, s') = \text{Portfolio Value}_{t+1} - \text{Portfolio Value}_t$$

Simple but ignores risk. The agent may take excessive risk for higher expected returns.

### Risk-Adjusted Rewards

**Sharpe-inspired**: $R_t = \frac{r_t^{\text{portfolio}}}{\sigma_t^{\text{rolling}}}$

**Sortino-inspired**: Penalize only downside deviation.

**Differential Sharpe ratio**: Dense reward signal that directly optimizes the Sharpe ratio.

### Cost-Inclusive Rewards

$$R_t = r_t^{\text{portfolio}} - c \cdot |\Delta \mathbf{w}_t| - \text{slippage}_t - \text{borrow\_cost}_t$$

where $c$ is the transaction cost rate and $\Delta \mathbf{w}_t$ is the portfolio weight change.

### Drawdown-Penalized Rewards

$$R_t = r_t^{\text{portfolio}} - \lambda \cdot \max\left(0, \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t}\right)$$

Encourages the agent to avoid large drawdowns.

### Regulatory and Constraint Rewards

$$R_t = R_t^{\text{base}} - \mu \cdot \mathbb{1}[\text{constraint violated}]$$

Penalize violations of position limits, concentration limits, or risk budgets.

## Summary

The reward function is the sole mechanism for communicating goals to the RL agent. Designing effective reward functions requires careful thought about the true objective, potential for reward hacking, sparsity, and multi-objective trade-offs. In financial RL, reward design must balance profitability, risk management, transaction costs, and regulatory compliance.
