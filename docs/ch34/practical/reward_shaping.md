# Reward Shaping

Reward Shaping is an important concept in practical RL techniques. and finance-specific reward functions. This implementation provides a hands-on demonstration of the key algorithms and data structures involved, illustrating both the theoretical foundations and practical considerations for real-world deployment.

## Code

```python
"""
Chapter 34.6.1: Reward Shaping
================================
Potential-based reward shaping, reward normalization,
and finance-specific reward functions.
"""

import torch
import numpy as np
import gymnasium as gym
from collections import deque

# ========================================================================
# Main
# ========================================================================


# ---------------------------------------------------------------------------
# Potential-Based Reward Shaping
# ---------------------------------------------------------------------------

class PotentialBasedShaping:
    """
    Potential-based reward shaping: r' = r + γΦ(s') - Φ(s)
    Guaranteed to preserve optimal policy.
    """
    
    def __init__(self, potential_fn, gamma=0.99):
        self.potential_fn = potential_fn
        self.gamma = gamma
    
    def shape(self, reward, state, next_state):
        phi_s = self.potential_fn(state)
        phi_s_next = self.potential_fn(next_state)
        return reward + self.gamma * phi_s_next - phi_s


class RewardNormalizer:
    """Running normalization of rewards using Welford's algorithm."""
    
    def __init__(self, clip=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip = clip
    
    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def normalize(self, reward):
        self.update(reward)
        return np.clip(reward / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)


class RewardClipper:
    """Clip rewards to [-clip, clip] range."""
    def __init__(self, clip=1.0):
        self.clip = clip
    
    def __call__(self, reward):
        return np.clip(reward, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# Finance Reward Functions
# ---------------------------------------------------------------------------

class SharpeReward:
    """
    Sharpe-ratio based reward for portfolio management.
    Computes rolling Sharpe ratio as reward signal.
    """
    
    def __init__(self, window=20, risk_free_rate=0.0, annualize=252):
        self.returns = deque(maxlen=window)
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
    
    def __call__(self, portfolio_return):
        self.returns.append(portfolio_return)
        if len(self.returns) < 5:
            return portfolio_return
        
        returns = np.array(self.returns)
        excess = returns - self.risk_free_rate / self.annualize
        
        mean_return = excess.mean()
        std_return = excess.std() + 1e-8
        
        sharpe = mean_return / std_return * np.sqrt(self.annualize)
        return sharpe


class DrawdownPenalty:
    """Penalize portfolio drawdowns."""
    
    def __init__(self, penalty_coef=1.0):
        self.peak = 1.0
        self.penalty_coef = penalty_coef
    
    def __call__(self, portfolio_value, base_reward):
        self.peak = max(self.peak, portfolio_value)
        drawdown = (self.peak - portfolio_value) / self.peak
        return base_reward - self.penalty_coef * drawdown


def demo_reward_shaping():
    """Compare training with different reward signals."""
    print("=" * 60)
    print("Reward Shaping Comparison")
    print("=" * 60)
    
    # CartPole with different reward shaping
    env = gym.make("CartPole-v1")
    
    # Define potential: closer to center = higher potential
    def center_potential(obs):
        return -abs(obs[0]) - abs(obs[2]) * 0.5  # Penalize position and angle
    
    shaper = PotentialBasedShaping(center_potential, gamma=0.99)
    normalizer = RewardNormalizer()
    
    # Collect some transitions and show shaped rewards
    obs, _ = env.reset()
    print(f"\n{'Step':>4} {'Raw':>8} {'Shaped':>8} {'Normalized':>10}")
    print("-" * 34)
    
    for step in range(10):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        
        shaped = shaper.shape(reward, obs, next_obs)
        normalized = normalizer.normalize(reward)
        
        print(f"{step:>4} {reward:>8.3f} {shaped:>8.3f} {normalized:>10.3f}")
        
        obs = next_obs
        if term or trunc:
            break
    
    env.close()
    
    # Finance reward demo
    print("\n" + "-" * 40)
    print("Finance Reward Functions")
    print("-" * 40)
    
    sharpe_reward = SharpeReward(window=20)
    dd_penalty = DrawdownPenalty(penalty_coef=2.0)
    
    np.random.seed(42)
    portfolio_value = 100.0
    
    print(f"\n{'Day':>4} {'Return':>8} {'Sharpe':>8} {'DD Penalty':>12} {'Value':>8}")
    print("-" * 45)
    
    for day in range(30):
        daily_return = np.random.normal(0.001, 0.02)
        portfolio_value *= (1 + daily_return)
        
        sr = sharpe_reward(daily_return)
        dd_r = dd_penalty(portfolio_value, daily_return)
        
        if day % 5 == 0:
            print(f"{day:>4} {daily_return:>8.4f} {sr:>8.3f} {dd_r:>12.4f} {portfolio_value:>8.2f}")


if __name__ == "__main__":
    demo_reward_shaping()```

## Discussion

The implementation centers on the `PotentialBasedShaping`, `RewardNormalizer`, `RewardClipper` classes, which encapsulate the core logic of reward shaping. The code follows a modular design that separates the algorithmic components from the demonstration and evaluation logic.

The demonstration function shows the practical application of these components on standard reinforcement learning benchmarks. By examining the output, one can observe how the algorithm's performance varies with different hyperparameter choices and problem configurations.

From a practical standpoint, this implementation prioritizes clarity over raw performance. Production systems would typically incorporate additional optimizations such as batched computation, GPU acceleration, and more sophisticated hyperparameter tuning. Nevertheless, the core algorithmic ideas demonstrated here transfer directly to large-scale applications.

## Exercises

**Exercise 1.**
Run the demonstration code and record the key output metrics. Modify one hyperparameter (such as the learning rate, hidden dimension, or number of layers) and describe how the results change.

??? success "Solution to Exercise 1"
    After running the demo, systematically vary the chosen hyperparameter while keeping others fixed. For example, doubling the hidden dimension typically improves representational capacity but increases computation time. The learning rate has a non-monotonic effect: too small leads to slow convergence, while too large causes instability. Document the specific numbers for at least three different values of the chosen hyperparameter.

---

**Exercise 2.**
Explain the role of the key architectural choices in the implementation. Why are specific activation functions, normalization strategies, or loss functions used? What would happen if you substituted alternatives?

??? success "Solution to Exercise 2"
    The architectural choices reflect established best practices in practical RL techniques. For instance, ReLU activations provide non-linearity while avoiding vanishing gradients for positive inputs. The loss function is chosen to match the task type (cross-entropy for classification, MSE for regression). Substituting alternatives (e.g., sigmoid activations, L1 loss) would change the optimization landscape and potentially degrade performance, though some substitutions may be beneficial in specific scenarios.

---

**Exercise 3.**
Extend the implementation to handle a more challenging scenario: either a larger dataset, a different problem variant, or an additional feature. Describe your modification and evaluate its impact on performance.

??? success "Solution to Exercise 3"
    One natural extension is to add regularization (dropout, weight decay) or a more sophisticated architecture (additional layers, skip connections). Implement the chosen extension, train on the same data, and compare metrics before and after. The extension should demonstrate understanding of both the original algorithm and the modification's theoretical motivation.
