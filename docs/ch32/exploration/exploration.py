"""
Chapter 32.9: Exploration Strategies
======================================
Epsilon-greedy, UCB, Boltzmann, exploration bonuses, and comparisons.
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt


# =============================================================================
# 1. Multi-Armed Bandit for Exploration Comparison
# =============================================================================

class MultiArmedBandit:
    """K-armed bandit with Gaussian rewards."""

    def __init__(self, k: int = 10, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.k = k
        self.q_true = np.random.randn(k)  # True action values
        self.optimal_action = np.argmax(self.q_true)

    def step(self, action: int) -> float:
        return np.random.normal(self.q_true[action], 1.0)


# =============================================================================
# 2. Exploration Strategies
# =============================================================================

class EpsilonGreedy:
    """ε-greedy exploration."""

    def __init__(self, k: int, epsilon: float = 0.1, decay: float = 1.0):
        self.k = k
        self.epsilon = epsilon
        self.decay = decay
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        eps = self.epsilon * (self.decay ** self.t)
        if np.random.random() < eps:
            return np.random.randint(self.k)
        return int(np.argmax(self.Q))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class UCB:
    """Upper Confidence Bound exploration."""

    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        # Try each action once first
        for a in range(self.k):
            if self.N[a] == 0:
                return a
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class Boltzmann:
    """Boltzmann (softmax) exploration."""

    def __init__(self, k: int, tau: float = 1.0, tau_decay: float = 1.0):
        self.k = k
        self.tau = tau
        self.tau_decay = tau_decay
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        temp = max(self.tau * (self.tau_decay ** self.t), 0.01)
        # Numerically stable softmax
        q_scaled = self.Q / temp
        q_scaled -= np.max(q_scaled)
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)
        return int(np.random.choice(self.k, p=probs))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class CountBasedBonus:
    """ε-greedy with count-based exploration bonus."""

    def __init__(self, k: int, epsilon: float = 0.05, beta: float = 1.0):
        self.k = k
        self.epsilon = epsilon
        self.beta = beta
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        # Q + exploration bonus
        bonus = self.beta / np.sqrt(self.N + 1)
        augmented_Q = self.Q + bonus
        return int(np.argmax(augmented_Q))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# =============================================================================
# 3. Bandit Experiment
# =============================================================================

def run_bandit_experiment(n_steps=1000, n_runs=200, k=10):
    """Compare exploration strategies on the k-armed bandit."""
    strategies = {
        "ε-greedy (ε=0.1)": lambda: EpsilonGreedy(k, epsilon=0.1),
        "ε-greedy (ε=0.01)": lambda: EpsilonGreedy(k, epsilon=0.01),
        "ε-greedy (decay)": lambda: EpsilonGreedy(k, epsilon=0.5, decay=0.995),
        "UCB (c=2)": lambda: UCB(k, c=2.0),
        "Boltzmann (τ=0.5)": lambda: Boltzmann(k, tau=0.5),
        "Count Bonus": lambda: CountBasedBonus(k, epsilon=0.05, beta=1.0),
    }

    results = {}

    for name, create_fn in strategies.items():
        avg_rewards = np.zeros(n_steps)
        optimal_pct = np.zeros(n_steps)

        for run in range(n_runs):
            bandit = MultiArmedBandit(k, seed=run)
            agent = create_fn()

            for t in range(n_steps):
                action = agent.select_action()
                reward = bandit.step(action)
                agent.update(action, reward)
                avg_rewards[t] += reward
                optimal_pct[t] += (action == bandit.optimal_action)

        avg_rewards /= n_runs
        optimal_pct /= n_runs
        results[name] = {"rewards": avg_rewards, "optimal": optimal_pct}

    return results


# =============================================================================
# 4. Demonstrations
# =============================================================================

def demo_exploration():
    """Run and display exploration comparison."""
    print("=" * 65)
    print("Exploration Strategy Comparison: 10-Armed Bandit")
    print("=" * 65)

    results = run_bandit_experiment(n_steps=1000, n_runs=200, k=10)

    print(f"\n{'Strategy':<25} {'Avg Reward (last 100)':>22} {'Optimal %':>12}")
    print("-" * 60)
    for name, data in results.items():
        avg_r = np.mean(data["rewards"][-100:])
        opt_pct = np.mean(data["optimal"][-100:]) * 100
        print(f"{name:<25} {avg_r:>22.4f} {opt_pct:>11.1f}%")

    return results


def demo_temperature_effect():
    """Show effect of temperature on Boltzmann action selection."""
    print("\n" + "=" * 65)
    print("Boltzmann Temperature Effect")
    print("=" * 65)

    Q = np.array([3.0, 2.5, 2.0, 1.0, 0.0])
    action_names = [f"a{i}(Q={q})" for i, q in enumerate(Q)]

    temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\nQ-values: {Q}")
    print(f"\n{'τ':>6}", end="")
    for name in action_names:
        print(f"  {name:>12}", end="")
    print()

    for tau in temps:
        q_scaled = Q / tau
        q_scaled -= np.max(q_scaled)
        probs = np.exp(q_scaled) / np.sum(np.exp(q_scaled))
        print(f"{tau:>6.1f}", end="")
        for p in probs:
            print(f"  {p:>12.4f}", end="")
        print()


# =============================================================================
# 5. Visualization
# =============================================================================

def plot_exploration_comparison(results):
    """Plot exploration strategy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#e74c3c', '#c0392b', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    # Average reward
    ax = axes[0]
    for (name, data), color in zip(results.items(), colors):
        window = 20
        smoothed = np.convolve(data["rewards"], np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=color, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Optimal action percentage
    ax = axes[1]
    for (name, data), color in zip(results.items(), colors):
        window = 20
        smoothed = np.convolve(data["optimal"], np.ones(window)/window, mode='valid')
        ax.plot(smoothed * 100, label=name, color=color, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Optimal Action %')
    ax.set_title('Percentage of Optimal Action Selection')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Exploration Strategies: 10-Armed Bandit', fontsize=14)
    plt.tight_layout()
    plt.savefig("exploration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exploration_comparison.png")


def plot_boltzmann_distributions():
    """Visualize Boltzmann distributions for different temperatures."""
    Q = np.array([3.0, 2.5, 2.0, 1.0, 0.0])
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, len(temps), figsize=(15, 3.5), sharey=True)

    for ax, tau in zip(axes, temps):
        q_scaled = Q / tau
        q_scaled -= np.max(q_scaled)
        probs = np.exp(q_scaled) / np.sum(np.exp(q_scaled))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(Q))]
        ax.bar(range(len(Q)), probs, color=colors)
        ax.set_title(f'τ = {tau}')
        ax.set_xlabel('Action')
        ax.set_xticks(range(len(Q)))
        ax.set_xticklabels([f'Q={q}' for q in Q], fontsize=7, rotation=45)

    axes[0].set_ylabel('P(action)')
    plt.suptitle('Boltzmann Exploration: Effect of Temperature', fontsize=13)
    plt.tight_layout()
    plt.savefig("boltzmann_temperature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: boltzmann_temperature.png")


def plot_ucb_illustration():
    """Illustrate UCB action selection over time."""
    k = 5
    bandit = MultiArmedBandit(k, seed=42)
    agent = UCB(k, c=2.0)

    n_steps = 100
    selections = np.zeros((n_steps, k))

    for t in range(n_steps):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)
        selections[t] = agent.Q + agent.c * np.sqrt(
            np.log(agent.t + 1) / (agent.N + 1e-10))

    fig, ax = plt.subplots(figsize=(10, 5))
    for a in range(k):
        ax.plot(selections[:, a], label=f'UCB(a={a}, Q*={bandit.q_true[a]:.2f})',
                linewidth=1.5)
    ax.axhline(y=bandit.q_true[bandit.optimal_action], color='k',
               linestyle='--', alpha=0.3, label='Best Q*')
    ax.set_xlabel('Step')
    ax.set_ylabel('UCB Value')
    ax.set_title('UCB Values Over Time (optimism decreases with visits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ucb_illustration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: ucb_illustration.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Main comparison
    results = demo_exploration()
    plot_exploration_comparison(results)

    # 2. Temperature effect
    demo_temperature_effect()
    plot_boltzmann_distributions()

    # 3. UCB illustration
    plot_ucb_illustration()

    print("\n✓ Exploration Strategies demonstrations complete.")
