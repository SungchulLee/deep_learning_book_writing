"""
Chapter 32.1: Reinforcement Learning Introduction - Overview
============================================================
Demonstrates the basic RL concepts with simple examples including
the agent-environment interface, reward signals, and trajectory generation.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt


# =============================================================================
# 1. Basic RL Abstractions
# =============================================================================

class Action(Enum):
    """Simple discrete action space."""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


@dataclass
class Transition:
    """A single transition in the agent-environment interaction."""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


@dataclass
class Episode:
    """A complete episode (trajectory) of agent-environment interaction."""
    transitions: List[Transition] = field(default_factory=list)

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool):
        self.transitions.append(Transition(state, action, reward, next_state, done))

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        return len(self.transitions)

    def compute_return(self, gamma: float = 0.99) -> float:
        """Compute discounted return G_0 = sum_{k=0}^{T-1} gamma^k * R_{k+1}."""
        G = 0.0
        for t in reversed(self.transitions):
            G = t.reward + gamma * G
        return G

    def compute_returns_at_each_step(self, gamma: float = 0.99) -> List[float]:
        """Compute G_t for each time step t."""
        returns = [0.0] * len(self.transitions)
        G = 0.0
        for i in reversed(range(len(self.transitions))):
            G = self.transitions[i].reward + gamma * G
            returns[i] = G
        return returns


# =============================================================================
# 2. Simple Grid World Environment
# =============================================================================

class SimpleGridWorld:
    """
    A simple 4x4 grid world environment demonstrating the agent-environment interface.

    Grid layout:
        S . . .
        . X . .
        . . . .
        . . . G

    S = Start (0,0), G = Goal (3,3), X = Wall (1,1)
    The agent receives -1 for each step, +10 for reaching the goal,
    and -5 for hitting a wall.
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = {(1, 1)}
        self.state = self.start

        # Action mappings: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
        self.action_effects = {
            0: (0, -1),   # LEFT
            1: (0, 1),    # RIGHT
            2: (-1, 0),   # UP
            3: (1, 0),    # DOWN
        }

    def reset(self) -> int:
        """Reset environment to start state. Returns state index."""
        self.state = self.start
        return self._state_to_index(self.state)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute action and return (next_state, reward, done).
        This is the core agent-environment interface.
        """
        row, col = self.state
        dr, dc = self.action_effects[action]
        new_row, new_col = row + dr, col + dc

        # Check boundaries
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            new_row, new_col = row, col  # Stay in place

        # Check walls
        if (new_row, new_col) in self.walls:
            reward = -5.0
            new_row, new_col = row, col  # Bounce back
        elif (new_row, new_col) == self.goal:
            reward = 10.0
        else:
            reward = -1.0

        self.state = (new_row, new_col)
        done = self.state == self.goal
        return self._state_to_index(self.state), reward, done

    def _state_to_index(self, state: Tuple[int, int]) -> int:
        return state[0] * self.size + state[1]

    def _index_to_state(self, index: int) -> Tuple[int, int]:
        return (index // self.size, index % self.size)

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


# =============================================================================
# 3. Simple Policies
# =============================================================================

class RandomPolicy:
    """A uniformly random policy - selects actions with equal probability."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:
        return np.random.randint(self.n_actions)

    def action_prob(self, state: int, action: int) -> float:
        return 1.0 / self.n_actions


class GreedyPolicy:
    """A greedy policy that always moves toward the goal."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.goal = (grid_size - 1, grid_size - 1)

    def select_action(self, state: int) -> int:
        row, col = state // self.grid_size, state % self.grid_size
        goal_row, goal_col = self.goal

        # Prefer moving toward goal (simple heuristic)
        if row < goal_row:
            return 3  # DOWN
        elif col < goal_col:
            return 1  # RIGHT
        elif row > goal_row:
            return 2  # UP
        else:
            return 0  # LEFT


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy: explore with probability epsilon, exploit otherwise."""

    def __init__(self, greedy_policy, n_actions: int, epsilon: float = 0.1):
        self.greedy_policy = greedy_policy
        self.n_actions = n_actions
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.greedy_policy.select_action(state)


# =============================================================================
# 4. Episode Generation and Return Computation
# =============================================================================

def generate_episode(env: SimpleGridWorld, policy, max_steps: int = 100) -> Episode:
    """
    Generate a complete episode using the given policy.
    Demonstrates the agent-environment interaction loop.
    """
    episode = Episode()
    state = env.reset()

    for _ in range(max_steps):
        action = policy.select_action(state)
        next_state, reward, done = env.step(action)
        episode.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    return episode


def demonstrate_return_computation():
    """Demonstrate different return computations."""
    # Simple reward sequence
    rewards = [1.0, 0.0, -1.0, 2.0, 5.0]

    print("=" * 60)
    print("Return Computation Demonstration")
    print("=" * 60)
    print(f"\nReward sequence: {rewards}")

    for gamma in [1.0, 0.99, 0.9, 0.5, 0.0]:
        # Compute G_0
        G = 0.0
        for k, r in enumerate(rewards):
            G += (gamma ** k) * r
        print(f"  γ = {gamma:.2f} → G_0 = {G:.4f}")

    # Show recursive property: G_t = R_{t+1} + γ * G_{t+1}
    gamma = 0.9
    print(f"\nRecursive property (γ = {gamma}):")
    returns = [0.0] * len(rewards)
    returns[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    for t, (r, G) in enumerate(zip(rewards, returns)):
        print(f"  t={t}: R_{t+1} = {r:+.1f}, G_t = {G:.4f}")
        if t < len(rewards) - 1:
            check = r + gamma * returns[t + 1]
            print(f"         Check: {r} + {gamma} * {returns[t+1]:.4f} = {check:.4f} ✓")


# =============================================================================
# 5. Comparing Policies
# =============================================================================

def compare_policies():
    """Compare different policies on the grid world."""
    env = SimpleGridWorld(size=4)
    n_episodes = 1000
    gamma = 0.99

    policies = {
        "Random": RandomPolicy(n_actions=4),
        "Greedy": GreedyPolicy(grid_size=4),
        "ε-Greedy (ε=0.3)": EpsilonGreedyPolicy(
            GreedyPolicy(grid_size=4), n_actions=4, epsilon=0.3
        ),
    }

    results = {}

    print("\n" + "=" * 60)
    print("Policy Comparison on 4x4 Grid World")
    print("=" * 60)

    for name, policy in policies.items():
        returns = []
        lengths = []
        for _ in range(n_episodes):
            episode = generate_episode(env, policy)
            returns.append(episode.compute_return(gamma))
            lengths.append(episode.length)

        results[name] = {
            "returns": returns,
            "lengths": lengths,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(lengths),
            "success_rate": sum(1 for ep_len in lengths if ep_len < 100) / n_episodes,
        }

        print(f"\n{name}:")
        print(f"  Mean Return:  {results[name]['mean_return']:.2f} ± {results[name]['std_return']:.2f}")
        print(f"  Mean Length:  {results[name]['mean_length']:.1f}")
        print(f"  Success Rate: {results[name]['success_rate']:.1%}")

    return results


# =============================================================================
# 6. Visualization
# =============================================================================

def plot_policy_comparison(results: Dict):
    """Plot comparison of policy performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Return distributions
    ax = axes[0]
    for name, data in results.items():
        ax.hist(data["returns"], bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Discounted Return (G₀)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution by Policy")
    ax.legend()

    # Episode lengths
    ax = axes[1]
    names = list(results.keys())
    mean_lengths = [results[n]["mean_length"] for n in names]
    ax.bar(names, mean_lengths, color=["#e74c3c", "#2ecc71", "#3498db"])
    ax.set_ylabel("Mean Episode Length")
    ax.set_title("Average Steps to Goal")
    ax.tick_params(axis='x', rotation=15)

    # Success rates
    ax = axes[2]
    success_rates = [results[n]["success_rate"] for n in names]
    ax.bar(names, success_rates, color=["#e74c3c", "#2ecc71", "#3498db"])
    ax.set_ylabel("Success Rate")
    ax.set_title("Fraction Reaching Goal")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig("rl_overview_policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: rl_overview_policy_comparison.png")


def plot_trajectory(env: SimpleGridWorld, policy, title: str = "Agent Trajectory"):
    """Visualize a single episode trajectory on the grid."""
    episode = generate_episode(env, policy)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Draw grid
    for i in range(env.size + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5)
        ax.axvline(x=i, color='gray', linewidth=0.5)

    # Draw walls
    for (r, c) in env.walls:
        ax.fill([c, c + 1, c + 1, c], [env.size - r - 1, env.size - r - 1, env.size - r, env.size - r],
                color='black', alpha=0.7)

    # Draw start and goal
    sr, sc = env.start
    ax.fill([sc, sc + 1, sc + 1, sc],
            [env.size - sr - 1, env.size - sr - 1, env.size - sr, env.size - sr],
            color='lightblue', alpha=0.5)
    ax.text(sc + 0.5, env.size - sr - 0.5, 'S', ha='center', va='center', fontsize=14, fontweight='bold')

    gr, gc = env.goal
    ax.fill([gc, gc + 1, gc + 1, gc],
            [env.size - gr - 1, env.size - gr - 1, env.size - gr, env.size - gr],
            color='lightgreen', alpha=0.5)
    ax.text(gc + 0.5, env.size - gr - 0.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw trajectory
    if episode.transitions:
        coords = []
        for t in episode.transitions:
            r, c = t.state // env.size, t.state % env.size
            coords.append((c + 0.5, env.size - r - 0.5))
        # Add final state
        last = episode.transitions[-1]
        r, c = last.next_state // env.size, last.next_state % env.size
        coords.append((c + 0.5, env.size - r - 0.5))

        xs, ys = zip(*coords)
        ax.plot(xs, ys, 'b-o', markersize=4, alpha=0.6, linewidth=1.5)
        # Arrow for direction
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            ax.annotate('', xy=coords[i + 1], xytext=coords[i],
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.4))

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\nSteps: {episode.length}, Return: {episode.compute_return(0.99):.2f}")

    filename = f"rl_overview_trajectory.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# 7. Financial RL Example: Simple Trading Environment
# =============================================================================

class SimpleTradingEnv:
    """
    Minimal trading environment to illustrate RL concepts in finance.

    State: [price_change, position, unrealized_pnl]
    Actions: 0=sell, 1=hold, 2=buy
    Reward: realized + unrealized P&L change - transaction costs
    """

    def __init__(self, prices: np.ndarray, transaction_cost: float = 0.001):
        self.prices = prices
        self.transaction_cost = transaction_cost
        self.t = 0
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0

    def reset(self) -> np.ndarray:
        self.t = 1  # Start at t=1 to have a price change
        self.position = 0
        self.entry_price = 0.0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # Map action: 0=sell(-1), 1=hold(0), 2=buy(+1)
        target_position = action - 1

        # Compute reward
        price_change = self.prices[self.t] - self.prices[self.t - 1]
        position_reward = self.position * price_change

        # Transaction cost for position changes
        trade_size = abs(target_position - self.position)
        cost = trade_size * self.transaction_cost * self.prices[self.t]

        reward = position_reward - cost

        # Update position
        if target_position != self.position:
            self.entry_price = self.prices[self.t]
        self.position = target_position

        # Advance time
        self.t += 1
        done = self.t >= len(self.prices) - 1

        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        price_change = (self.prices[self.t] - self.prices[self.t - 1]) / self.prices[self.t - 1]
        unrealized = self.position * (self.prices[self.t] - self.entry_price) if self.position != 0 else 0.0
        return np.array([price_change, self.position, unrealized])


def demonstrate_trading_env():
    """Run a simple trading demo."""
    np.random.seed(42)

    # Generate synthetic price series (geometric Brownian motion)
    n_steps = 100
    dt = 1.0 / 252
    mu = 0.10   # 10% annual drift
    sigma = 0.20  # 20% annual volatility
    S0 = 100.0

    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_steps)
    prices = S0 * np.cumprod(1 + returns)
    prices = np.insert(prices, 0, S0)

    env = SimpleTradingEnv(prices, transaction_cost=0.001)

    # Simple momentum policy: buy if price went up, sell if down
    state = env.reset()
    total_reward = 0.0
    actions_taken = []

    for _ in range(n_steps - 1):
        # Momentum strategy
        if state[0] > 0.001:  # Price went up
            action = 2  # Buy
        elif state[0] < -0.001:  # Price went down
            action = 0  # Sell
        else:
            action = 1  # Hold

        next_state, reward, done = env.step(action)
        total_reward += reward
        actions_taken.append(action)
        state = next_state
        if done:
            break

    print("\n" + "=" * 60)
    print("Simple Trading Environment Demo")
    print("=" * 60)
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    print(f"Total reward (momentum strategy): {total_reward:.4f}")
    print(f"Buy/Hold/Sell actions: {actions_taken.count(2)}/{actions_taken.count(1)}/{actions_taken.count(0)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Demonstrate return computation
    demonstrate_return_computation()

    # 2. Compare policies
    results = compare_policies()

    # 3. Visualizations
    plot_policy_comparison(results)

    env = SimpleGridWorld(size=4)
    plot_trajectory(env, GreedyPolicy(grid_size=4), title="Greedy Policy Trajectory")

    # 4. Trading environment demo
    demonstrate_trading_env()

    print("\n✓ All demonstrations complete.")
