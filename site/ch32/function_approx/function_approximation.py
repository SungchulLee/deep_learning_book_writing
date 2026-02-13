"""
Chapter 32.8: Function Approximation in RL
============================================
Linear function approximation, tile coding, semi-gradient TD,
and convergence demonstrations.
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


# =============================================================================
# 1. Feature Representations
# =============================================================================

class PolynomialFeatures:
    """Polynomial feature expansion."""

    def __init__(self, order: int = 2):
        self.order = order

    def __call__(self, state: np.ndarray) -> np.ndarray:
        features = [1.0]  # Bias
        for o in range(1, self.order + 1):
            for s in state:
                features.append(s ** o)
        # Cross terms for order 2
        if self.order >= 2 and len(state) >= 2:
            for i in range(len(state)):
                for j in range(i + 1, len(state)):
                    features.append(state[i] * state[j])
        return np.array(features)

    @property
    def dim(self):
        return None  # Depends on state dim


class TileCoding:
    """Tile coding feature representation."""

    def __init__(self, n_tilings: int = 8, n_tiles: int = 8,
                 state_low: np.ndarray = None, state_high: np.ndarray = None):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.state_low = state_low if state_low is not None else np.array([-1.0, -1.0])
        self.state_high = state_high if state_high is not None else np.array([1.0, 1.0])
        self.state_dim = len(self.state_low)
        self.total_tiles = n_tilings * (n_tiles ** self.state_dim)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        features = np.zeros(self.total_tiles)
        scaled = (state - self.state_low) / (self.state_high - self.state_low)
        scaled = np.clip(scaled, 0, 0.999)

        for tiling in range(self.n_tilings):
            offset = tiling / self.n_tilings
            tile_indices = ((scaled + offset) * self.n_tiles).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.n_tiles - 1)
            idx = tiling * (self.n_tiles ** self.state_dim)
            for d in range(self.state_dim):
                idx += tile_indices[d] * (self.n_tiles ** d)
            features[idx] = 1.0

        return features

    @property
    def dim(self):
        return self.total_tiles


class RBFFeatures:
    """Radial Basis Function features."""

    def __init__(self, centers: np.ndarray, sigma: float = 0.5):
        self.centers = centers
        self.sigma = sigma

    def __call__(self, state: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(self.centers - state, axis=1)
        return np.exp(-dists ** 2 / (2 * self.sigma ** 2))

    @property
    def dim(self):
        return len(self.centers)


# =============================================================================
# 2. Mountain Car Environment
# =============================================================================

class MountainCarEnv:
    """Continuous-state Mountain Car. State: (position, velocity)."""

    def __init__(self):
        self.min_pos, self.max_pos = -1.2, 0.5
        self.min_vel, self.max_vel = -0.07, 0.07
        self.goal_pos = 0.5

    def reset(self):
        self.pos = np.random.uniform(-0.6, -0.4)
        self.vel = 0.0
        return np.array([self.pos, self.vel])

    def step(self, action):
        # action: 0=left, 1=none, 2=right
        force = action - 1  # {-1, 0, 1}
        self.vel += 0.001 * force - 0.0025 * np.cos(3 * self.pos)
        self.vel = np.clip(self.vel, self.min_vel, self.max_vel)
        self.pos += self.vel
        self.pos = np.clip(self.pos, self.min_pos, self.max_pos)
        if self.pos == self.min_pos:
            self.vel = 0.0
        done = self.pos >= self.goal_pos
        return np.array([self.pos, self.vel]), -1.0, done

    @property
    def n_actions(self):
        return 3


# =============================================================================
# 3. Semi-Gradient TD(0) with Linear FA
# =============================================================================

class LinearTD:
    """Semi-gradient TD(0) with linear function approximation for V."""

    def __init__(self, feature_fn, n_features: int, alpha: float = 0.01,
                 gamma: float = 1.0):
        self.feature_fn = feature_fn
        self.w = np.zeros(n_features)
        self.alpha = alpha
        self.gamma = gamma

    def predict(self, state: np.ndarray) -> float:
        return np.dot(self.w, self.feature_fn(state))

    def update(self, state, reward, next_state, done):
        x = self.feature_fn(state)
        v_next = 0.0 if done else self.predict(next_state)
        td_error = reward + self.gamma * v_next - np.dot(self.w, x)
        self.w += self.alpha * td_error * x
        return td_error


# =============================================================================
# 4. Semi-Gradient SARSA with Linear FA
# =============================================================================

class LinearSARSA:
    """Semi-gradient SARSA with linear function approximation for Q."""

    def __init__(self, feature_fn, n_features: int, n_actions: int,
                 alpha: float = 0.01, gamma: float = 1.0, epsilon: float = 0.1):
        self.feature_fn = feature_fn
        self.n_actions = n_actions
        self.w = np.zeros((n_actions, n_features))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def q_value(self, state, action):
        return np.dot(self.w[action], self.feature_fn(state))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = [self.q_value(state, a) for a in range(self.n_actions)]
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, next_action, done):
        x = self.feature_fn(state)
        q_next = 0.0 if done else self.q_value(next_state, next_action)
        td_error = reward + self.gamma * q_next - np.dot(self.w[action], x)
        self.w[action] += self.alpha * td_error * x
        return td_error


# =============================================================================
# 5. Training Loop
# =============================================================================

def train_mountain_car(n_episodes=500, alpha=0.01):
    """Train linear SARSA on Mountain Car."""
    env = MountainCarEnv()
    tiles = TileCoding(
        n_tilings=8, n_tiles=8,
        state_low=np.array([-1.2, -0.07]),
        state_high=np.array([0.5, 0.07])
    )
    agent = LinearSARSA(tiles, tiles.dim, env.n_actions,
                         alpha=alpha / 8, gamma=1.0, epsilon=0.1)

    episode_lengths = []

    for ep in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        steps = 0

        for _ in range(1000):
            next_state, reward, done = env.step(action)
            steps += 1
            next_action = agent.select_action(next_state) if not done else 0
            agent.update(state, action, reward, next_state, next_action, done)
            if done:
                break
            state, action = next_state, next_action

        episode_lengths.append(steps)

    return agent, episode_lengths


# =============================================================================
# 6. Demonstrations
# =============================================================================

def demo_feature_types():
    """Demonstrate different feature representations."""
    print("=" * 60)
    print("Feature Representations")
    print("=" * 60)

    state = np.array([0.3, -0.02])
    print(f"\nState: {state}")

    # Polynomial
    poly = PolynomialFeatures(order=2)
    x_poly = poly(state)
    print(f"\nPolynomial (order 2): dim={len(x_poly)}")
    print(f"  Features: {x_poly.round(4)}")

    # Tile coding
    tiles = TileCoding(n_tilings=4, n_tiles=4,
                       state_low=np.array([-1.2, -0.07]),
                       state_high=np.array([0.5, 0.07]))
    x_tiles = tiles(state)
    active = np.where(x_tiles > 0)[0]
    print(f"\nTile coding (4 tilings, 4x4): dim={tiles.dim}")
    print(f"  Active tiles: {active} ({len(active)} active)")

    # RBF
    centers = np.random.uniform(-1, 1, (10, 2))
    rbf = RBFFeatures(centers, sigma=0.5)
    x_rbf = rbf(state)
    print(f"\nRBF (10 centers): dim={rbf.dim}")
    print(f"  Features: {x_rbf.round(4)}")


def demo_mountain_car():
    """Train and evaluate on Mountain Car."""
    print("\n" + "=" * 60)
    print("Mountain Car: Linear SARSA with Tile Coding")
    print("=" * 60)

    agent, lengths = train_mountain_car(n_episodes=500, alpha=0.05)

    # Report progress
    windows = [(0, 50), (50, 100), (100, 200), (200, 500)]
    for start, end in windows:
        avg = np.mean(lengths[start:end])
        print(f"  Episodes {start}-{end}: avg steps = {avg:.0f}")

    return agent, lengths


def demo_convergence_issue():
    """Demonstrate potential divergence with off-policy + FA + bootstrapping."""
    print("\n" + "=" * 60)
    print("Deadly Triad: Divergence Demonstration")
    print("=" * 60)

    # Simple 2-state example that can diverge with off-policy linear TD
    # State 0: features [1, 2], State 1: features [2, 1]
    gamma = 0.99
    alpha = 0.01

    w = np.array([1.0, 1.0])
    features = {0: np.array([1.0, 2.0]), 1: np.array([2.0, 1.0])}

    # On-policy (50/50 visits) - should converge
    w_on = w.copy()
    on_policy_norms = []
    for t in range(1000):
        s = np.random.choice([0, 1])
        s_next = 1 - s
        r = 0.0
        x = features[s]
        x_next = features[s_next]
        td_error = r + gamma * np.dot(w_on, x_next) - np.dot(w_on, x)
        w_on += alpha * td_error * x
        on_policy_norms.append(np.linalg.norm(w_on))

    # Off-policy (always visit state 0, learn about both)
    w_off = w.copy()
    off_policy_norms = []
    for t in range(1000):
        s = 0  # Always visit state 0
        s_next = 1
        r = 0.0
        x = features[s]
        x_next = features[s_next]
        td_error = r + gamma * np.dot(w_off, x_next) - np.dot(w_off, x)
        w_off += alpha * td_error * x
        off_policy_norms.append(np.linalg.norm(w_off))

    print(f"\nOn-policy:  ||w|| final = {on_policy_norms[-1]:.4f}")
    print(f"Off-policy: ||w|| final = {off_policy_norms[-1]:.4f}")
    print(f"Off-policy diverging: {off_policy_norms[-1] > 10 * on_policy_norms[-1]}")

    return on_policy_norms, off_policy_norms


# =============================================================================
# 7. Visualization
# =============================================================================

def plot_mountain_car_learning(lengths):
    """Plot Mountain Car learning curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lengths, alpha=0.3, linewidth=0.5, color='blue')
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(lengths)), smoothed, 'r-', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Goal')
    ax.set_title('Mountain Car: Linear SARSA with Tile Coding')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fa_mountain_car_learning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fa_mountain_car_learning.png")


def plot_convergence_demo(on_norms, off_norms):
    """Plot on-policy vs off-policy weight norms."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(on_norms, label='On-policy', linewidth=1.5)
    ax.plot(off_norms, label='Off-policy', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('||w||')
    ax.set_title('Deadly Triad: On-Policy (stable) vs Off-Policy (potentially unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fa_convergence_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fa_convergence_demo.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Feature representations
    demo_feature_types()

    # 2. Mountain Car training
    agent, lengths = demo_mountain_car()
    plot_mountain_car_learning(lengths)

    # 3. Convergence issues
    on_norms, off_norms = demo_convergence_issue()
    plot_convergence_demo(on_norms, off_norms)

    print("\n✓ Function Approximation demonstrations complete.")
