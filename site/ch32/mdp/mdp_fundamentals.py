"""
Chapter 32.2: Markov Decision Processes
========================================
Implements MDP data structures, transition dynamics, reward functions,
discount factor analysis, and financial MDP examples.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


# =============================================================================
# 1. MDP Data Structure
# =============================================================================

class FiniteMDP:
    """
    A finite Markov Decision Process defined by (S, A, P, R, gamma).

    Parameters
    ----------
    n_states : int
        Number of states |S|
    n_actions : int
        Number of actions |A|
    transitions : np.ndarray
        P[s, a, s'] = Pr(S'=s' | S=s, A=a), shape (n_states, n_actions, n_states)
    rewards : np.ndarray
        R[s, a] = E[R | S=s, A=a], shape (n_states, n_actions)
    gamma : float
        Discount factor
    terminal_states : set
        Set of terminal (absorbing) state indices
    """

    def __init__(self, n_states: int, n_actions: int,
                 transitions: np.ndarray, rewards: np.ndarray,
                 gamma: float = 0.99, terminal_states: set = None):
        assert transitions.shape == (n_states, n_actions, n_states)
        assert rewards.shape == (n_states, n_actions)
        # Verify row-stochastic
        assert np.allclose(transitions.sum(axis=2), 1.0), "Transitions must sum to 1"
        assert np.all(transitions >= 0), "Transitions must be non-negative"

        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transitions
        self.R = rewards
        self.gamma = gamma
        self.terminal_states = terminal_states or set()

    def transition_matrix_for_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute P_π[s, s'] = Σ_a π(a|s) P(s'|s,a)
        policy shape: (n_states, n_actions) — stochastic policy
        """
        P_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P_pi[s] += policy[s, a] * self.P[s, a]
        return P_pi

    def reward_vector_for_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute r_π[s] = Σ_a π(a|s) R(s,a)
        """
        return np.sum(policy * self.R, axis=1)

    def expected_reward(self, state: int, action: int) -> float:
        """R(s,a) = E[R_{t+1} | S_t=s, A_t=a]"""
        return self.R[state, action]

    def sample_transition(self, state: int, action: int) -> Tuple[int, float]:
        """Sample next state and reward from the MDP."""
        next_state = np.random.choice(self.n_states, p=self.P[state, action])
        reward = self.R[state, action]
        return next_state, reward

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    def stationary_distribution(self, policy: np.ndarray) -> np.ndarray:
        """Compute stationary distribution d_π via eigenvalue decomposition."""
        P_pi = self.transition_matrix_for_policy(policy)
        # Left eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        d = np.real(eigenvectors[:, idx])
        d = d / d.sum()  # Normalize
        return np.abs(d)  # Ensure non-negative

    def print_summary(self):
        print(f"MDP: {self.n_states} states, {self.n_actions} actions, γ={self.gamma}")
        print(f"Terminal states: {self.terminal_states}")
        print(f"Reward range: [{self.R.min():.2f}, {self.R.max():.2f}]")


# =============================================================================
# 2. Example MDP: Simple Grid World
# =============================================================================

def create_gridworld_mdp(size: int = 4, gamma: float = 0.99) -> FiniteMDP:
    """
    Create a grid world MDP.
    Goal at (size-1, size-1) with reward +10.
    Step cost of -1. Wall at (1,1) with penalty -5.
    """
    n_states = size * size
    n_actions = 4  # UP, RIGHT, DOWN, LEFT
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    goal = (size - 1, size - 1)
    goal_idx = goal[0] * size + goal[1]
    walls = {(1, 1)}

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for r in range(size):
        for c in range(size):
            s = r * size + c
            if (r, c) == goal:
                # Terminal / absorbing state
                for a in range(n_actions):
                    P[s, a, s] = 1.0
                    R[s, a] = 0.0
                continue

            for a, (dr, dc) in enumerate(deltas):
                nr, nc = r + dr, c + dc
                # Boundary check
                if not (0 <= nr < size and 0 <= nc < size):
                    nr, nc = r, c
                # Wall check
                if (nr, nc) in walls:
                    P[s, a, s] = 1.0  # Bounce back
                    R[s, a] = -5.0
                elif (nr, nc) == goal:
                    ns = nr * size + nc
                    P[s, a, ns] = 1.0
                    R[s, a] = 10.0
                else:
                    ns = nr * size + nc
                    P[s, a, ns] = 1.0
                    R[s, a] = -1.0

    return FiniteMDP(n_states, n_actions, P, R, gamma, terminal_states={goal_idx})


# =============================================================================
# 3. Example MDP: Stochastic Wind Grid (demonstrates stochastic transitions)
# =============================================================================

def create_stochastic_gridworld(size: int = 4, wind_prob: float = 0.2,
                                 gamma: float = 0.99) -> FiniteMDP:
    """
    Grid world with stochastic wind. With probability wind_prob,
    the agent is pushed to a random adjacent cell instead of moving as intended.
    """
    n_states = size * size
    n_actions = 4
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    goal = (size - 1, size - 1)
    goal_idx = goal[0] * size + goal[1]

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    def clip_state(r, c):
        return max(0, min(size - 1, r)), max(0, min(size - 1, c))

    for r in range(size):
        for c in range(size):
            s = r * size + c
            if (r, c) == goal:
                for a in range(n_actions):
                    P[s, a, s] = 1.0
                continue

            for a, (dr, dc) in enumerate(deltas):
                # Intended move
                nr, nc = clip_state(r + dr, c + dc)
                intended_ns = nr * size + nc

                # Wind: random adjacent cell
                wind_states = []
                for wd, wc in deltas:
                    wr, wcc = clip_state(r + wd, c + wc)
                    wind_states.append(wr * size + wcc)

                # Assign probabilities
                P[s, a, intended_ns] += (1 - wind_prob)
                for ws in wind_states:
                    P[s, a, ws] += wind_prob / len(wind_states)

                # Reward
                if (nr, nc) == goal:
                    R[s, a] = 10.0
                else:
                    R[s, a] = -1.0

    return FiniteMDP(n_states, n_actions, P, R, gamma, terminal_states={goal_idx})


# =============================================================================
# 4. Financial MDP: Simple Trading
# =============================================================================

def create_trading_mdp(n_price_states: int = 5, gamma: float = 0.99) -> FiniteMDP:
    """
    Simple trading MDP with discrete price levels and positions.

    States: (price_level, position) where
        price_level ∈ {0, 1, ..., n_price_states-1}
        position ∈ {-1 (short), 0 (flat), 1 (long)}
    Actions: 0=sell, 1=hold, 2=buy

    Transitions: Price moves up/down with state-dependent probabilities
    (momentum effect: current direction slightly persists).
    """
    n_positions = 3  # short, flat, long
    n_states = n_price_states * n_positions
    n_actions = 3  # sell, hold, buy

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    tc = 0.5  # Transaction cost

    def state_idx(price, pos):
        return price * n_positions + (pos + 1)  # pos ∈ {-1,0,1} → {0,1,2}

    def decode_state(s):
        price = s // n_positions
        pos = (s % n_positions) - 1
        return price, pos

    for s in range(n_states):
        price, pos = decode_state(s)

        for a in range(n_actions):
            new_pos = a - 1  # 0→-1(sell), 1→0(hold), 2→1(buy)

            # Transaction cost for position change
            trade_cost = tc * abs(new_pos - pos)

            # Price transitions (mean-reverting with momentum)
            mid = n_price_states // 2
            # Slight mean reversion
            if price > mid:
                p_up, p_down = 0.35, 0.45
            elif price < mid:
                p_up, p_down = 0.45, 0.35
            else:
                p_up, p_down = 0.40, 0.40
            p_stay = 1.0 - p_up - p_down

            for next_price_delta, prob in [(-1, p_down), (0, p_stay), (1, p_up)]:
                next_price = np.clip(price + next_price_delta, 0, n_price_states - 1)
                ns = state_idx(next_price, new_pos)

                P[s, a, ns] += prob

                # Reward: position * price_change - transaction_cost
                price_change = next_price_delta * 1.0
                R[s, a] += prob * (new_pos * price_change - trade_cost)

    return FiniteMDP(n_states, n_actions, P, R, gamma)


# =============================================================================
# 5. MDP Analysis Functions
# =============================================================================

def analyze_transition_properties(mdp: FiniteMDP, policy: np.ndarray):
    """Analyze properties of the transition matrix under a policy."""
    P_pi = mdp.transition_matrix_for_policy(policy)

    print("\n" + "=" * 60)
    print("Transition Matrix Analysis")
    print("=" * 60)

    # Check stochasticity
    row_sums = P_pi.sum(axis=1)
    print(f"Row sums (should be 1): min={row_sums.min():.6f}, max={row_sums.max():.6f}")

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(P_pi)
    eigenvalues_sorted = sorted(eigenvalues, key=lambda x: -abs(x))
    print(f"Top 5 eigenvalues: {[f'{e:.4f}' for e in eigenvalues_sorted[:5]]}")

    # Stationary distribution
    d = mdp.stationary_distribution(policy)
    print(f"Stationary distribution entropy: {-np.sum(d * np.log(d + 1e-10)):.4f}")
    print(f"Most visited state: {np.argmax(d)} (prob={d.max():.4f})")

    # Mixing time estimate (spectral gap)
    abs_eigenvalues = sorted(np.abs(eigenvalues), reverse=True)
    if len(abs_eigenvalues) > 1 and abs_eigenvalues[1] < 1:
        spectral_gap = 1 - abs_eigenvalues[1]
        mixing_time = 1.0 / spectral_gap
        print(f"Spectral gap: {spectral_gap:.4f}")
        print(f"Estimated mixing time: {mixing_time:.1f} steps")

    return P_pi, d


def analyze_discount_effects(mdp: FiniteMDP, state: int = 0):
    """Analyze how discount factor affects value estimates."""
    print("\n" + "=" * 60)
    print("Discount Factor Effects")
    print("=" * 60)

    # Use random policy
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions

    gammas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]

    for gamma in gammas:
        # Compute V_π via matrix inversion: V = (I - γP_π)^{-1} r_π
        P_pi = mdp.transition_matrix_for_policy(policy)
        r_pi = mdp.reward_vector_for_policy(policy)

        try:
            I = np.eye(mdp.n_states)
            V = np.linalg.solve(I - gamma * P_pi, r_pi)
            print(f"  γ={gamma:.3f}: V(s={state})={V[state]:.4f}, "
                  f"mean(V)={V.mean():.4f}, max(V)={V.max():.4f}")
        except np.linalg.LinAlgError:
            print(f"  γ={gamma:.3f}: Singular matrix (γ=1 for continuing task)")


# =============================================================================
# 6. Simulation
# =============================================================================

class MDPSimulator:
    """Simulate episodes from an MDP."""

    def __init__(self, mdp: FiniteMDP):
        self.mdp = mdp

    def simulate_episode(self, policy: np.ndarray, start_state: int = 0,
                         max_steps: int = 1000) -> Dict:
        states, actions, rewards = [start_state], [], []
        state = start_state

        for t in range(max_steps):
            if self.mdp.is_terminal(state):
                break
            action = np.random.choice(self.mdp.n_actions, p=policy[state])
            next_state, reward = self.mdp.sample_transition(state, action)
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            state = next_state

        return {"states": states, "actions": actions, "rewards": rewards,
                "length": len(actions),
                "return": sum(self.mdp.gamma**t * r for t, r in enumerate(rewards))}

    def monte_carlo_value(self, policy: np.ndarray, n_episodes: int = 1000,
                           start_state: int = 0) -> float:
        """Estimate V_π(s) via Monte Carlo."""
        returns = []
        for _ in range(n_episodes):
            ep = self.simulate_episode(policy, start_state)
            returns.append(ep["return"])
        return np.mean(returns)


# =============================================================================
# 7. Visualization
# =============================================================================

def plot_transition_heatmap(mdp: FiniteMDP, action: int = 0):
    """Visualize transition matrix for a specific action."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mdp.P[:, action, :], cmap='Blues', aspect='auto')
    ax.set_xlabel("Next State s'")
    ax.set_ylabel("Current State s")
    ax.set_title(f"Transition Probabilities P(s'|s, a={action})")
    plt.colorbar(im, label="Probability")
    plt.tight_layout()
    plt.savefig("mdp_transition_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_transition_heatmap.png")


def plot_reward_structure(mdp: FiniteMDP):
    """Visualize reward function."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mdp.R, cmap='RdYlGn', aspect='auto')
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title("Reward Function R(s, a)")
    plt.colorbar(im, label="Expected Reward")
    plt.tight_layout()
    plt.savefig("mdp_reward_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_reward_structure.png")


def plot_discount_analysis():
    """Visualize discount factor effects."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Discount weights
    steps = np.arange(100)
    ax = axes[0]
    for gamma in [0.5, 0.9, 0.95, 0.99]:
        ax.plot(steps, gamma**steps, label=f'γ={gamma}')
    ax.set_xlabel('Steps into Future')
    ax.set_ylabel('Weight γ^k')
    ax.set_title('Discount Weights')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Effective horizon
    ax = axes[1]
    gammas = np.linspace(0.01, 0.999, 200)
    ax.plot(gammas, 1 / (1 - gammas))
    ax.set_xlabel('γ')
    ax.set_ylabel('1/(1-γ)')
    ax.set_title('Effective Horizon')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Max possible return
    ax = axes[2]
    R_max = 10
    for gamma in [0.5, 0.9, 0.95, 0.99]:
        horizon = np.arange(1, 200)
        partial_sum = R_max * (1 - gamma**horizon) / (1 - gamma)
        ax.plot(horizon, partial_sum, label=f'γ={gamma}')
    ax.axhline(y=R_max / (1 - 0.99), color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Horizon (steps)')
    ax.set_ylabel('Partial Return')
    ax.set_title(f'Cumulative Discounted Reward (R_max={R_max})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mdp_discount_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_discount_analysis.png")


def plot_stationary_distribution(mdp: FiniteMDP, policy: np.ndarray):
    """Plot stationary distribution."""
    d = mdp.stationary_distribution(policy)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(mdp.n_states), d, color='steelblue', alpha=0.7)
    ax.set_xlabel('State')
    ax.set_ylabel('Stationary Probability d_π(s)')
    ax.set_title('Stationary Distribution Under Random Policy')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("mdp_stationary_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_stationary_dist.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Create and analyze deterministic grid world
    print("=" * 60)
    print("1. Deterministic Grid World MDP")
    print("=" * 60)
    mdp = create_gridworld_mdp(size=4, gamma=0.99)
    mdp.print_summary()

    # Random policy
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    P_pi, d = analyze_transition_properties(mdp, policy)

    # 2. Stochastic grid world
    print("\n" + "=" * 60)
    print("2. Stochastic Grid World MDP (wind_prob=0.2)")
    print("=" * 60)
    stoch_mdp = create_stochastic_gridworld(size=4, wind_prob=0.2)
    stoch_mdp.print_summary()
    analyze_transition_properties(stoch_mdp, policy)

    # 3. Trading MDP
    print("\n" + "=" * 60)
    print("3. Trading MDP")
    print("=" * 60)
    trade_mdp = create_trading_mdp(n_price_states=5, gamma=0.99)
    trade_mdp.print_summary()
    trade_policy = np.ones((trade_mdp.n_states, trade_mdp.n_actions)) / trade_mdp.n_actions
    analyze_transition_properties(trade_mdp, trade_policy)

    # 4. Discount factor effects
    analyze_discount_effects(mdp, state=0)

    # 5. Monte Carlo simulation
    print("\n" + "=" * 60)
    print("5. Monte Carlo Value Estimation")
    print("=" * 60)
    sim = MDPSimulator(mdp)
    for start in [0, 4, 8]:
        mc_value = sim.monte_carlo_value(policy, n_episodes=5000, start_state=start)
        print(f"  V_π(s={start}) ≈ {mc_value:.4f}")

    # 6. Visualizations
    plot_transition_heatmap(stoch_mdp, action=1)
    plot_reward_structure(mdp)
    plot_discount_analysis()
    plot_stationary_distribution(stoch_mdp, policy)

    print("\n✓ MDP demonstrations complete.")
