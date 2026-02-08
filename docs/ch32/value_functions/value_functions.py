"""
Chapter 32.3: Value Functions and Bellman Equations
====================================================
Implements state value functions, action value functions, advantage functions,
Bellman equations (evaluation and optimality), and financial interpretations.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


# =============================================================================
# 1. MDP Definition (reusable)
# =============================================================================

class FiniteMDP:
    """Finite MDP for value function computation."""

    def __init__(self, n_states: int, n_actions: int,
                 P: np.ndarray, R: np.ndarray, gamma: float = 0.99,
                 terminal_states: set = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P          # (S, A, S')
        self.R = R          # (S, A)
        self.gamma = gamma
        self.terminal_states = terminal_states or set()

    def transition_matrix_policy(self, policy: np.ndarray) -> np.ndarray:
        """P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)"""
        P_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P_pi[s] += policy[s, a] * self.P[s, a]
        return P_pi

    def reward_vector_policy(self, policy: np.ndarray) -> np.ndarray:
        """r_pi[s] = sum_a pi(a|s) R(s,a)"""
        return np.sum(policy * self.R, axis=1)


def create_gridworld(size: int = 4, gamma: float = 0.99) -> FiniteMDP:
    """Create 4x4 grid world with goal at bottom-right."""
    n_states = size * size
    n_actions = 4
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
    goal = (size-1, size-1)
    goal_idx = goal[0]*size + goal[1]

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for r in range(size):
        for c in range(size):
            s = r * size + c
            if (r, c) == goal:
                for a in range(n_actions):
                    P[s, a, s] = 1.0
                continue
            for a, (dr, dc) in enumerate(deltas):
                nr = max(0, min(size-1, r+dr))
                nc = max(0, min(size-1, c+dc))
                ns = nr * size + nc
                P[s, a, ns] = 1.0
                R[s, a] = 10.0 if (nr, nc) == goal else -1.0

    return FiniteMDP(n_states, n_actions, P, R, gamma, {goal_idx})


# =============================================================================
# 2. State Value Function V_pi
# =============================================================================

def compute_V_pi_exact(mdp: FiniteMDP, policy: np.ndarray) -> np.ndarray:
    """
    Compute V_pi exactly via matrix inversion:
    V_pi = (I - gamma * P_pi)^{-1} r_pi
    """
    P_pi = mdp.transition_matrix_policy(policy)
    r_pi = mdp.reward_vector_policy(policy)
    I = np.eye(mdp.n_states)
    V = np.linalg.solve(I - mdp.gamma * P_pi, r_pi)
    return V


def compute_V_pi_iterative(mdp: FiniteMDP, policy: np.ndarray,
                            tol: float = 1e-8, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    """
    Compute V_pi via iterative Bellman equation application.
    V_{k+1}(s) = sum_a pi(a|s) [R(s,a) + gamma * sum_s' P(s'|s,a) V_k(s')]
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        V_new = np.zeros(mdp.n_states)
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                V_new[s] += policy[s, a] * (
                    mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
                )
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            return V, iteration + 1

    return V, max_iter


def compute_V_pi_monte_carlo(mdp: FiniteMDP, policy: np.ndarray,
                              n_episodes: int = 5000,
                              max_steps: int = 200) -> np.ndarray:
    """Estimate V_pi via Monte Carlo sampling (first-visit)."""
    returns_sum = np.zeros(mdp.n_states)
    returns_count = np.zeros(mdp.n_states)

    for _ in range(n_episodes):
        # Generate episode from random start
        state = np.random.randint(mdp.n_states)
        states, rewards = [state], []

        for _ in range(max_steps):
            if state in mdp.terminal_states:
                break
            action = np.random.choice(mdp.n_actions, p=policy[state])
            next_state = np.random.choice(mdp.n_states, p=mdp.P[state, action])
            reward = mdp.R[state, action]
            rewards.append(reward)
            states.append(next_state)
            state = next_state

        # Compute returns and update (first-visit)
        G = 0.0
        visited = set()
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + mdp.gamma * G
            s = states[t]
            if s not in visited:
                visited.add(s)
                returns_sum[s] += G
                returns_count[s] += 1

    # Avoid division by zero
    mask = returns_count > 0
    V = np.zeros(mdp.n_states)
    V[mask] = returns_sum[mask] / returns_count[mask]
    return V


# =============================================================================
# 3. Action Value Function Q_pi
# =============================================================================

def compute_Q_pi(mdp: FiniteMDP, V_pi: np.ndarray) -> np.ndarray:
    """
    Compute Q_pi from V_pi using one-step lookahead:
    Q_pi(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V_pi(s')
    """
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            Q[s, a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V_pi)
    return Q


def compute_advantage(Q_pi: np.ndarray, V_pi: np.ndarray) -> np.ndarray:
    """
    Compute advantage function: A_pi(s,a) = Q_pi(s,a) - V_pi(s)
    """
    return Q_pi - V_pi[:, np.newaxis]


# =============================================================================
# 4. Bellman Optimality
# =============================================================================

def bellman_optimality_update(mdp: FiniteMDP, V: np.ndarray) -> np.ndarray:
    """
    One step of the Bellman optimality operator:
    V'(s) = max_a [R(s,a) + gamma * sum_s' P(s'|s,a) V(s')]
    """
    V_new = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        q_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            q_values[a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
        V_new[s] = np.max(q_values)
    return V_new


def compute_V_star(mdp: FiniteMDP, tol: float = 1e-8,
                    max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    """Compute V* via value iteration (repeated Bellman optimality backup)."""
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iter):
        V_new = bellman_optimality_update(mdp, V)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            return V, iteration + 1

    return V, max_iter


def compute_Q_star(mdp: FiniteMDP, V_star: np.ndarray) -> np.ndarray:
    """Compute Q* from V*."""
    return compute_Q_pi(mdp, V_star)


def extract_optimal_policy(mdp: FiniteMDP, Q_star: np.ndarray) -> np.ndarray:
    """
    Extract deterministic optimal policy from Q*.
    pi*(s) = argmax_a Q*(s,a)
    """
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    best_actions = np.argmax(Q_star, axis=1)
    for s in range(mdp.n_states):
        policy[s, best_actions[s]] = 1.0
    return policy


# =============================================================================
# 5. Contraction Mapping Demonstration
# =============================================================================

def demonstrate_contraction(mdp: FiniteMDP, n_iterations: int = 50):
    """Show that the Bellman operator is a gamma-contraction."""
    # Two different initial value functions
    V1 = np.random.randn(mdp.n_states) * 10
    V2 = np.random.randn(mdp.n_states) * 10

    distances = [np.max(np.abs(V1 - V2))]
    gamma_bound = [distances[0]]

    for i in range(n_iterations):
        V1 = bellman_optimality_update(mdp, V1)
        V2 = bellman_optimality_update(mdp, V2)
        dist = np.max(np.abs(V1 - V2))
        distances.append(dist)
        gamma_bound.append(distances[0] * mdp.gamma ** (i + 1))

    return distances, gamma_bound


# =============================================================================
# 6. Demonstrations
# =============================================================================

def demo_value_functions():
    """Compare different methods for computing value functions."""
    mdp = create_gridworld(size=4, gamma=0.99)

    # Random policy
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions

    print("=" * 65)
    print("Value Function Computation Methods Comparison")
    print("=" * 65)

    # Exact
    V_exact = compute_V_pi_exact(mdp, policy)
    print(f"\n1. Exact (matrix inversion):")
    print(f"   V_pi = {V_exact.round(2)}")

    # Iterative
    V_iter, iters = compute_V_pi_iterative(mdp, policy)
    print(f"\n2. Iterative ({iters} iterations):")
    print(f"   V_pi = {V_iter.round(2)}")
    print(f"   Max error vs exact: {np.max(np.abs(V_iter - V_exact)):.2e}")

    # Monte Carlo
    V_mc = compute_V_pi_monte_carlo(mdp, policy, n_episodes=10000)
    print(f"\n3. Monte Carlo (10000 episodes):")
    print(f"   V_pi = {V_mc.round(2)}")
    print(f"   Max error vs exact: {np.max(np.abs(V_mc - V_exact)):.2e}")

    # Q-function
    Q = compute_Q_pi(mdp, V_exact)
    print(f"\n4. Q-function Q_pi(s,a) [first 4 states]:")
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for s in range(4):
        q_str = ', '.join([f"{action_names[a]}:{Q[s,a]:.2f}" for a in range(4)])
        print(f"   s={s}: {q_str}")

    # Advantage
    A = compute_advantage(Q, V_exact)
    print(f"\n5. Advantage A_pi(s,a) [first 4 states]:")
    for s in range(4):
        a_str = ', '.join([f"{action_names[a]}:{A[s,a]:+.2f}" for a in range(4)])
        print(f"   s={s}: {a_str}")
    print(f"   Advantage sums to 0? {np.allclose(np.sum(A * policy, axis=1), 0)}")

    return mdp, V_exact, Q, A


def demo_bellman_optimality():
    """Demonstrate Bellman optimality equations and value iteration."""
    mdp = create_gridworld(size=4, gamma=0.99)

    print("\n" + "=" * 65)
    print("Bellman Optimality: V* and Optimal Policy")
    print("=" * 65)

    # Compute V*
    V_star, iters = compute_V_star(mdp)
    print(f"\nV* (converged in {iters} iterations):")
    V_grid = V_star.reshape(4, 4)
    for row in V_grid:
        print("  " + "  ".join(f"{v:7.2f}" for v in row))

    # Q*
    Q_star = compute_Q_star(mdp, V_star)

    # Optimal policy
    pi_star = extract_optimal_policy(mdp, Q_star)
    action_symbols = ['↑', '→', '↓', '←']
    print(f"\nOptimal Policy:")
    for r in range(4):
        row_str = ""
        for c in range(4):
            s = r * 4 + c
            best_a = np.argmax(pi_star[s])
            row_str += f"  {action_symbols[best_a]}  "
        print("  " + row_str)

    # Compare V* vs V_pi (random)
    policy_random = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    V_random = compute_V_pi_exact(mdp, policy_random)
    V_optimal = compute_V_pi_exact(mdp, pi_star)

    print(f"\nPerformance comparison:")
    print(f"  V_random(s=0) = {V_random[0]:.4f}")
    print(f"  V_optimal(s=0) = {V_optimal[0]:.4f}")
    print(f"  V*(s=0) = {V_star[0]:.4f}")
    print(f"  V_optimal matches V*? {np.allclose(V_optimal, V_star, atol=0.01)}")

    return V_star, Q_star, pi_star


# =============================================================================
# 7. Visualization
# =============================================================================

def plot_value_functions(mdp, V_pi, V_star, pi_star):
    """Visualize value functions and optimal policy."""
    size = 4
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # V_pi (random policy)
    ax = axes[0]
    im = ax.imshow(V_pi.reshape(size, size), cmap='RdYlGn', aspect='equal')
    for r in range(size):
        for c in range(size):
            ax.text(c, r, f'{V_pi[r*size+c]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_title('V_π (Random Policy)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # V*
    ax = axes[1]
    im = ax.imshow(V_star.reshape(size, size), cmap='RdYlGn', aspect='equal')
    for r in range(size):
        for c in range(size):
            ax.text(c, r, f'{V_star[r*size+c]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_title('V* (Optimal)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Optimal policy with V*
    ax = axes[2]
    action_symbols = ['↑', '→', '↓', '←']
    ax.imshow(V_star.reshape(size, size), cmap='RdYlGn', alpha=0.3, aspect='equal')
    for r in range(size):
        for c in range(size):
            s = r * size + c
            best_a = np.argmax(pi_star[s])
            ax.text(c, r, action_symbols[best_a], ha='center', va='center',
                   fontsize=20, fontweight='bold')
    ax.set_title('π* (Optimal Policy)')

    plt.suptitle('Value Functions and Optimal Policy (4×4 Grid World)', fontsize=14)
    plt.tight_layout()
    plt.savefig("value_functions_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: value_functions_comparison.png")


def plot_contraction(distances, gamma_bound):
    """Plot contraction mapping convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, 'b-o', markersize=3, label='Actual distance ||V1 - V2||∞')
    ax.plot(gamma_bound, 'r--', alpha=0.7, label='γ^k bound')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max-Norm Distance')
    ax.set_title('Bellman Optimality Operator: Contraction Property')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bellman_contraction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bellman_contraction.png")


def plot_q_function_heatmap(Q, title="Q(s,a)"):
    """Visualize Q-function as heatmap."""
    fig, ax = plt.subplots(figsize=(6, 8))
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    im = ax.imshow(Q, cmap='RdYlGn', aspect='auto')
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_xticks(range(4))
    ax.set_xticklabels(action_names)
    ax.set_title(title)
    plt.colorbar(im, label='Q-value')
    plt.tight_layout()
    plt.savefig("q_function_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: q_function_heatmap.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Value function computation methods
    mdp, V_pi, Q_pi, A_pi = demo_value_functions()

    # 2. Bellman optimality
    V_star, Q_star, pi_star = demo_bellman_optimality()

    # 3. Contraction mapping
    print("\n" + "=" * 65)
    print("Contraction Mapping Demonstration")
    print("=" * 65)
    distances, gamma_bound = demonstrate_contraction(mdp, n_iterations=50)
    print(f"Initial distance: {distances[0]:.4f}")
    print(f"After 10 iters:   {distances[10]:.6f}")
    print(f"After 50 iters:   {distances[-1]:.10f}")

    # 4. Visualizations
    V_random = compute_V_pi_exact(
        mdp, np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions)
    plot_value_functions(mdp, V_random, V_star, pi_star)
    plot_contraction(distances, gamma_bound)
    plot_q_function_heatmap(Q_star, title="Q*(s,a) - Optimal Q-Function")

    print("\n✓ Value Functions and Bellman Equations demonstrations complete.")
