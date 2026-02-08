"""
Chapter 32.4: Dynamic Programming
===================================
Implements policy evaluation, policy improvement, policy iteration,
value iteration, and financial portfolio optimization via DP.
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import time


# =============================================================================
# 1. MDP Definition
# =============================================================================

class FiniteMDP:
    def __init__(self, n_states, n_actions, P, R, gamma=0.99, terminal_states=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.R = R
        self.gamma = gamma
        self.terminal_states = terminal_states or set()


def create_gridworld(size=4, gamma=0.99):
    n_states = size * size
    n_actions = 4
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
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
# 2. Policy Evaluation
# =============================================================================

def policy_evaluation(mdp: FiniteMDP, policy: np.ndarray,
                       tol: float = 1e-8, max_iter: int = 10000) -> Tuple[np.ndarray, List[float]]:
    """
    Iterative Policy Evaluation.
    V_{k+1}(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
    
    Returns (V_pi, convergence_history).
    """
    V = np.zeros(mdp.n_states)
    history = []

    for iteration in range(max_iter):
        V_new = np.zeros(mdp.n_states)
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                V_new[s] += policy[s, a] * (
                    mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
                )
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tol:
            break

    return V, history


def policy_evaluation_exact(mdp: FiniteMDP, policy: np.ndarray) -> np.ndarray:
    """Exact policy evaluation via matrix inversion: V = (I - γP_π)^{-1} r_π"""
    P_pi = np.zeros((mdp.n_states, mdp.n_states))
    r_pi = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            P_pi[s] += policy[s, a] * mdp.P[s, a]
            r_pi[s] += policy[s, a] * mdp.R[s, a]
    return np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * P_pi, r_pi)


# =============================================================================
# 3. Policy Improvement
# =============================================================================

def policy_improvement(mdp: FiniteMDP, V: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Greedy policy improvement.
    π'(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    Returns (new_policy, is_stable).
    """
    new_policy = np.zeros((mdp.n_states, mdp.n_actions))
    is_stable = True

    for s in range(mdp.n_states):
        q_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            q_values[a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)

        best_action = np.argmax(q_values)
        new_policy[s, best_action] = 1.0

    return new_policy, is_stable


# =============================================================================
# 4. Policy Iteration
# =============================================================================

def policy_iteration(mdp: FiniteMDP, use_exact_eval: bool = True,
                      max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Policy Iteration: alternate evaluation and improvement until stable.
    
    Returns (optimal_policy, V_star, iteration_history).
    """
    # Initialize random deterministic policy
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        policy[s, np.random.randint(mdp.n_actions)] = 1.0

    history = []

    for iteration in range(max_iter):
        # Policy Evaluation
        if use_exact_eval:
            V = policy_evaluation_exact(mdp, policy)
            eval_iters = 1
        else:
            V, conv_hist = policy_evaluation(mdp, policy)
            eval_iters = len(conv_hist)

        # Policy Improvement
        new_policy, _ = policy_improvement(mdp, V)

        # Check stability
        policy_changed = not np.array_equal(new_policy, policy)
        history.append({
            "iteration": iteration + 1,
            "eval_iters": eval_iters,
            "mean_V": np.mean(V),
            "policy_changed": policy_changed,
        })

        if not policy_changed:
            return new_policy, V, history

        policy = new_policy

    return policy, V, history


# =============================================================================
# 5. Value Iteration
# =============================================================================

def value_iteration(mdp: FiniteMDP, tol: float = 1e-8,
                     max_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Value Iteration: V_{k+1}(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
    
    Returns (optimal_policy, V_star, convergence_history).
    """
    V = np.zeros(mdp.n_states)
    history = []

    for iteration in range(max_iter):
        V_new = np.zeros(mdp.n_states)
        for s in range(mdp.n_states):
            q_values = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                q_values[a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
            V_new[s] = np.max(q_values)

        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tol:
            break

    # Extract policy
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        q_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            q_values[a] = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
        policy[s, np.argmax(q_values)] = 1.0

    return policy, V, history


# =============================================================================
# 6. Modified Policy Iteration
# =============================================================================

def modified_policy_iteration(mdp: FiniteMDP, k_eval: int = 5,
                                max_iter: int = 1000,
                                tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Modified Policy Iteration: k sweeps of evaluation before improvement.
    k=1 → value iteration, k=∞ → policy iteration.
    """
    V = np.zeros(mdp.n_states)
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    history = []

    for iteration in range(max_iter):
        # Partial policy evaluation (k sweeps)
        for _ in range(k_eval):
            V_new = np.zeros(mdp.n_states)
            for s in range(mdp.n_states):
                for a in range(mdp.n_actions):
                    V_new[s] += policy[s, a] * (
                        mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
                    )
            V = V_new

        # Policy improvement
        new_policy, _ = policy_improvement(mdp, V)
        delta = np.max(np.abs(V - np.zeros_like(V)))  # Track V change

        # Also compute Bellman error
        V_opt_backup = np.zeros(mdp.n_states)
        for s in range(mdp.n_states):
            q_vals = [mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
                      for a in range(mdp.n_actions)]
            V_opt_backup[s] = max(q_vals)
        bellman_error = np.max(np.abs(V_opt_backup - V))
        history.append(bellman_error)

        policy = new_policy
        if bellman_error < tol:
            break

    return policy, V, history


# =============================================================================
# 7. Financial MDP: Simple Portfolio Optimization
# =============================================================================

def create_portfolio_mdp(n_market_states: int = 5, gamma: float = 0.95):
    """
    Portfolio MDP: Market regime × allocation.
    
    States: market_regime (bull/neutral/bear discretized into n levels)
    Actions: conservative(0), balanced(1), aggressive(2)
    """
    n_actions = 3
    n_states = n_market_states

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # Market regime transitions (semi-persistent)
    for s in range(n_states):
        for a in range(n_actions):
            for s_next in range(n_states):
                diff = abs(s_next - s)
                P[s, a, s_next] = np.exp(-diff)  # Prefer nearby states
            P[s, a] /= P[s, a].sum()

    # Rewards: depend on market state and allocation
    # Bull market (high s) → aggressive good, bear (low s) → conservative good
    for s in range(n_states):
        market_level = (s - n_states // 2) / (n_states // 2)  # [-1, 1]

        # Conservative: stable, small positive
        R[s, 0] = 0.02 + 0.01 * market_level

        # Balanced: moderate, some market sensitivity
        R[s, 1] = 0.03 + 0.03 * market_level

        # Aggressive: high exposure to market
        R[s, 2] = 0.01 + 0.08 * market_level

    return FiniteMDP(n_states, n_actions, P, R, gamma)


# =============================================================================
# 8. Demonstrations
# =============================================================================

def demo_policy_evaluation():
    """Demonstrate policy evaluation convergence."""
    mdp = create_gridworld(size=4, gamma=0.99)
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions

    print("=" * 65)
    print("Policy Evaluation: Iterative vs Exact")
    print("=" * 65)

    # Iterative
    V_iter, history = policy_evaluation(mdp, policy)
    print(f"\nIterative: converged in {len(history)} iterations")
    print(f"  V_pi = {V_iter.round(2)}")

    # Exact
    V_exact = policy_evaluation_exact(mdp, policy)
    print(f"\nExact (matrix inversion):")
    print(f"  V_pi = {V_exact.round(2)}")
    print(f"  Max difference: {np.max(np.abs(V_iter - V_exact)):.2e}")

    return history


def demo_policy_iteration():
    """Demonstrate policy iteration."""
    mdp = create_gridworld(size=4, gamma=0.99)

    print("\n" + "=" * 65)
    print("Policy Iteration")
    print("=" * 65)

    pi_star, V_star, history = policy_iteration(mdp, use_exact_eval=True)

    print(f"\nConverged in {len(history)} iterations:")
    for h in history:
        print(f"  Iter {h['iteration']}: mean(V)={h['mean_V']:.4f}, "
              f"changed={h['policy_changed']}")

    # Show optimal policy
    action_symbols = ['↑', '→', '↓', '←']
    print(f"\nOptimal Policy (4×4 grid):")
    for r in range(4):
        row = ""
        for c in range(4):
            s = r * 4 + c
            row += f" {action_symbols[np.argmax(pi_star[s])]} "
        print(f"  {row}")

    print(f"\nV* grid:")
    for r in range(4):
        row = " ".join(f"{V_star[r*4+c]:7.2f}" for c in range(4))
        print(f"  {row}")

    return V_star, pi_star


def demo_value_iteration():
    """Demonstrate value iteration."""
    mdp = create_gridworld(size=4, gamma=0.99)

    print("\n" + "=" * 65)
    print("Value Iteration")
    print("=" * 65)

    pi_star, V_star, history = value_iteration(mdp)

    print(f"Converged in {len(history)} iterations")
    print(f"Final Bellman error: {history[-1]:.2e}")

    return V_star, history


def demo_comparison():
    """Compare PI, VI, and modified PI."""
    sizes = [4, 5, 6, 7]
    results = {"PI": [], "VI": [], "MPI(k=3)": [], "MPI(k=10)": []}

    print("\n" + "=" * 65)
    print("Algorithm Comparison Across Grid Sizes")
    print("=" * 65)

    for size in sizes:
        mdp = create_gridworld(size=size, gamma=0.99)

        # Policy Iteration
        t0 = time.time()
        _, V_pi, hist_pi = policy_iteration(mdp)
        t_pi = time.time() - t0
        results["PI"].append({"size": size, "time": t_pi, "iters": len(hist_pi)})

        # Value Iteration
        t0 = time.time()
        _, V_vi, hist_vi = value_iteration(mdp)
        t_vi = time.time() - t0
        results["VI"].append({"size": size, "time": t_vi, "iters": len(hist_vi)})

        # Modified PI (k=3)
        t0 = time.time()
        _, V_mpi3, hist_mpi3 = modified_policy_iteration(mdp, k_eval=3)
        t_mpi3 = time.time() - t0
        results["MPI(k=3)"].append({"size": size, "time": t_mpi3, "iters": len(hist_mpi3)})

        # Modified PI (k=10)
        t0 = time.time()
        _, V_mpi10, hist_mpi10 = modified_policy_iteration(mdp, k_eval=10)
        t_mpi10 = time.time() - t0
        results["MPI(k=10)"].append({"size": size, "time": t_mpi10, "iters": len(hist_mpi10)})

        print(f"\n{size}×{size} grid ({size**2} states):")
        print(f"  PI:      {len(hist_pi):4d} iters, {t_pi:.4f}s, V(0)={V_pi[0]:.4f}")
        print(f"  VI:      {len(hist_vi):4d} iters, {t_vi:.4f}s, V(0)={V_vi[0]:.4f}")
        print(f"  MPI(3):  {len(hist_mpi3):4d} iters, {t_mpi3:.4f}s, V(0)={V_mpi3[0]:.4f}")
        print(f"  MPI(10): {len(hist_mpi10):4d} iters, {t_mpi10:.4f}s, V(0)={V_mpi10[0]:.4f}")

    return results


def demo_portfolio_dp():
    """DP for portfolio allocation."""
    mdp = create_portfolio_mdp(n_market_states=5, gamma=0.95)

    print("\n" + "=" * 65)
    print("Financial Application: Portfolio Allocation via DP")
    print("=" * 65)

    pi_star, V_star, _ = value_iteration(mdp)
    action_names = ['Conservative', 'Balanced', 'Aggressive']
    state_names = ['Strong Bear', 'Bear', 'Neutral', 'Bull', 'Strong Bull']

    print(f"\nOptimal Allocation by Market Regime:")
    for s in range(mdp.n_states):
        best_a = np.argmax(pi_star[s])
        print(f"  {state_names[s]:>12}: {action_names[best_a]:<14} (V*={V_star[s]:.4f})")


# =============================================================================
# 9. Visualization
# =============================================================================

def plot_convergence_comparison(eval_hist, vi_hist):
    """Plot convergence of policy evaluation vs value iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(eval_hist, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max |V_{k+1} - V_k|')
    ax.set_title('Policy Evaluation Convergence')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(vi_hist, 'r-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Bellman Error')
    ax.set_title('Value Iteration Convergence')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Dynamic Programming: Convergence Rates', fontsize=14)
    plt.tight_layout()
    plt.savefig("dp_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: dp_convergence.png")


def plot_optimal_policy(V_star, pi_star, size=4):
    """Visualize optimal policy and value function."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    action_symbols = ['↑', '→', '↓', '←']

    # Value function
    ax = axes[0]
    V_grid = V_star.reshape(size, size)
    im = ax.imshow(V_grid, cmap='RdYlGn', aspect='equal')
    for r in range(size):
        for c in range(size):
            ax.text(c, r, f'{V_grid[r, c]:.1f}', ha='center', va='center', fontsize=9)
    ax.set_title('V* (Optimal Value Function)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Policy
    ax = axes[1]
    ax.imshow(V_grid, cmap='RdYlGn', alpha=0.3, aspect='equal')
    for r in range(size):
        for c in range(size):
            s = r * size + c
            best_a = np.argmax(pi_star[s])
            ax.text(c, r, action_symbols[best_a], ha='center', va='center',
                   fontsize=18, fontweight='bold')
    ax.set_title('π* (Optimal Policy)')

    plt.suptitle('Dynamic Programming: Grid World Solution', fontsize=14)
    plt.tight_layout()
    plt.savefig("dp_optimal_policy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: dp_optimal_policy.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Policy evaluation
    eval_hist = demo_policy_evaluation()

    # 2. Policy iteration
    V_pi, pi_star = demo_policy_iteration()

    # 3. Value iteration
    V_vi, vi_hist = demo_value_iteration()

    # 4. Algorithm comparison
    demo_comparison()

    # 5. Financial application
    demo_portfolio_dp()

    # 6. Visualizations
    plot_convergence_comparison(eval_hist, vi_hist)
    plot_optimal_policy(V_pi, pi_star, size=4)

    print("\n✓ Dynamic Programming demonstrations complete.")
