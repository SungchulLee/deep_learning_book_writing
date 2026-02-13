"""
Chapter 32.6: Temporal Difference Methods
==========================================
TD(0) prediction, SARSA, Q-Learning, Expected SARSA, Double Q-Learning.
"""

import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
import matplotlib.pyplot as plt


# =============================================================================
# 1. Cliff Walking Environment
# =============================================================================

class CliffWalkingEnv:
    """4x12 grid. Start=(3,0), Goal=(3,11), Cliff=(3,1..10)."""

    def __init__(self):
        self.rows, self.cols = 4, 12
        self.start, self.goal = (3, 0), (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}
        self.deltas = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.deltas[action]
        nr = np.clip(self.state[0]+dr, 0, self.rows-1)
        nc = np.clip(self.state[1]+dc, 0, self.cols-1)
        ns = (int(nr), int(nc))
        if ns in self.cliff:
            self.state = self.start
            return self.state, -100.0, False
        self.state = ns
        done = ns == self.goal
        return self.state, -1.0, done

    @property
    def n_actions(self):
        return 4


# =============================================================================
# 2. TD(0) Prediction
# =============================================================================

def td0_prediction(env, policy_fn, n_episodes=500, alpha=0.1, gamma=1.0):
    """
    TD(0): V(S) ← V(S) + α[R + γV(S') - V(S)]
    """
    V = defaultdict(float)
    td_errors = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_errors = []
        while not done:
            action = policy_fn(state)
            next_state, reward, done = env.step(action)
            td_error = reward + gamma * V[next_state] * (not done) - V[state]
            V[state] += alpha * td_error
            ep_errors.append(abs(td_error))
            state = next_state
        td_errors.append(np.mean(ep_errors) if ep_errors else 0)

    return dict(V), td_errors


# =============================================================================
# 3. SARSA (On-Policy TD Control)
# =============================================================================

def sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    SARSA: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
    """
    Q = defaultdict(float)
    rewards_per_episode = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state = env.reset()
        action = eps_greedy(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = eps_greedy(next_state)

            td_target = reward + gamma * Q[(next_state, next_action)] * (not done)
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state, action = next_state, next_action

        rewards_per_episode.append(total_reward)

    return dict(Q), rewards_per_episode


# =============================================================================
# 4. Q-Learning (Off-Policy TD Control)
# =============================================================================

def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Q-Learning: Q(S,A) ← Q(S,A) + α[R + γ max_a' Q(S',a') - Q(S,A)]
    """
    Q = defaultdict(float)
    rewards_per_episode = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = eps_greedy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            max_q_next = max(Q[(next_state, a)] for a in range(env.n_actions))
            td_target = reward + gamma * max_q_next * (not done)
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state

        rewards_per_episode.append(total_reward)

    return dict(Q), rewards_per_episode


# =============================================================================
# 5. Expected SARSA
# =============================================================================

def expected_sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Expected SARSA: Q(S,A) ← Q(S,A) + α[R + γ Σ_a' π(a'|S')Q(S',a') - Q(S,A)]
    """
    Q = defaultdict(float)
    rewards_per_episode = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    def expected_q(state):
        q_vals = np.array([Q[(state, a)] for a in range(env.n_actions)])
        best = np.argmax(q_vals)
        probs = np.ones(env.n_actions) * epsilon / env.n_actions
        probs[best] += 1 - epsilon
        return np.dot(probs, q_vals)

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = eps_greedy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            td_target = reward + gamma * expected_q(next_state) * (not done)
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state

        rewards_per_episode.append(total_reward)

    return dict(Q), rewards_per_episode


# =============================================================================
# 6. Double Q-Learning
# =============================================================================

def double_q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Double Q-Learning: reduces maximization bias.
    """
    Q1 = defaultdict(float)
    Q2 = defaultdict(float)
    rewards_per_episode = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q1[(state, a)] + Q2[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = eps_greedy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if np.random.random() < 0.5:
                best_a = max(range(env.n_actions), key=lambda a: Q1[(next_state, a)])
                td_target = reward + gamma * Q2[(next_state, best_a)] * (not done)
                Q1[(state, action)] += alpha * (td_target - Q1[(state, action)])
            else:
                best_a = max(range(env.n_actions), key=lambda a: Q2[(next_state, a)])
                td_target = reward + gamma * Q1[(next_state, best_a)] * (not done)
                Q2[(state, action)] += alpha * (td_target - Q2[(state, action)])

            state = next_state

        rewards_per_episode.append(total_reward)

    return dict(Q1), dict(Q2), rewards_per_episode


# =============================================================================
# 7. Demonstrations
# =============================================================================

def demo_td_methods():
    """Compare SARSA, Q-Learning, Expected SARSA on Cliff Walking."""
    print("=" * 65)
    print("TD Control Methods: Cliff Walking Comparison")
    print("=" * 65)

    n_runs = 20
    n_episodes = 500

    all_results = {}
    for name, algo in [("SARSA", sarsa), ("Q-Learning", q_learning),
                        ("Expected SARSA", expected_sarsa)]:
        run_rewards = []
        for _ in range(n_runs):
            env = CliffWalkingEnv()
            _, rewards = algo(env, n_episodes=n_episodes, alpha=0.5, epsilon=0.1)
            run_rewards.append(rewards)

        mean_rewards = np.mean(run_rewards, axis=0)
        all_results[name] = mean_rewards

        # Smooth for display
        window = 20
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        print(f"\n{name}:")
        print(f"  Final avg reward (last 50 ep): {np.mean(mean_rewards[-50:]):.1f}")

    return all_results


def demo_policy_comparison():
    """Show learned policies for SARSA vs Q-Learning."""
    env = CliffWalkingEnv()

    print("\n" + "=" * 65)
    print("Learned Policies Comparison")
    print("=" * 65)

    for name, algo in [("SARSA", sarsa), ("Q-Learning", q_learning)]:
        Q, _ = algo(env, n_episodes=5000, alpha=0.5, epsilon=0.1)

        action_symbols = ['↑', '→', '↓', '←']
        print(f"\n{name} Policy:")
        for r in range(4):
            row = ""
            for c in range(12):
                s = (r, c)
                if s in env.cliff:
                    row += " ☠ "
                elif s == env.goal:
                    row += " G "
                else:
                    q_vals = [Q.get((s, a), 0.0) for a in range(4)]
                    best = np.argmax(q_vals)
                    row += f" {action_symbols[best]} "
            print(f"  {row}")


def demo_double_q():
    """Demonstrate Double Q-Learning reduces maximization bias."""
    print("\n" + "=" * 65)
    print("Double Q-Learning: Reducing Maximization Bias")
    print("=" * 65)

    env = CliffWalkingEnv()
    _, rewards_q = q_learning(env, n_episodes=1000, alpha=0.5, epsilon=0.1)
    env2 = CliffWalkingEnv()
    _, _, rewards_dq = double_q_learning(env2, n_episodes=1000, alpha=0.5, epsilon=0.1)

    print(f"Q-Learning   last 100 avg: {np.mean(rewards_q[-100:]):.1f}")
    print(f"Double Q     last 100 avg: {np.mean(rewards_dq[-100:]):.1f}")

    return rewards_q, rewards_dq


# =============================================================================
# 8. Visualization
# =============================================================================

def plot_td_comparison(results):
    """Plot learning curves for TD methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 20
    colors = {'SARSA': '#3498db', 'Q-Learning': '#e74c3c', 'Expected SARSA': '#2ecc71'}

    for name, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=colors.get(name, 'gray'), linewidth=1.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward per Episode (smoothed)')
    ax.set_title('TD Control Methods: Cliff Walking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-200, 0)
    plt.tight_layout()
    plt.savefig("td_methods_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: td_methods_comparison.png")


def plot_q_values_heatmap(Q, env, title="Q-Values"):
    """Visualize max Q-values as heatmap."""
    grid = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            q_vals = [Q.get((s, a), 0.0) for a in range(env.n_actions)]
            grid[r, c] = max(q_vals)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto')
    ax.set_title(f'{title}: max_a Q(s,a)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            if s in env.cliff:
                ax.text(c, r, '☠', ha='center', va='center', fontsize=10)
            elif s == env.goal:
                ax.text(c, r, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.text(c, r, f'{grid[r,c]:.0f}', ha='center', va='center', fontsize=7)

    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    fname = f"td_{title.lower().replace(' ', '_')}_qvalues.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. TD methods comparison
    results = demo_td_methods()
    plot_td_comparison(results)

    # 2. Policy comparison
    demo_policy_comparison()

    # 3. Q-value visualization
    env = CliffWalkingEnv()
    Q_sarsa, _ = sarsa(env, n_episodes=5000, alpha=0.5, epsilon=0.1)
    plot_q_values_heatmap(Q_sarsa, CliffWalkingEnv(), "SARSA")

    env = CliffWalkingEnv()
    Q_ql, _ = q_learning(env, n_episodes=5000, alpha=0.5, epsilon=0.1)
    plot_q_values_heatmap(Q_ql, CliffWalkingEnv(), "Q-Learning")

    # 4. Double Q-Learning
    demo_double_q()

    print("\n✓ Temporal Difference methods demonstrations complete.")
