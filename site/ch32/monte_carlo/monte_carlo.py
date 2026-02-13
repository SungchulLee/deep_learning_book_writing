"""
Chapter 32.5: Monte Carlo Methods
===================================
MC prediction, MC control, off-policy MC, and importance sampling.
"""

import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
import matplotlib.pyplot as plt


# =============================================================================
# 1. Blackjack Environment
# =============================================================================

class BlackjackEnv:
    """Simplified Blackjack. State: (player_sum, dealer_showing, usable_ace)."""

    def _draw_card(self):
        return min(np.random.randint(1, 14), 10)

    def _hand_value(self, hand):
        total = sum(hand)
        usable_ace = 1 in hand and total + 10 <= 21
        if usable_ace:
            total += 10
        return total, usable_ace

    def reset(self):
        self.player = [self._draw_card(), self._draw_card()]
        self.dealer = [self._draw_card(), self._draw_card()]
        while self._hand_value(self.player)[0] < 12:
            self.player.append(self._draw_card())
        return self._get_state()

    def _get_state(self):
        psum, ace = self._hand_value(self.player)
        return (psum, self.dealer[0], ace)

    def step(self, action):
        if action == 1:  # Hit
            self.player.append(self._draw_card())
            if self._hand_value(self.player)[0] > 21:
                return self._get_state(), -1.0, True
            return self._get_state(), 0.0, False
        else:  # Stick - dealer plays
            while self._hand_value(self.dealer)[0] < 17:
                self.dealer.append(self._draw_card())
            p = self._hand_value(self.player)[0]
            d = self._hand_value(self.dealer)[0]
            if d > 21 or p > d:
                return self._get_state(), 1.0, True
            elif p == d:
                return self._get_state(), 0.0, True
            else:
                return self._get_state(), -1.0, True


def generate_episode(env, policy_fn):
    """Generate episode: list of (state, action, reward)."""
    state = env.reset()
    episode = []
    done = False
    while not done:
        action = policy_fn(state)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


# =============================================================================
# 2. MC Prediction (First-Visit and Every-Visit)
# =============================================================================

def mc_prediction_first_visit(env, policy_fn, n_episodes=50000, gamma=1.0):
    """First-visit MC prediction for V_pi."""
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy_fn)
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                N[state] += 1
                V[state] += (G - V[state]) / N[state]
    return dict(V), dict(N)


def mc_prediction_every_visit(env, policy_fn, n_episodes=50000, gamma=1.0):
    """Every-visit MC prediction for V_pi."""
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy_fn)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            N[state] += 1
            V[state] += (G - V[state]) / N[state]
    return dict(V), dict(N)


def mc_prediction_Q(env, policy_fn, n_episodes=50000, gamma=1.0):
    """First-visit MC prediction for Q_pi."""
    Q = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy_fn)
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                Q[sa] += (G - Q[sa]) / N[sa]
    return dict(Q), dict(N)


# =============================================================================
# 3. MC Control (On-Policy, epsilon-greedy)
# =============================================================================

def mc_control_on_policy(env, n_episodes=500000, gamma=1.0,
                          epsilon=0.1):
    """On-policy MC control with epsilon-greedy."""
    Q = defaultdict(float)
    N = defaultdict(int)
    n_actions = 2

    def epsilon_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        q_vals = [Q.get((state, a), 0.0) for a in range(n_actions)]
        return int(np.argmax(q_vals))

    for ep in range(n_episodes):
        episode = generate_episode(env, epsilon_greedy)
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                Q[sa] += (G - Q[sa]) / N[sa]

    # Extract greedy policy
    policy = {}
    states = set(s for (s, a) in Q.keys())
    for s in states:
        q_vals = [Q.get((s, a), 0.0) for a in range(n_actions)]
        policy[s] = int(np.argmax(q_vals))

    return dict(Q), policy


# =============================================================================
# 4. MC Control with Exploring Starts
# =============================================================================

def mc_control_exploring_starts(env, n_episodes=500000, gamma=1.0):
    """MC control with exploring starts (MC-ES)."""
    Q = defaultdict(float)
    N = defaultdict(int)
    policy = defaultdict(lambda: 0)
    n_actions = 2

    for _ in range(n_episodes):
        # Exploring start: random initial state-action
        state = env.reset()
        first_action = np.random.randint(n_actions)

        # Generate episode
        episode = []
        action = first_action
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if not done:
                action = policy[state]

        # Update Q and policy
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                N[sa] += 1
                Q[sa] += (G - Q[sa]) / N[sa]
                # Greedy improvement
                q_vals = [Q.get((state, a), 0.0) for a in range(n_actions)]
                policy[state] = int(np.argmax(q_vals))

    return dict(Q), dict(policy)


# =============================================================================
# 5. Off-Policy MC with Importance Sampling
# =============================================================================

def off_policy_mc_prediction(env, target_policy_fn, behavior_policy_fn,
                              behavior_probs_fn, target_probs_fn,
                              n_episodes=100000, gamma=1.0):
    """
    Off-policy MC prediction using weighted importance sampling.

    target_policy_fn: state -> action (deterministic target)
    behavior_policy_fn: state -> action (stochastic behavior)
    behavior_probs_fn: (state, action) -> probability under b
    target_probs_fn: (state, action) -> probability under pi
    """
    Q = defaultdict(float)
    C = defaultdict(float)

    for _ in range(n_episodes):
        episode = generate_episode(env, behavior_policy_fn)
        G = 0.0
        W = 1.0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa = (state, action)

            C[sa] += W
            Q[sa] += (W / C[sa]) * (G - Q[sa])

            # Update importance weight
            pi_prob = target_probs_fn(state, action)
            b_prob = behavior_probs_fn(state, action)

            if b_prob == 0:
                break
            W *= pi_prob / b_prob
            if W == 0:
                break

    return dict(Q)


def importance_sampling_demo():
    """Demonstrate ordinary vs weighted importance sampling."""
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("Importance Sampling: Ordinary vs Weighted")
    print("=" * 60)

    # Simple example: estimate E_pi[X] using samples from b
    # pi = N(2, 1), b = N(0, 2)
    n_samples = 10000
    b_samples = np.random.normal(0, 2, n_samples)
    true_mean = 2.0

    # IS ratios
    from scipy import stats
    pi_density = stats.norm.pdf(b_samples, loc=2, scale=1)
    b_density = stats.norm.pdf(b_samples, loc=0, scale=2)
    rho = pi_density / (b_density + 1e-10)

    # Ordinary IS
    ois_estimate = np.mean(rho * b_samples)
    # Weighted IS
    wis_estimate = np.sum(rho * b_samples) / np.sum(rho)

    # Running estimates
    ois_running = np.cumsum(rho * b_samples) / np.arange(1, n_samples + 1)
    wis_running = np.cumsum(rho * b_samples) / np.cumsum(rho)

    print(f"True E_pi[X] = {true_mean}")
    print(f"Ordinary IS estimate: {ois_estimate:.4f}")
    print(f"Weighted IS estimate: {wis_estimate:.4f}")
    print(f"OIS variance: {np.var(rho * b_samples):.4f}")
    print(f"Effective sample size: {np.sum(rho)**2 / np.sum(rho**2):.1f} / {n_samples}")

    return ois_running, wis_running, true_mean


# =============================================================================
# 6. Demonstrations
# =============================================================================

def demo_mc_prediction():
    """Demonstrate MC prediction on Blackjack."""
    env = BlackjackEnv()

    # Simple policy: stick on 20 or 21
    def simple_policy(state):
        return 0 if state[0] >= 20 else 1

    print("=" * 60)
    print("MC Prediction: Blackjack (stick on 20/21)")
    print("=" * 60)

    # First-visit
    V_first, N_first = mc_prediction_first_visit(env, simple_policy, n_episodes=100000)
    # Every-visit
    V_every, N_every = mc_prediction_every_visit(env, simple_policy, n_episodes=100000)

    # Show some values
    print(f"\n{'State':<25} {'V(first)':>10} {'V(every)':>10} {'N':>8}")
    print("-" * 55)
    sample_states = [(20, 10, False), (20, 1, False), (18, 10, False),
                     (15, 10, False), (12, 4, True)]
    for s in sample_states:
        vf = V_first.get(s, float('nan'))
        ve = V_every.get(s, float('nan'))
        n = N_first.get(s, 0)
        print(f"{str(s):<25} {vf:>10.4f} {ve:>10.4f} {n:>8}")

    return V_first


def demo_mc_control():
    """Demonstrate MC control on Blackjack."""
    env = BlackjackEnv()

    print("\n" + "=" * 60)
    print("MC Control: Learning Optimal Blackjack Policy")
    print("=" * 60)

    # On-policy control
    Q_on, policy_on = mc_control_on_policy(env, n_episodes=500000, epsilon=0.1)

    # ES control
    Q_es, policy_es = mc_control_exploring_starts(env, n_episodes=500000)

    # Show learned policy for no usable ace
    print("\nLearned Policy (no usable ace) — 0=stick, 1=hit:")
    print(f"{'Player Sum':>12}", end="")
    for d in range(1, 11):
        print(f" D={d:>2}", end="")
    print()

    for p_sum in range(21, 11, -1):
        print(f"  Sum={p_sum:>2}    ", end="")
        for dealer in range(1, 11):
            s = (p_sum, dealer, False)
            a = policy_es.get(s, '?')
            print(f"   {a} ", end="")
        print()

    return Q_es, policy_es


def demo_convergence():
    """Show MC convergence over episodes."""
    env = BlackjackEnv()
    target_state = (18, 10, False)

    def simple_policy(state):
        return 0 if state[0] >= 20 else 1

    episode_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    estimates = []

    for n in episode_counts:
        V, _ = mc_prediction_first_visit(env, simple_policy, n_episodes=n)
        estimates.append(V.get(target_state, 0.0))

    print("\n" + "=" * 60)
    print(f"MC Convergence for state {target_state}")
    print("=" * 60)
    for n, v in zip(episode_counts, estimates):
        print(f"  {n:>7} episodes: V = {v:.4f}")

    return episode_counts, estimates


# =============================================================================
# 7. Visualization
# =============================================================================

def plot_blackjack_value(V, title="V_π"):
    """Plot Blackjack value function."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, usable_ace in enumerate([False, True]):
        ax = axes[idx]
        grid = np.zeros((10, 10))
        for p_sum in range(12, 22):
            for dealer in range(1, 11):
                state = (p_sum, dealer, usable_ace)
                grid[21 - p_sum, dealer - 1] = V.get(state, 0.0)

        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto',
                       extent=[0.5, 10.5, 11.5, 21.5])
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(f'{title} (Usable Ace: {usable_ace})')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f'Blackjack: {title}', fontsize=14)
    plt.tight_layout()
    plt.savefig("mc_blackjack_value.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mc_blackjack_value.png")


def plot_is_comparison(ois_running, wis_running, true_mean):
    """Plot IS convergence comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(ois_running)
    x = np.arange(1, n + 1)

    ax.plot(x, ois_running, alpha=0.7, linewidth=0.8, label='Ordinary IS')
    ax.plot(x, wis_running, alpha=0.7, linewidth=0.8, label='Weighted IS')
    ax.axhline(y=true_mean, color='k', linestyle='--', label=f'True mean ({true_mean})')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Estimate')
    ax.set_title('Importance Sampling: Convergence Comparison')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mc_importance_sampling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mc_importance_sampling.png")


def plot_mc_convergence(episode_counts, estimates):
    """Plot MC convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_counts, estimates, 'bo-', markersize=6)
    if len(estimates) > 1:
        ax.axhline(y=estimates[-1], color='r', linestyle='--',
                   alpha=0.5, label=f'Final: {estimates[-1]:.4f}')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('V(s)')
    ax.set_title('MC Prediction Convergence')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mc_convergence.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. MC Prediction
    V_first = demo_mc_prediction()

    # 2. MC Control
    Q_es, policy_es = demo_mc_control()

    # 3. Convergence
    ep_counts, estimates = demo_convergence()

    # 4. Importance Sampling
    try:
        ois_run, wis_run, true_m = importance_sampling_demo()
        plot_is_comparison(ois_run, wis_run, true_m)
    except ImportError:
        print("scipy not available, skipping IS demo")

    # 5. Visualizations
    plot_blackjack_value(V_first, title="V_π (stick on 20/21)")
    plot_mc_convergence(ep_counts, estimates)

    print("\n✓ Monte Carlo Methods demonstrations complete.")
