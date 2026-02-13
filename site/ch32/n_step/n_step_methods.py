"""
Chapter 32.7: N-Step Methods and Eligibility Traces
=====================================================
N-step TD, n-step SARSA, TD(lambda), eligibility traces, SARSA(lambda).
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


# =============================================================================
# 1. Random Walk Environment (for prediction)
# =============================================================================

class RandomWalkEnv:
    """19-state random walk (Sutton & Barto Example 7.1). States 1-19, terminals 0,20."""

    def __init__(self, n_states=19):
        self.n_states = n_states
        self.start = n_states // 2 + 1

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action=None):
        step = 1 if np.random.random() < 0.5 else -1
        self.state += step
        if self.state >= self.n_states + 1:
            return self.state, 1.0, True
        elif self.state <= 0:
            return self.state, -1.0, True
        return self.state, 0.0, False


# =============================================================================
# 2. Cliff Walking (for control)
# =============================================================================

class CliffWalking:
    def __init__(self):
        self.rows, self.cols = 4, 12
        self.start, self.goal = (3,0), (3,11)
        self.cliff = {(3,c) for c in range(1,11)}
        self.deltas = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.deltas[action]
        nr = int(np.clip(self.state[0]+dr, 0, self.rows-1))
        nc = int(np.clip(self.state[1]+dc, 0, self.cols-1))
        ns = (nr, nc)
        if ns in self.cliff:
            self.state = self.start
            return self.state, -100.0, False
        self.state = ns
        return self.state, -1.0, ns == self.goal

    @property
    def n_actions(self):
        return 4


# =============================================================================
# 3. N-Step TD Prediction
# =============================================================================

def n_step_td_prediction(env, n, n_episodes=100, alpha=0.1, gamma=1.0):
    """N-step TD prediction for the random walk."""
    V = np.zeros(env.n_states + 2)  # Include terminals
    true_values = np.linspace(-1, 1, env.n_states + 2)

    errors = []
    for _ in range(n_episodes):
        state = env.reset()
        states = [state]
        rewards = [0.0]
        T = float('inf')
        t = 0

        while True:
            if t < T:
                next_state, reward, done = env.step()
                states.append(next_state)
                rewards.append(reward)
                if done:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**(i-tau-1) * rewards[i]
                        for i in range(tau+1, min(tau+n, int(T))+1))
                if tau + n < T:
                    G += gamma**n * V[states[tau+n]]
                s_tau = states[tau]
                if 0 < s_tau <= env.n_states:
                    V[s_tau] += alpha * (G - V[s_tau])

            if tau == T - 1:
                break
            t += 1

        rms = np.sqrt(np.mean((V[1:-1] - true_values[1:-1])**2))
        errors.append(rms)

    return V, errors


# =============================================================================
# 4. N-Step SARSA
# =============================================================================

def n_step_sarsa(env, n, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """N-step SARSA for control."""
    Q = defaultdict(float)
    rewards_per_ep = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state = env.reset()
        action = eps_greedy(state)
        states = [state]
        actions = [action]
        rewards_list = [0.0]
        T = float('inf')
        t = 0
        total_reward = 0

        while True:
            if t < T:
                next_state, reward, done = env.step(action)
                total_reward += reward
                states.append(next_state)
                rewards_list.append(reward)
                if done:
                    T = t + 1
                else:
                    next_action = eps_greedy(next_state)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**(i-tau-1) * rewards_list[i]
                        for i in range(tau+1, min(tau+n, int(T))+1))
                if tau + n < T:
                    G += gamma**n * Q[(states[tau+n], actions[tau+n])]
                sa = (states[tau], actions[tau])
                Q[sa] += alpha * (G - Q[sa])

            if tau == T - 1:
                break
            t += 1
            if t < T:
                action = actions[t]

        rewards_per_ep.append(total_reward)

    return dict(Q), rewards_per_ep


# =============================================================================
# 5. TD(lambda) with Eligibility Traces
# =============================================================================

def td_lambda_prediction(env, lam, n_episodes=100, alpha=0.1, gamma=1.0):
    """TD(λ) prediction with accumulating traces."""
    V = np.zeros(env.n_states + 2)
    true_values = np.linspace(-1, 1, env.n_states + 2)
    errors = []

    for _ in range(n_episodes):
        state = env.reset()
        e = np.zeros(env.n_states + 2)  # Eligibility traces
        done = False

        while not done:
            next_state, reward, done = env.step()
            delta = reward + gamma * V[next_state] * (not done) - V[state]

            # Update trace
            if 0 < state <= env.n_states:
                e[state] += 1  # Accumulating trace

            # Update all values
            V += alpha * delta * e
            e *= gamma * lam  # Decay traces

            state = next_state

        rms = np.sqrt(np.mean((V[1:-1] - true_values[1:-1])**2))
        errors.append(rms)

    return V, errors


# =============================================================================
# 6. SARSA(lambda) with Eligibility Traces
# =============================================================================

def sarsa_lambda(env, lam, n_episodes=500, alpha=0.5, gamma=1.0,
                  epsilon=0.1, trace_type='accumulating'):
    """SARSA(λ) with eligibility traces for control."""
    Q = defaultdict(float)
    rewards_per_ep = []

    def eps_greedy(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        q_vals = [Q[(state, a)] for a in range(env.n_actions)]
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state = env.reset()
        action = eps_greedy(state)
        e = defaultdict(float)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = eps_greedy(next_state) if not done else 0

            delta = reward + gamma * Q[(next_state, next_action)] * (not done) \
                    - Q[(state, action)]

            # Update trace
            if trace_type == 'replacing':
                e[(state, action)] = 1.0
            else:
                e[(state, action)] += 1.0

            # Update Q for all traced state-action pairs
            for sa in list(e.keys()):
                Q[sa] += alpha * delta * e[sa]
                e[sa] *= gamma * lam
                if e[sa] < 1e-10:
                    del e[sa]

            state, action = next_state, next_action

        rewards_per_ep.append(total_reward)

    return dict(Q), rewards_per_ep


# =============================================================================
# 7. Demonstrations
# =============================================================================

def demo_n_step_prediction():
    """Compare n-step TD for different n values."""
    print("=" * 65)
    print("N-Step TD Prediction: Random Walk")
    print("=" * 65)

    n_values = [1, 2, 4, 8, 16]
    results = {}

    for n in n_values:
        errors_all = []
        for _ in range(10):
            env = RandomWalkEnv(19)
            _, errors = n_step_td_prediction(env, n, n_episodes=100, alpha=0.1)
            errors_all.append(errors)
        mean_errors = np.mean(errors_all, axis=0)
        results[n] = mean_errors
        print(f"  n={n:>2}: Final RMS = {mean_errors[-1]:.4f}")

    return results


def demo_td_lambda_vs_nstep():
    """Compare TD(λ) with n-step TD."""
    print("\n" + "=" * 65)
    print("TD(λ) vs N-Step TD Prediction")
    print("=" * 65)

    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]
    results = {}

    for lam in lambdas:
        errors_all = []
        for _ in range(10):
            env = RandomWalkEnv(19)
            _, errors = td_lambda_prediction(env, lam, n_episodes=100, alpha=0.1)
            errors_all.append(errors)
        mean_errors = np.mean(errors_all, axis=0)
        results[lam] = mean_errors
        print(f"  λ={lam:.2f}: Final RMS = {mean_errors[-1]:.4f}")

    return results


def demo_sarsa_lambda():
    """Compare SARSA(λ) for different λ on Cliff Walking."""
    print("\n" + "=" * 65)
    print("SARSA(λ): Cliff Walking")
    print("=" * 65)

    lambdas = [0.0, 0.5, 0.8, 0.95]
    results = {}

    for lam in lambdas:
        runs = []
        for _ in range(5):
            env = CliffWalking()
            _, rewards = sarsa_lambda(env, lam, n_episodes=500, alpha=0.5, epsilon=0.1)
            runs.append(rewards)
        mean_r = np.mean(runs, axis=0)
        results[lam] = mean_r
        print(f"  λ={lam:.2f}: Avg reward (last 50) = {np.mean(mean_r[-50:]):.1f}")

    return results


# =============================================================================
# 8. Visualization
# =============================================================================

def plot_n_step_comparison(results):
    """Plot n-step TD prediction errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for n, errors in results.items():
        ax.plot(errors, label=f'n={n}', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('RMS Error')
    ax.set_title('N-Step TD Prediction: Effect of n')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("n_step_td_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: n_step_td_comparison.png")


def plot_td_lambda_comparison(results):
    """Plot TD(λ) for different λ values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for lam, errors in results.items():
        ax.plot(errors, label=f'λ={lam}', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('RMS Error')
    ax.set_title('TD(λ) Prediction: Effect of λ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("td_lambda_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: td_lambda_comparison.png")


def plot_sarsa_lambda(results):
    """Plot SARSA(λ) learning curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 20
    for lam, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'λ={lam}', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('SARSA(λ): Cliff Walking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-200, 0)
    plt.tight_layout()
    plt.savefig("sarsa_lambda_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sarsa_lambda_comparison.png")


def plot_eligibility_trace_decay():
    """Visualize eligibility trace behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trace decay for different lambda
    ax = axes[0]
    steps = np.arange(30)
    gamma = 0.99
    for lam in [0.0, 0.5, 0.8, 0.9, 0.95]:
        trace = (gamma * lam) ** steps
        ax.plot(steps, trace, label=f'λ={lam}', linewidth=2)
    ax.set_xlabel('Steps Since Visit')
    ax.set_ylabel('Trace Value e(s)')
    ax.set_title('Eligibility Trace Decay (γ=0.99)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accumulating vs replacing traces
    ax = axes[1]
    n_steps = 50
    visit_times = [0, 5, 10, 25]
    gamma_lam = 0.9 * 0.8

    # Accumulating
    e_acc = np.zeros(n_steps)
    for t in range(n_steps):
        e_acc[t] = gamma_lam * (e_acc[t-1] if t > 0 else 0)
        if t in visit_times:
            e_acc[t] += 1.0

    # Replacing
    e_rep = np.zeros(n_steps)
    for t in range(n_steps):
        e_rep[t] = gamma_lam * (e_rep[t-1] if t > 0 else 0)
        if t in visit_times:
            e_rep[t] = 1.0

    ax.plot(e_acc, 'b-', label='Accumulating', linewidth=2)
    ax.plot(e_rep, 'r--', label='Replacing', linewidth=2)
    for vt in visit_times:
        ax.axvline(x=vt, color='gray', alpha=0.3, linestyle=':')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Trace Value')
    ax.set_title('Accumulating vs Replacing Traces')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eligibility_traces_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: eligibility_traces_demo.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. N-step TD prediction
    nstep_results = demo_n_step_prediction()
    plot_n_step_comparison(nstep_results)

    # 2. TD(λ) prediction
    tdl_results = demo_td_lambda_vs_nstep()
    plot_td_lambda_comparison(tdl_results)

    # 3. SARSA(λ) control
    sarsa_results = demo_sarsa_lambda()
    plot_sarsa_lambda(sarsa_results)

    # 4. Eligibility trace visualization
    plot_eligibility_trace_decay()

    print("\n✓ N-Step Methods and Eligibility Traces demonstrations complete.")
