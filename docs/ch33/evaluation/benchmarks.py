"""
33.6.2 Benchmarks
==================

Benchmark evaluation utilities and standardized scoring.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from typing import Dict, List, Callable


# ---------------------------------------------------------------------------
# Human-Normalized Scoring
# ---------------------------------------------------------------------------

ATARI_BASELINES = {
    'Breakout': {'random': 1.7, 'human': 30.5},
    'Pong': {'random': -20.7, 'human': 14.6},
    'SpaceInvaders': {'random': 148.0, 'human': 1668.7},
    'Seaquest': {'random': 68.4, 'human': 42054.7},
    'Qbert': {'random': 163.9, 'human': 13455.0},
}


def human_normalized_score(agent_score: float, game: str) -> float:
    """Compute human-normalized score for Atari games."""
    if game not in ATARI_BASELINES:
        raise ValueError(f"Unknown game: {game}")
    rand = ATARI_BASELINES[game]['random']
    human = ATARI_BASELINES[game]['human']
    return (agent_score - rand) / (human - rand) * 100


def d4rl_normalized_score(agent_score: float, random_score: float,
                          expert_score: float) -> float:
    """D4RL normalized score: 0 = random, 100 = expert."""
    return (agent_score - random_score) / (expert_score - random_score + 1e-8) * 100


# ---------------------------------------------------------------------------
# Benchmark Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(env_name: str, select_action_fn: Callable,
                   n_episodes: int = 100, max_steps: int = 10000,
                   seed: int = 42) -> Dict[str, float]:
    """Comprehensive agent evaluation."""
    env = gym.make(env_name)
    returns, lengths = [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_r, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            action = select_action_fn(state)
            state, r, term, trunc, _ = env.step(action)
            total_r += r; steps += 1
            done = term or trunc
        returns.append(total_r)
        lengths.append(steps)

    env.close()
    returns = np.array(returns)
    lengths = np.array(lengths)
    return {
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'median_return': np.median(returns),
        'min_return': returns.min(),
        'max_return': returns.max(),
        'iqr_return': np.percentile(returns, 75) - np.percentile(returns, 25),
        'mean_length': lengths.mean(),
        'n_episodes': n_episodes,
    }


def benchmark_suite(env_name: str, agents: Dict[str, Callable],
                    n_episodes: int = 50) -> Dict[str, Dict]:
    """Compare multiple agents on the same environment."""
    results = {}
    for name, action_fn in agents.items():
        result = evaluate_agent(env_name, action_fn, n_episodes)
        results[name] = result
    return results


def print_benchmark_table(results: Dict[str, Dict]):
    """Pretty-print benchmark comparison."""
    header = f"{'Agent':<25s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>6s} {'Max':>6s}"
    print(header)
    print("-" * len(header))
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_return']):
        print(f"{name:<25s} {r['mean_return']:>8.1f} {r['std_return']:>8.1f} "
              f"{r['median_return']:>8.1f} {r['min_return']:>6.0f} {r['max_return']:>6.0f}")


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

CLASSIC_CONTROL = {
    'CartPole-v1': {'solved': 475, 'max_steps': 500},
    'MountainCar-v0': {'solved': -110, 'max_steps': 200},
    'LunarLander-v2': {'solved': 200, 'max_steps': 1000},
    'Acrobot-v1': {'solved': -100, 'max_steps': 500},
}


def check_solved(env_name: str, mean_return: float) -> bool:
    """Check if environment is 'solved' by standard criteria."""
    if env_name in CLASSIC_CONTROL:
        return mean_return >= CLASSIC_CONTROL[env_name]['solved']
    return False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_benchmarks():
    print("=" * 60)
    print("Benchmarks Demo")
    print("=" * 60)

    # Human-normalized scores
    print("\n--- Human-Normalized Scoring ---")
    for game, scores in ATARI_BASELINES.items():
        for agent_score in [scores['random'], scores['human'], scores['human'] * 1.5]:
            hn = human_normalized_score(agent_score, game)
            print(f"  {game}: score={agent_score:.0f} → {hn:.1f}% human-normalized")

    # Benchmark on CartPole
    print("\n--- CartPole Benchmark ---")
    env = gym.make('CartPole-v1')
    sd, ad = env.observation_space.shape[0], env.action_space.n
    env.close()

    agents = {
        'Random': lambda s: np.random.randint(ad),
        'Always Left': lambda s: 0,
        'Always Right': lambda s: 1,
    }

    results = benchmark_suite('CartPole-v1', agents, n_episodes=50)
    print_benchmark_table(results)

    # Check solved
    for name, r in results.items():
        solved = check_solved('CartPole-v1', r['mean_return'])
        print(f"  {name}: {'SOLVED' if solved else 'not solved'} "
              f"(need ≥ {CLASSIC_CONTROL['CartPole-v1']['solved']})")

    # D4RL scoring
    print("\n--- D4RL Normalized Scoring ---")
    for agent_s in [10, 30, 50, 80, 100]:
        norm = d4rl_normalized_score(agent_s, random_score=10, expert_score=100)
        print(f"  Agent score={agent_s}: D4RL normalized = {norm:.1f}")

    print("\nBenchmarks demo complete!")


if __name__ == "__main__":
    demo_benchmarks()
