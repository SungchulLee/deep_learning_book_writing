"""
33.6.1 Training Curves
========================

Utilities for logging, smoothing, and visualizing RL training curves.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import json
import os


class TrainingLogger:
    """Comprehensive training logger for RL experiments."""

    def __init__(self, name: str = "experiment"):
        self.name = name
        self.data: Dict[str, List[float]] = {
            'episode_rewards': [], 'episode_lengths': [],
            'losses': [], 'q_values': [], 'grad_norms': [],
            'epsilons': [], 'eval_rewards': [], 'eval_episodes': [],
        }

    def log_episode(self, reward, length, epsilon=None):
        self.data['episode_rewards'].append(reward)
        self.data['episode_lengths'].append(length)
        if epsilon is not None:
            self.data['epsilons'].append(epsilon)

    def log_step(self, loss=None, q_value=None, grad_norm=None):
        if loss is not None: self.data['losses'].append(loss)
        if q_value is not None: self.data['q_values'].append(q_value)
        if grad_norm is not None: self.data['grad_norms'].append(grad_norm)

    def log_eval(self, episode, reward):
        self.data['eval_episodes'].append(episode)
        self.data['eval_rewards'].append(reward)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.data.items()}, f)

    @classmethod
    def load(cls, path: str) -> 'TrainingLogger':
        logger = cls()
        with open(path) as f:
            logger.data = json.load(f)
        return logger


def moving_average(data: List[float], window: int = 100) -> np.ndarray:
    """Simple moving average."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def exponential_moving_average(data: List[float], beta: float = 0.99) -> np.ndarray:
    """Exponential moving average."""
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = beta * ema[i-1] + (1 - beta) * data[i]
    return ema


def percentile_bands(data: List[float], window: int = 100) -> Dict[str, np.ndarray]:
    """Compute rolling percentile bands."""
    n = len(data)
    medians, lows, highs = [], [], []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = data[start:i+1]
        medians.append(np.median(chunk))
        lows.append(np.percentile(chunk, 25))
        highs.append(np.percentile(chunk, 75))
    return {'median': np.array(medians), 'p25': np.array(lows), 'p75': np.array(highs)}


def plot_training_curves(logger: TrainingLogger, save_path: str = 'training_curves.png'):
    """Generate comprehensive training curve plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves: {logger.name}', fontsize=14)

    # 1. Episode rewards
    ax = axes[0, 0]
    rewards = logger.data['episode_rewards']
    if rewards:
        ax.plot(rewards, alpha=0.2, color='blue')
        if len(rewards) >= 50:
            ma = moving_average(rewards, 50)
            ax.plot(range(49, len(rewards)), ma, color='red', label='MA(50)')
        ax.set_title('Episode Rewards'); ax.set_xlabel('Episode'); ax.set_ylabel('Return')
        ax.legend()

    # 2. Loss
    ax = axes[0, 1]
    losses = logger.data['losses']
    if losses:
        ax.plot(losses, alpha=0.2, color='orange')
        if len(losses) >= 100:
            ma = moving_average(losses, 100)
            ax.plot(range(99, len(losses)), ma, color='red')
        ax.set_title('Training Loss'); ax.set_xlabel('Step'); ax.set_ylabel('Loss')
        ax.set_yscale('log')

    # 3. Q-values
    ax = axes[0, 2]
    qvals = logger.data['q_values']
    if qvals:
        ax.plot(qvals, alpha=0.3, color='green')
        if len(qvals) >= 100:
            ma = moving_average(qvals, 100)
            ax.plot(range(99, len(qvals)), ma, color='darkgreen')
        ax.set_title('Mean Q-Value'); ax.set_xlabel('Step')

    # 4. Episode lengths
    ax = axes[1, 0]
    lengths = logger.data['episode_lengths']
    if lengths:
        ax.plot(lengths, alpha=0.2, color='purple')
        if len(lengths) >= 50:
            ma = moving_average(lengths, 50)
            ax.plot(range(49, len(lengths)), ma, color='darkviolet')
        ax.set_title('Episode Length'); ax.set_xlabel('Episode')

    # 5. Gradient norms
    ax = axes[1, 1]
    gnorms = logger.data['grad_norms']
    if gnorms:
        ax.plot(gnorms, alpha=0.2, color='brown')
        if len(gnorms) >= 100:
            ma = moving_average(gnorms, 100)
            ax.plot(range(99, len(gnorms)), ma, color='darkred')
        ax.set_title('Gradient Norm'); ax.set_xlabel('Step')

    # 6. Eval rewards
    ax = axes[1, 2]
    eval_ep = logger.data['eval_episodes']
    eval_r = logger.data['eval_rewards']
    if eval_ep:
        ax.plot(eval_ep, eval_r, 'o-', color='teal')
        ax.set_title('Evaluation Return'); ax.set_xlabel('Episode')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_multi_seed(all_rewards: List[List[float]], labels: List[str] = None,
                    window: int = 50, save_path: str = 'multi_seed.png'):
    """Plot multiple runs with mean and std bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, rewards in enumerate(all_rewards):
        label = labels[i] if labels else f'Run {i+1}'
        color = colors[i % len(colors)]
        smoothed = moving_average(rewards, window)
        x = range(window - 1, len(rewards))
        ax.plot(x, smoothed, color=color, label=label)
        ax.fill_between(x, smoothed - np.std(rewards[-len(smoothed):]),
                        smoothed + np.std(rewards[-len(smoothed):]),
                        alpha=0.1, color=color)

    ax.set_xlabel('Episode'); ax.set_ylabel('Return')
    ax.set_title('Multi-Seed Comparison'); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-seed plot saved to {save_path}")


def demo_training_curves():
    print("=" * 60)
    print("Training Curves Demo")
    print("=" * 60)

    # Generate synthetic training data
    np.random.seed(42)
    n_episodes = 500
    logger = TrainingLogger("DQN-CartPole")

    for ep in range(n_episodes):
        # Simulate improving rewards
        base = min(200, 20 + ep * 0.4)
        reward = max(10, base + np.random.randn() * 30)
        length = int(reward)
        epsilon = max(0.01, 1.0 - ep / 200)

        logger.log_episode(reward, length, epsilon)

        for _ in range(length):
            loss = max(0.001, 1.0 / (1 + ep * 0.01) + np.random.randn() * 0.1)
            q_val = base * 0.5 + np.random.randn() * 2
            grad_norm = np.abs(np.random.randn()) * 2
            logger.log_step(loss, q_val, grad_norm)

        if ep % 50 == 0:
            logger.log_eval(ep, base + np.random.randn() * 10)

    # Plot
    plot_training_curves(logger, 'training_curves.png')

    # Smoothing comparison
    print("\n--- Smoothing Methods ---")
    rewards = logger.data['episode_rewards']
    ma50 = moving_average(rewards, 50)
    ema99 = exponential_moving_average(rewards, 0.99)
    bands = percentile_bands(rewards, 50)
    print(f"  Raw: final = {rewards[-1]:.1f}")
    print(f"  MA(50): final = {ma50[-1]:.1f}")
    print(f"  EMA(0.99): final = {ema99[-1]:.1f}")
    print(f"  Median: final = {bands['median'][-1]:.1f}")

    # Multi-seed
    all_rewards = []
    for seed in range(3):
        np.random.seed(seed)
        rews = [max(10, min(200, 20 + ep*0.4) + np.random.randn()*30) for ep in range(300)]
        all_rewards.append(rews)
    plot_multi_seed(all_rewards, ['Seed 0', 'Seed 1', 'Seed 2'])

    print("\nTraining curves demo complete!")


if __name__ == "__main__":
    demo_training_curves()
