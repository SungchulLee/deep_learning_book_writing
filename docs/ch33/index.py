"""
Chapter 33: Value-Based Deep Reinforcement Learning
====================================================

Overview and utility functions used throughout the chapter.
This module provides common components for value-based deep RL experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------------------
# Common data structures
# ---------------------------------------------------------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Basic replay buffer (used in many sections)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(np.array(batch.action))
        rewards = torch.FloatTensor(np.array(batch.reward))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(np.array(batch.done))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Simple Q-Network (baseline architecture)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Basic feed-forward Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    """Polyak averaging for target network update."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    """Copy all parameters from source to target."""
    target.load_state_dict(source.state_dict())


def epsilon_schedule(step: int, eps_start: float = 1.0, eps_end: float = 0.01,
                     eps_decay: int = 10000) -> float:
    """Linear epsilon decay schedule."""
    return max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_agent(env, q_network: nn.Module, n_episodes: int = 10,
                   device: str = 'cpu') -> Dict[str, float]:
    """Evaluate a Q-network agent over multiple episodes."""
    returns = []
    lengths = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_network(state_t).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        returns.append(total_reward)
        lengths.append(steps)
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'min_return': np.min(returns),
        'max_return': np.max(returns)
    }


def plot_training_curves(rewards: List[float], window: int = 100,
                         title: str = "Training Curve"):
    """Plot training rewards with a rolling average."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color='blue', label='Episode reward')
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(rewards)), rolling, color='red',
                label=f'{window}-episode average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curve saved to training_curve.png")


# ---------------------------------------------------------------------------
# Demo: Quick CartPole check
# ---------------------------------------------------------------------------

def demo_cartpole_random():
    """Demo: random agent on CartPole to verify environment setup."""
    env = gym.make('CartPole-v1')
    state, _ = env.reset()
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state: {state}")

    total_reward = 0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Random agent: {steps} steps, total reward = {total_reward}")
    env.close()
    return total_reward


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 33: Value-Based Deep RL — Setup Verification")
    print("=" * 60)

    # Verify dependencies
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"NumPy version: {np.__version__}")

    # Test basic components
    print("\n--- Replay Buffer Test ---")
    buf = ReplayBuffer(1000)
    for i in range(100):
        buf.push(np.random.randn(4), np.random.randint(2),
                 np.random.randn(), np.random.randn(4), False)
    s, a, r, ns, d = buf.sample(32)
    print(f"Sampled batch shapes: states={s.shape}, actions={a.shape}")

    print("\n--- Q-Network Test ---")
    qnet = QNetwork(state_dim=4, action_dim=2)
    test_state = torch.randn(1, 4)
    q_values = qnet(test_state)
    print(f"Q-values for test state: {q_values.detach().numpy()}")

    print("\n--- CartPole Random Agent ---")
    demo_cartpole_random()

    print("\n--- Epsilon Schedule ---")
    for step in [0, 2500, 5000, 7500, 10000]:
        eps = epsilon_schedule(step)
        print(f"  Step {step:>6d}: epsilon = {eps:.4f}")

    print("\nSetup verification complete!")
