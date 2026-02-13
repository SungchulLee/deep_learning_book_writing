"""
33.1.3 Target Networks
======================

Demonstration of hard vs soft target network updates and their
impact on training stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, Dict, List
import copy
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = Transition(*zip(*random.sample(self.buffer, batch_size)))
        return (
            torch.FloatTensor(np.array(batch.state)),
            torch.LongTensor(np.array(batch.action)),
            torch.FloatTensor(np.array(batch.reward)),
            torch.FloatTensor(np.array(batch.next_state)),
            torch.FloatTensor(np.array(batch.done, dtype=np.float32)),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Target Network Update Functions
# ---------------------------------------------------------------------------

def hard_update(target_net: nn.Module, online_net: nn.Module):
    """Hard update: copy all parameters from online to target."""
    target_net.load_state_dict(online_net.state_dict())


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float = 0.005):
    """Soft (Polyak) update: θ⁻ ← τθ + (1-τ)θ⁻"""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def compute_parameter_distance(net1: nn.Module, net2: nn.Module) -> float:
    """L2 distance between two networks' parameters."""
    dist = 0.0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        dist += (p1 - p2).pow(2).sum().item()
    return np.sqrt(dist)


# ---------------------------------------------------------------------------
# DQN Agent with configurable target update strategy
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN Agent supporting both hard and soft target updates."""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 update_mode: str = 'hard',  # 'hard', 'soft', or 'none'
                 hard_update_freq: int = 100,
                 tau: float = 0.005,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 eps_start: float = 1.0, eps_end: float = 0.01,
                 eps_decay: int = 5000):
        self.gamma = gamma
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.update_mode = update_mode
        self.hard_update_freq = hard_update_freq
        self.tau = tau

        # Networks
        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        hard_update(self.target_net, self.online_net)
        self.target_net.eval()  # Target net never trains

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # Exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step_count = 0
        self.update_count = 0

        # Tracking
        self.losses: List[float] = []
        self.target_distances: List[float] = []

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            self.step_count += 1
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.update_mode == 'none':
                # No target network — uses online net for targets
                next_q = self.online_net(next_states).max(dim=1)[0]
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        # Target network update
        if self.update_mode == 'hard' and self.update_count % self.hard_update_freq == 0:
            hard_update(self.target_net, self.online_net)
        elif self.update_mode == 'soft':
            soft_update(self.target_net, self.online_net, self.tau)

        # Track parameter distance
        dist = compute_parameter_distance(self.online_net, self.target_net)
        self.target_distances.append(dist)

        return loss_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_agent(agent: DQNAgent, env_name: str = 'CartPole-v1',
                n_episodes: int = 300, min_buffer: int = 500) -> List[float]:
    """Train a DQN agent and return episode rewards."""
    env = gym.make(env_name)
    rewards_history = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            if len(agent.buffer) >= min_buffer:
                agent.update()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

    env.close()
    return rewards_history


# ---------------------------------------------------------------------------
# Demo: Compare target update strategies
# ---------------------------------------------------------------------------

def demo_target_networks():
    """Compare training with different target network strategies."""
    print("=" * 60)
    print("Target Networks Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    configs = {
        'No Target Net': {'update_mode': 'none'},
        'Hard Update (C=100)': {'update_mode': 'hard', 'hard_update_freq': 100},
        'Hard Update (C=500)': {'update_mode': 'hard', 'hard_update_freq': 500},
        'Soft Update (τ=0.005)': {'update_mode': 'soft', 'tau': 0.005},
        'Soft Update (τ=0.05)': {'update_mode': 'soft', 'tau': 0.05},
    }

    n_episodes = 200
    results = {}

    for name, cfg in configs.items():
        print(f"\nTraining: {name}")
        # Fix seed for fair comparison
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        agent = DQNAgent(state_dim, action_dim, lr=1e-3, **cfg)
        rewards = train_agent(agent, n_episodes=n_episodes)
        results[name] = {
            'rewards': rewards,
            'losses': agent.losses,
            'target_distances': agent.target_distances,
        }

        # Summary statistics
        last_50 = rewards[-50:]
        print(f"  Last 50 episodes: mean={np.mean(last_50):.1f}, "
              f"std={np.std(last_50):.1f}, max={np.max(last_50):.0f}")
        if agent.losses:
            last_losses = agent.losses[-100:]
            print(f"  Recent loss: mean={np.mean(last_losses):.4f}")
        if agent.target_distances:
            print(f"  Final online-target distance: {agent.target_distances[-1]:.4f}")

    # --- Comparison summary ---
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Strategy':<25s} {'Mean(last50)':>12s} {'Std':>8s} {'Max':>6s}")
    print("-" * 55)
    for name, data in results.items():
        last_50 = data['rewards'][-50:]
        print(f"{name:<25s} {np.mean(last_50):>12.1f} {np.std(last_50):>8.1f} "
              f"{np.max(last_50):>6.0f}")

    # --- Demonstrate parameter tracking ---
    print("\n--- Parameter Distance Evolution ---")
    for name, data in results.items():
        dists = data['target_distances']
        if dists:
            print(f"  {name}: start={dists[0]:.4f}, end={dists[-1]:.4f}, "
                  f"mean={np.mean(dists):.4f}")

    # --- Hard vs Soft update mechanics ---
    print("\n--- Update Mechanics Illustration ---")
    net_a = QNetwork(4, 2)
    net_b = QNetwork(4, 2)
    hard_update(net_b, net_a)

    # Simulate 10 gradient steps on net_a
    optimizer = optim.Adam(net_a.parameters(), lr=0.01)
    for _ in range(10):
        dummy_loss = net_a(torch.randn(16, 4)).sum()
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

    dist_before = compute_parameter_distance(net_a, net_b)
    print(f"  After 10 gradient steps, online-target distance: {dist_before:.4f}")

    # Soft update
    net_soft = copy.deepcopy(net_b)
    for _ in range(100):
        soft_update(net_soft, net_a, tau=0.01)
    dist_soft = compute_parameter_distance(net_a, net_soft)
    print(f"  After 100 soft updates (τ=0.01): distance = {dist_soft:.4f}")

    # Hard update
    net_hard = copy.deepcopy(net_b)
    hard_update(net_hard, net_a)
    dist_hard = compute_parameter_distance(net_a, net_hard)
    print(f"  After hard update: distance = {dist_hard:.6f}")

    print("\nTarget networks demo complete!")


if __name__ == "__main__":
    demo_target_networks()
