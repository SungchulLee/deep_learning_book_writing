"""
33.1.1 DQN Fundamentals
========================

Core DQN components: Q-network architectures, action selection,
and the basic DQN training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, Optional
import random

# ---------------------------------------------------------------------------
# Transition tuple
# ---------------------------------------------------------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Q-Network Architectures
# ---------------------------------------------------------------------------

class MLPQNetwork(nn.Module):
    """MLP Q-Network for vector observation spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions given state batch."""
        return self.net(x)


class ConvQNetwork(nn.Module):
    """Convolutional Q-Network for Atari-style pixel observations.
    
    Input: (batch, 4, 84, 84) — 4 stacked grayscale frames.
    Output: (batch, action_dim) — Q-value per action.
    """

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute flattened size: 64 * 7 * 7 = 3136 for 84x84 input
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        features = self.conv(x).view(x.size(0), -1)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Epsilon-Greedy Action Selection
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """Epsilon-greedy action selection with linear annealing."""

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.01,
                 eps_decay_steps: int = 10000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, q_values: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            q_values: Q-values for current state, shape (1, action_dim)
            training: if False, always take greedy action
            
        Returns:
            Selected action index
        """
        if training:
            self.step_count += 1
            if random.random() < self.epsilon:
                return random.randrange(q_values.shape[1])
        return q_values.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# Basic DQN Agent (simplified, no replay/target yet)
# ---------------------------------------------------------------------------

class BasicDQNAgent:
    """Minimal DQN agent demonstrating core Q-learning with neural networks.
    
    Note: This is intentionally simplified. Full DQN with experience replay
    and target networks is in implementation.py.
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, hidden_dims: Tuple[int, ...] = (128, 128),
                 device: str = 'cpu'):
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim

        self.q_network = MLPQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.explorer = EpsilonGreedy()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return self.explorer.select_action(q_values, training)

    def update(self, state, action, reward, next_state, done) -> float:
        """Single-sample online update (for illustration; not recommended)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Current Q-value
        q_value = self.q_network(state_t)[0, action]

        # TD target
        with torch.no_grad():
            next_q = self.q_network(next_state_t).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        # MSE loss
        loss = nn.functional.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ---------------------------------------------------------------------------
# TD Error Computation
# ---------------------------------------------------------------------------

def compute_td_error(q_network: nn.Module, target_network: nn.Module,
                     states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor,
                     dones: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute TD errors for a batch of transitions.
    
    Args:
        q_network: Online Q-network
        target_network: Target Q-network
        states: (batch, state_dim)
        actions: (batch,)
        rewards: (batch,)
        next_states: (batch, state_dim)
        dones: (batch,) — 1.0 if terminal
        gamma: discount factor
        
    Returns:
        TD errors of shape (batch,)
    """
    # Q(s, a) for the taken actions
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: r + γ max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones) * gamma * next_q_values

    td_errors = targets - q_values
    return td_errors


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_fundamentals():
    """Demonstrate DQN fundamentals on CartPole."""
    print("=" * 60)
    print("DQN Fundamentals Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # --- Q-Network demo ---
    print("\n--- MLP Q-Network ---")
    q_net = MLPQNetwork(state_dim, action_dim)
    print(f"Architecture:\n{q_net}")
    total_params = sum(p.numel() for p in q_net.parameters())
    print(f"Total parameters: {total_params:,}")

    state, _ = env.reset()
    state_t = torch.FloatTensor(state).unsqueeze(0)
    q_values = q_net(state_t)
    print(f"\nQ-values for initial state: {q_values.detach().numpy()}")
    print(f"Greedy action: {q_values.argmax().item()}")

    # --- Epsilon-greedy demo ---
    print("\n--- Epsilon-Greedy Schedule ---")
    explorer = EpsilonGreedy(eps_start=1.0, eps_end=0.01, eps_decay_steps=1000)
    for step in [0, 250, 500, 750, 1000, 2000]:
        explorer.step_count = step
        print(f"  Step {step:>5d}: ε = {explorer.epsilon:.4f}")

    # --- Basic online DQN (few episodes, for illustration) ---
    print("\n--- Basic Online DQN (simplified, 50 episodes) ---")
    agent = BasicDQNAgent(state_dim, action_dim, lr=1e-3)
    episode_rewards = []

    for ep in range(50):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep+1:>3d}: reward = {total_reward:.0f}, "
                  f"avg(10) = {avg:.1f}, ε = {agent.explorer.epsilon:.3f}")

    # --- TD error computation demo ---
    print("\n--- TD Error Computation ---")
    target_net = MLPQNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    batch_states = torch.randn(8, state_dim)
    batch_actions = torch.randint(0, action_dim, (8,))
    batch_rewards = torch.randn(8)
    batch_next_states = torch.randn(8, state_dim)
    batch_dones = torch.zeros(8)

    td_errors = compute_td_error(q_net, target_net, batch_states, batch_actions,
                                  batch_rewards, batch_next_states, batch_dones)
    print(f"TD errors (batch of 8): {td_errors.detach().numpy().round(4)}")
    print(f"Mean absolute TD error: {td_errors.abs().mean().item():.4f}")

    # --- Conv Q-Network shape check ---
    print("\n--- Conv Q-Network (Atari-style) ---")
    conv_net = ConvQNetwork(action_dim=4)
    dummy_frames = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    conv_q = conv_net(dummy_frames.float())
    print(f"Input shape: {dummy_frames.shape}")
    print(f"Output Q-values shape: {conv_q.shape}")
    conv_params = sum(p.numel() for p in conv_net.parameters())
    print(f"Total parameters: {conv_params:,}")

    env.close()
    print("\nFundamentals demo complete!")


if __name__ == "__main__":
    demo_fundamentals()
