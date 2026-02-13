"""
33.1.4 DQN Implementation
==========================

Complete, production-ready DQN implementation with logging,
evaluation, and checkpointing on CartPole-v1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Optional
import random
import time
import json
import os

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Complete DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Full DQN agent with experience replay, target network, and logging."""

    def __init__(self, state_dim: int, action_dim: int,
                 # Network
                 hidden_dims: Tuple[int, ...] = (128, 128),
                 # Training
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 # Replay buffer
                 buffer_capacity: int = 50000,
                 min_buffer_size: int = 1000,
                 # Target network
                 target_update_freq: int = 200,
                 # Exploration
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 eps_decay_steps: int = 10000,
                 # Misc
                 grad_clip: float = 10.0,
                 loss_fn: str = 'huber',  # 'mse' or 'huber'
                 device: str = 'cpu'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.device = device

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # Networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        if loss_fn == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)

        # Counters
        self.total_steps = 0
        self.update_count = 0

        # Logging
        self.log: Dict[str, List] = {
            'losses': [],
            'q_values': [],
            'grad_norms': [],
            'epsilons': [],
        }

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[float]:
        """Perform one training step. Returns loss or None if buffer too small."""
        if len(self.buffer) < self.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values for taken actions
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = self.loss_fn(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)

        self.optimizer.step()
        self.update_count += 1

        # Target network update
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Logging
        loss_val = loss.item()
        self.log['losses'].append(loss_val)
        self.log['q_values'].append(q_values.mean().item())
        self.log['grad_norms'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        self.log['epsilons'].append(self.epsilon)

        return loss_val

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate with greedy policy (ε=0)."""
        returns = []
        lengths = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_r = 0.0
            steps = 0
            done = False
            while not done:
                action = self.select_action(state, training=False)
                state, r, term, trunc, _ = env.step(action)
                total_r += r
                steps += 1
                done = term or trunc
            returns.append(total_r)
            lengths.append(steps)
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_length': np.mean(lengths),
        }

    def save(self, path: str):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']
        self.update_count = ckpt['update_count']


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_dqn(env_name: str = 'CartPole-v1',
              n_episodes: int = 500,
              eval_freq: int = 50,
              eval_episodes: int = 10,
              seed: int = 42,
              **agent_kwargs) -> Tuple[DQNAgent, Dict]:
    """Complete DQN training loop with evaluation."""

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, **agent_kwargs)

    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_returns': [],
        'eval_episodes': [],
        'wall_time': [],
    }

    start_time = time.time()

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        ep_length = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward
            ep_length += 1

        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(ep_length)
        history['wall_time'].append(time.time() - start_time)

        # Periodic evaluation
        if ep % eval_freq == 0:
            eval_result = agent.evaluate(eval_env, eval_episodes)
            history['eval_returns'].append(eval_result['mean_return'])
            history['eval_episodes'].append(ep)

            # Rolling average
            recent = history['episode_rewards'][-50:]
            print(f"Episode {ep:>4d} | "
                  f"Avg50: {np.mean(recent):>7.1f} | "
                  f"Eval: {eval_result['mean_return']:>7.1f} ± {eval_result['std_return']:.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {agent.total_steps:,} | "
                  f"Loss: {np.mean(agent.log['losses'][-100:]):.4f} | "
                  f"Q̄: {np.mean(agent.log['q_values'][-100:]):.2f}")

    env.close()
    eval_env.close()
    return agent, history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_dqn_implementation():
    """Full DQN training demo on CartPole."""
    print("=" * 70)
    print("DQN Implementation Demo — CartPole-v1")
    print("=" * 70)

    agent, history = train_dqn(
        env_name='CartPole-v1',
        n_episodes=300,
        eval_freq=50,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=50000,
        min_buffer_size=1000,
        target_update_freq=200,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=5000,
        loss_fn='huber',
    )

    # --- Final evaluation ---
    print("\n--- Final Evaluation (20 episodes, greedy) ---")
    eval_env = gym.make('CartPole-v1')
    final_eval = agent.evaluate(eval_env, n_episodes=20)
    eval_env.close()
    for k, v in final_eval.items():
        print(f"  {k}: {v:.2f}")

    # --- Training statistics ---
    print("\n--- Training Statistics ---")
    rewards = history['episode_rewards']
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Total steps: {agent.total_steps:,}")
    print(f"  Total updates: {agent.update_count:,}")
    print(f"  Best episode reward: {max(rewards):.0f}")
    print(f"  Last 50 avg: {np.mean(rewards[-50:]):.1f}")

    if agent.log['losses']:
        print(f"  Final loss (avg 100): {np.mean(agent.log['losses'][-100:]):.4f}")
    if agent.log['q_values']:
        print(f"  Final Q-value (avg 100): {np.mean(agent.log['q_values'][-100:]):.2f}")

    # --- Save checkpoint ---
    save_path = 'dqn_cartpole.pt'
    agent.save(save_path)
    print(f"\n  Checkpoint saved to {save_path}")

    # --- Verify load ---
    agent2 = DQNAgent(4, 2)
    agent2.load(save_path)
    eval_env = gym.make('CartPole-v1')
    loaded_eval = agent2.evaluate(eval_env, n_episodes=5)
    eval_env.close()
    print(f"  Loaded checkpoint eval: {loaded_eval['mean_return']:.1f}")

    print("\nDQN implementation demo complete!")


if __name__ == "__main__":
    demo_dqn_implementation()
