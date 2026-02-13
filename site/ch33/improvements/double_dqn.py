"""
33.2.1 Double DQN
==================

Implementation of Double DQN with comparison to standard DQN
showing overestimation bias reduction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Dict
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.cap = capacity; self.size = 0; self.ptr = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, ns, d):
        self.s[self.ptr]=s; self.a[self.ptr]=a; self.r[self.ptr]=r
        self.ns[self.ptr]=ns; self.d[self.ptr]=float(d)
        self.ptr=(self.ptr+1)%self.cap; self.size=min(self.size+1, self.cap)

    def sample(self, n):
        i = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[i]), torch.LongTensor(self.a[i]),
                torch.FloatTensor(self.r[i]), torch.FloatTensor(self.ns[i]),
                torch.FloatTensor(self.d[i]))

    def __len__(self): return self.size


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# Double DQN vs Standard DQN target computation
# ---------------------------------------------------------------------------

def dqn_target(online_net: nn.Module, target_net: nn.Module,
               next_states: torch.Tensor, rewards: torch.Tensor,
               dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Standard DQN target: y = r + γ max_a' Q_target(s', a')"""
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        return rewards + (1 - dones) * gamma * next_q


def double_dqn_target(online_net: nn.Module, target_net: nn.Module,
                      next_states: torch.Tensor, rewards: torch.Tensor,
                      dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Double DQN target: y = r + γ Q_target(s', argmax_a' Q_online(s', a'))"""
    with torch.no_grad():
        # Online network selects action
        best_actions = online_net(next_states).argmax(dim=1)
        # Target network evaluates
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        return rewards + (1 - dones) * gamma * next_q


# ---------------------------------------------------------------------------
# Unified DQN Agent with double flag
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """DQN agent supporting both standard and double DQN."""

    def __init__(self, state_dim: int, action_dim: int, double: bool = True,
                 lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64,
                 buffer_cap: int = 50000, target_freq: int = 200,
                 eps_end: float = 0.01, eps_decay: int = 5000):
        self.double = double
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer_cap, state_dim)

        self.step = 0
        self.updates = 0
        self.q_history: List[float] = []
        self.loss_history: List[float] = []

    @property
    def epsilon(self):
        return max(self.eps_end, 1.0 - (1.0 - self.eps_end) * self.step / self.eps_decay)

    def act(self, state, training=True):
        if training:
            self.step += 1
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < 500:
            return
        s, a, r, ns, d = self.buf.sample(self.batch_size)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Record Q-values
        self.q_history.append(q.mean().item())

        if self.double:
            targets = double_dqn_target(self.online, self.target, ns, r, d, self.gamma)
        else:
            targets = dqn_target(self.online, self.target, ns, r, d, self.gamma)

        loss = nn.functional.smooth_l1_loss(q, targets)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        self.loss_history.append(loss.item())

        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


def train(agent, env_name='CartPole-v1', episodes=300):
    env = gym.make(env_name)
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, float(done))
            agent.update()
            s = ns; total += r
        rewards.append(total)
    env.close()
    return rewards


# ---------------------------------------------------------------------------
# Overestimation analysis
# ---------------------------------------------------------------------------

def measure_overestimation(env_name='CartPole-v1', episodes=200):
    """Compare Q-value estimates between DQN and Double DQN."""
    print("\n--- Overestimation Analysis ---")
    results = {}

    for name, use_double in [('Standard DQN', False), ('Double DQN', True)]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        env = gym.make(env_name)
        sd = env.observation_space.shape[0]
        ad = env.action_space.n
        env.close()

        agent = DoubleDQNAgent(sd, ad, double=use_double)
        rewards = train(agent, env_name, episodes)

        results[name] = {
            'rewards': rewards,
            'q_values': agent.q_history,
            'losses': agent.loss_history,
        }

        last50 = rewards[-50:]
        q_vals = agent.q_history[-500:] if agent.q_history else [0]
        print(f"\n  {name}:")
        print(f"    Reward (last 50): {np.mean(last50):.1f} ± {np.std(last50):.1f}")
        print(f"    Mean Q-value (last 500): {np.mean(q_vals):.2f}")
        print(f"    Max Q-value: {max(agent.q_history) if agent.q_history else 0:.2f}")

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_double_dqn():
    print("=" * 60)
    print("Double DQN Demo")
    print("=" * 60)

    results = measure_overestimation('CartPole-v1', episodes=250)

    # Compare
    print("\n--- Summary ---")
    for name, data in results.items():
        r = data['rewards']
        q = data['q_values']
        print(f"  {name}:")
        print(f"    Final avg reward: {np.mean(r[-50:]):.1f}")
        print(f"    Q-value range: [{min(q):.2f}, {max(q):.2f}]")
        print(f"    Q-value mean: {np.mean(q):.2f}")

    # Show the difference is in Q-value scale
    if all(r['q_values'] for r in results.values()):
        dqn_q = np.mean(results['Standard DQN']['q_values'][-200:])
        ddqn_q = np.mean(results['Double DQN']['q_values'][-200:])
        print(f"\n  Overestimation reduction: {dqn_q - ddqn_q:.2f} "
              f"({(1 - ddqn_q/dqn_q)*100:.1f}% lower Q-values)")

    print("\nDouble DQN demo complete!")


if __name__ == "__main__":
    demo_double_dqn()
