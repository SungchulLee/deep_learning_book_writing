"""
33.3.1 N-Step Returns
======================

N-step return computation, n-step replay buffer, and DQN with n-step targets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Tuple, List, Optional
import random


# ---------------------------------------------------------------------------
# N-Step Return Buffer
# ---------------------------------------------------------------------------

class NStepBuffer:
    """Accumulates n transitions and computes n-step returns."""

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done) -> Optional[Tuple]:
        """Add transition. Returns completed n-step transition or None."""
        self.buffer.append((state, action, reward, next_state, done))

        if done:
            # Flush all remaining transitions at episode end
            results = []
            while self.buffer:
                results.append(self._compute_return())
                self.buffer.popleft()
            return results

        if len(self.buffer) == self.n_step:
            result = self._compute_return()
            return [result]

        return None

    def _compute_return(self) -> Tuple:
        """Compute n-step return from current buffer contents."""
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                break

        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        sn = self.buffer[-1][3]
        done = self.buffer[-1][4]
        n_actual = len(self.buffer)

        return (s0, a0, R, sn, done, n_actual)

    def reset(self):
        self.buffer.clear()


# ---------------------------------------------------------------------------
# N-Step Replay Buffer
# ---------------------------------------------------------------------------

class NStepReplayBuffer:
    """Replay buffer that stores pre-computed n-step transitions."""

    def __init__(self, capacity: int, state_dim: int, n_step: int = 3,
                 gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_buf = NStepBuffer(n_step, gamma)

        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.returns = np.zeros(capacity, np.float32)  # n-step return
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.n_steps = np.zeros(capacity, np.int32)  # actual steps used

        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        results = self.nstep_buf.push(state, action, reward, next_state, done)
        if results:
            for s0, a0, R, sn, d, n_actual in results:
                self.states[self.ptr] = s0
                self.actions[self.ptr] = a0
                self.returns[self.ptr] = R
                self.next_states[self.ptr] = sn
                self.dones[self.ptr] = float(d)
                self.n_steps[self.ptr] = n_actual
                self.ptr = (self.ptr + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.returns[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
            torch.IntTensor(self.n_steps[idx]),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# N-Step DQN Agent
# ---------------------------------------------------------------------------

class NStepDQNAgent:
    """DQN with n-step returns and Double DQN."""

    def __init__(self, state_dim, action_dim, n_step=3, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.01, eps_decay=5000):
        self.gamma = gamma
        self.gamma_n = gamma  # Will be raised to power n per sample
        self.n_step = n_step
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = NStepReplayBuffer(buf_cap, state_dim, n_step, gamma)

        self.step = 0
        self.updates = 0

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
        s, a, R, ns, d, n_actual = self.buf.sample(self.batch_size)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            best_a = self.online(ns).argmax(1)
            next_q = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            # γ^n varies per sample based on actual steps
            gamma_n = self.gamma ** n_actual.float()
            targets = R + (1 - d) * gamma_n * next_q

        loss = nn.functional.smooth_l1_loss(q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_nstep_returns():
    print("=" * 60)
    print("N-Step Returns Demo")
    print("=" * 60)

    # --- N-step return computation ---
    print("\n--- N-Step Return Examples ---")
    gamma = 0.99
    rewards = [1.0, 0.5, 2.0, 1.0, 0.0]
    for n in [1, 2, 3, 5]:
        R = sum(gamma**k * rewards[k] for k in range(min(n, len(rewards))))
        print(f"  n={n}: R^(n) = {R:.4f} + γ^{n} * Q(s_{n})")

    # --- Compare n-step values ---
    print("\n--- Training: Compare n=1, n=3, n=5 ---")
    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n
    env.close()

    results = {}
    for n in [1, 3, 5]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = NStepDQNAgent(sd, ad, n_step=n, lr=1e-3)
        env = gym.make('CartPole-v1')
        rewards_hist = []
        for ep in range(250):
            s, _ = env.reset(); total = 0; done = False
            while not done:
                a = agent.act(s)
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                agent.store(s, a, r, ns, done)
                agent.update()
                s = ns; total += r
            rewards_hist.append(total)
        env.close()
        results[n] = rewards_hist
        avg = np.mean(rewards_hist[-50:])
        print(f"  n={n}: avg(last50)={avg:.1f}, max={max(rewards_hist):.0f}")

    # --- Summary ---
    print("\n--- Comparison ---")
    for n, rews in results.items():
        last50 = rews[-50:]
        print(f"  n={n}: mean={np.mean(last50):.1f}, std={np.std(last50):.1f}")

    print("\nN-step returns demo complete!")


if __name__ == "__main__":
    demo_nstep_returns()
