"""
33.2.3 Prioritized Experience Replay
======================================

Sum-tree based prioritized replay buffer with importance sampling.
"""

import torch
import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Sum Tree
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum tree for O(log n) prioritized sampling.
    
    Leaves store priorities; internal nodes store partial sums.
    Total priority = root node value.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Update parent nodes after a leaf change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find the leaf node for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def update(self, idx: int, priority: float):
        """Update priority of leaf at tree index idx."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns data index."""
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def sample(self, value: float) -> Tuple[int, float, int]:
        """Sample leaf given cumulative value. Returns (tree_idx, priority, data_idx)."""
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return self.tree[self.capacity - 1: self.capacity - 1 + self.size].max()

    @property
    def min_priority(self) -> float:
        priorities = self.tree[self.capacity - 1: self.capacity - 1 + self.size]
        return priorities[priorities > 0].min() if (priorities > 0).any() else 1e-6


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with sum-tree sampling and IS correction."""

    def __init__(self, capacity: int, state_dim: int,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_steps: int = 100000,
                 epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.step_count = 0

        self.tree = SumTree(capacity)

        # Data storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    @property
    def beta(self) -> float:
        frac = min(1.0, self.step_count / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done, td_error: float = None):
        """Add transition. If td_error not given, use max priority."""
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = (self.tree.max_priority if self.tree.size > 0 else 1.0) ** self.alpha

        data_idx = self.tree.add(priority)
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = float(done)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with priorities. Returns (s, a, r, ns, d, weights, tree_indices)."""
        self.step_count += 1
        tree_indices = []
        data_indices = []

        # Stratified sampling: divide total priority into equal segments
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx, priority, data_idx = self.tree.sample(value)
            tree_indices.append(tree_idx)
            data_indices.append(data_idx)

        data_indices = np.array(data_indices)

        # Importance sampling weights
        beta = self.beta
        min_prob = self.tree.min_priority / (self.tree.total + 1e-8)
        max_weight = (min_prob * self.tree.size + 1e-8) ** (-beta)

        priorities = np.array([self.tree.tree[ti] for ti in tree_indices])
        probs = priorities / (self.tree.total + 1e-8)
        weights = (probs * self.tree.size + 1e-8) ** (-beta)
        weights = weights / (max_weight + 1e-8)  # Normalize

        return (
            torch.FloatTensor(self.states[data_indices]),
            torch.LongTensor(self.actions[data_indices]),
            torch.FloatTensor(self.rewards[data_indices]),
            torch.FloatTensor(self.next_states[data_indices]),
            torch.FloatTensor(self.dones[data_indices]),
            torch.FloatTensor(weights),
            tree_indices,
        )

    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray):
        """Update priorities after computing new TD errors."""
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_prioritized_replay():
    import gymnasium as gym
    import torch.nn as nn

    print("=" * 60)
    print("Prioritized Experience Replay Demo")
    print("=" * 60)

    # --- Sum Tree basics ---
    print("\n--- Sum Tree ---")
    tree = SumTree(8)
    priorities = [3.0, 1.0, 5.0, 2.0, 4.0]
    for p in priorities:
        tree.add(p)
    print(f"  Priorities: {priorities}")
    print(f"  Total: {tree.total}")
    print(f"  Max: {tree.max_priority}")

    # Sample distribution
    counts = np.zeros(5)
    for _ in range(10000):
        v = np.random.uniform(0, tree.total)
        _, _, di = tree.sample(v)
        counts[di] += 1
    counts /= counts.sum()
    expected = np.array(priorities) / sum(priorities)
    print(f"  Expected probs: {expected.round(3)}")
    print(f"  Sampled probs:  {counts.round(3)}")

    # --- PER with CartPole ---
    print("\n--- PER Training Demo ---")
    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    per_buf = PrioritizedReplayBuffer(10000, sd, alpha=0.6, beta_start=0.4)
    q_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    rewards_hist = []
    step = 0
    for ep in range(200):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            step += 1
            eps = max(0.01, 1.0 - step / 5000)
            if np.random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            per_buf.push(s, a, r, ns, float(done))

            if len(per_buf) >= 500:
                st, at, rt, nst, dt, w, ti = per_buf.sample(64)
                q = q_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = target_net(nst).max(1)[0]
                    tgt = rt + (1 - dt) * 0.99 * nq
                td_err = (tgt - q).detach().numpy()
                loss = (w * (q - tgt).pow(2)).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                per_buf.update_priorities(ti, td_err)

                if step % 200 == 0:
                    target_net.load_state_dict(q_net.state_dict())

            s = ns; total += r
        rewards_hist.append(total)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards_hist[-50:]):.1f}, "
                  f"β={per_buf.beta:.3f}")

    env.close()

    # --- IS weight analysis ---
    print("\n--- IS Weight Analysis ---")
    if len(per_buf) >= 64:
        _, _, _, _, _, weights, _ = per_buf.sample(64)
        print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Weight mean: {weights.mean():.4f}")
        print(f"  Weight std: {weights.std():.4f}")
        print(f"  Current β: {per_buf.beta:.4f}")

    print("\nPrioritized replay demo complete!")


if __name__ == "__main__":
    demo_prioritized_replay()
