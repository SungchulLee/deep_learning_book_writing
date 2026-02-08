"""
33.1.2 Experience Replay
========================

Multiple replay buffer implementations: basic deque-based, efficient
NumPy-based, and combined experience replay.
"""

import torch
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, Optional, Dict
import random
import time

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# 1. Basic Deque-Based Replay Buffer
# ---------------------------------------------------------------------------

class BasicReplayBuffer:
    """Simple deque-based replay buffer.
    
    Pros: Easy to implement, handles variable-length episodes.
    Cons: Python object overhead, slower for large buffers.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return (
            torch.FloatTensor(np.array(batch.state)),
            torch.LongTensor(np.array(batch.action)),
            torch.FloatTensor(np.array(batch.reward)),
            torch.FloatTensor(np.array(batch.next_state)),
            torch.FloatTensor(np.array(batch.done, dtype=np.float32)),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 2. Efficient NumPy-Based Replay Buffer
# ---------------------------------------------------------------------------

class EfficientReplayBuffer:
    """NumPy array-based circular replay buffer.
    
    Pre-allocates memory as contiguous NumPy arrays for efficiency.
    Avoids Python object overhead and enables fast batch indexing.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0  # Write pointer (circular)

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# 3. Combined Experience Replay (CER)
# ---------------------------------------------------------------------------

class CombinedReplayBuffer:
    """Combined Experience Replay: always includes the latest transition.
    
    Reference: Zhang & Sutton (2017), "A Deeper Look at Experience Replay"
    
    The most recent transition is always included in the batch, ensuring
    new experiences are immediately used for learning.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.buffer = EfficientReplayBuffer(capacity, state_dim)
        self.last_idx: Optional[int] = None

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.last_idx = self.buffer.ptr
        self.buffer.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if self.last_idx is None or batch_size <= 1:
            return self.buffer.sample(batch_size)

        # Sample batch_size - 1 random transitions + the latest one
        indices = np.random.randint(0, self.buffer.size, size=batch_size - 1)
        indices = np.append(indices, self.last_idx)
        return (
            torch.FloatTensor(self.buffer.states[indices]),
            torch.LongTensor(self.buffer.actions[indices]),
            torch.FloatTensor(self.buffer.rewards[indices]),
            torch.FloatTensor(self.buffer.next_states[indices]),
            torch.FloatTensor(self.buffer.dones[indices]),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 4. Frame-Stacking Replay Buffer (for Atari)
# ---------------------------------------------------------------------------

class FrameStackReplayBuffer:
    """Memory-efficient replay buffer for frame-stacked observations.
    
    Instead of storing 4 stacked frames per transition (redundant),
    stores individual frames and reconstructs stacks on sampling.
    Memory savings: ~75% for 4-frame stacks.
    """

    def __init__(self, capacity: int, frame_shape: Tuple[int, int] = (84, 84),
                 n_stack: int = 4):
        self.capacity = capacity
        self.n_stack = n_stack
        self.size = 0
        self.ptr = 0

        # Store individual frames (not stacks)
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, frame: np.ndarray, action: int, reward: float, done: bool):
        """Store a single frame transition."""
        self.frames[self.ptr] = frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stack(self, idx: int) -> np.ndarray:
        """Reconstruct a stacked observation ending at idx."""
        indices = []
        for i in range(self.n_stack):
            frame_idx = (idx - i) % self.capacity
            indices.append(frame_idx)
            # If we hit a terminal or go before buffer start, pad with zeros
            if self.dones[frame_idx] and i > 0:
                # Pad remaining frames with the first available frame
                indices.extend([frame_idx] * (self.n_stack - len(indices)))
                break
        indices.reverse()
        return self.frames[indices]  # (n_stack, H, W)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        # Avoid sampling from positions too close to buffer boundaries
        valid = np.arange(self.n_stack, self.size)
        indices = np.random.choice(valid, size=batch_size, replace=False)

        states = np.array([self._get_stack(i) for i in indices])
        next_states = np.array([self._get_stack((i + 1) % self.capacity) for i in indices])

        return (
            torch.FloatTensor(states),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(next_states),
            torch.FloatTensor(self.dones[indices].astype(np.float32)),
        )

    def __len__(self) -> int:
        return max(0, self.size - self.n_stack)


# ---------------------------------------------------------------------------
# Replay buffer statistics
# ---------------------------------------------------------------------------

def buffer_statistics(buffer, n_samples: int = 1000) -> Dict[str, float]:
    """Compute statistics over sampled transitions."""
    if len(buffer) < n_samples:
        n_samples = len(buffer)
    states, actions, rewards, next_states, dones = buffer.sample(n_samples)
    return {
        'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item(),
        'min_reward': rewards.min().item(),
        'max_reward': rewards.max().item(),
        'done_fraction': dones.mean().item(),
        'mean_state_norm': states.norm(dim=1).mean().item(),
        'unique_actions': len(actions.unique()),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_experience_replay():
    """Compare replay buffer implementations."""
    import gymnasium as gym

    print("=" * 60)
    print("Experience Replay Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]

    # --- Fill buffers ---
    buffers = {
        'Basic (deque)': BasicReplayBuffer(10000),
        'Efficient (numpy)': EfficientReplayBuffer(10000, state_dim),
        'Combined (CER)': CombinedReplayBuffer(10000, state_dim),
    }

    print("\nFilling buffers with 5000 transitions...")
    state, _ = env.reset()
    for i in range(5000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        for buf in buffers.values():
            buf.push(state, action, reward, next_state, done)
        if done:
            state, _ = env.reset()
        else:
            state = next_state

    # --- Benchmark sampling speed ---
    print("\n--- Sampling Speed Benchmark ---")
    batch_size = 64
    n_samples = 1000

    for name, buf in buffers.items():
        start = time.time()
        for _ in range(n_samples):
            buf.sample(batch_size)
        elapsed = time.time() - start
        print(f"  {name:>20s}: {n_samples} samples in {elapsed:.3f}s "
              f"({n_samples/elapsed:.0f} batches/sec)")

    # --- Buffer statistics ---
    print("\n--- Buffer Statistics ---")
    for name, buf in buffers.items():
        stats = buffer_statistics(buf)
        print(f"\n  {name}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")

    # --- Demonstrate decorrelation ---
    print("\n--- Decorrelation Demo ---")
    buf = EfficientReplayBuffer(10000, state_dim)
    state, _ = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.push(state, action, reward, next_state, done)
        if done:
            state, _ = env.reset()
        else:
            state = next_state

    # Sequential vs random: check temporal autocorrelation
    seq_states = torch.FloatTensor(buf.states[:100])
    rand_states, *_ = buf.sample(100)

    # Compute mean pairwise distance between consecutive samples
    seq_diffs = (seq_states[1:] - seq_states[:-1]).norm(dim=1).mean().item()
    rand_diffs = (rand_states[1:] - rand_states[:-1]).norm(dim=1).mean().item()
    print(f"  Mean consecutive distance (sequential): {seq_diffs:.4f}")
    print(f"  Mean consecutive distance (random):     {rand_diffs:.4f}")
    print(f"  Random sampling increases diversity by {rand_diffs/seq_diffs:.2f}x")

    # --- Memory usage comparison ---
    print("\n--- Memory Usage Estimates ---")
    for cap in [10_000, 100_000, 1_000_000]:
        deque_bytes = cap * (state_dim * 4 * 2 + 8 + 4 + 4 + 64)  # rough estimate
        numpy_bytes = cap * (state_dim * 4 * 2 + 8 + 4 + 4)  # float32 arrays
        print(f"  Capacity {cap:>10,d}: deque ~{deque_bytes/1e6:.1f} MB, "
              f"numpy ~{numpy_bytes/1e6:.1f} MB")

    env.close()
    print("\nExperience replay demo complete!")


if __name__ == "__main__":
    demo_experience_replay()
