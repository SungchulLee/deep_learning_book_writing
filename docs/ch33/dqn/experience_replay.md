# 경험 되살리기

겪음 되돌려 보기는 옮김을 버퍼에 담고 아무 작은 묶음을 뽑아 배워 익히기 자료의 때 얽힘을 끊는, 깊은 Q 그물의 결정적인 조각이다. 이 짜기는 되돌려 보기 버퍼의 여러 변형을 준다. 기본 두 끝 줄 바탕 버퍼, 효율 좋은 NumPy 배열 바탕 둥근 버퍼, 늘 가장 최근 옮김을 담는 아우른 겪음 되돌려 보기 버퍼, 픽셀 관찰을 위한 기억을 아끼는 틀 쌓기 버퍼가 그것이다. 이 여러 짜기가 단순함, 기억 효율, 표본 다양함 사이의 맞바꿈을 보여 준다.

## 1. 코드

```python
"""
33.1.2 겪음 되돌려 보기
========================

여러 되돌려 보기 담개 짜기: 기본 두 끝 줄 바탕, 빠른
NumPy 바탕, 그리고 섞은 겪음 되돌려 보기.
"""

import torch
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, Optional, Dict
import random
import time

# ========================================================================
# 메인
# ========================================================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# 1. 기본 두 끝 줄 바탕 되돌려 보기 담개
# ---------------------------------------------------------------------------

class BasicReplayBuffer:
    """간단한 두 끝 줄 바탕 되돌려 보기 담개.
    
    좋은 점: 짜기 쉽고 길이가 바뀌는 마당을 다룬다.
    나쁜 점: 파이썬 물체 덧짐이 있고 큰 담개에서는 느리다.
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
# 2. 빠른 NumPy 바탕 되돌려 보기 담개
# ---------------------------------------------------------------------------

class EfficientReplayBuffer:
    """NumPy 배열 바탕 고리 꼴 되돌려 보기 담개.
    
    빠르기를 위해 이어진 NumPy 배열로 기억을 미리 잡는다.
    파이썬 물체 덧짐을 피하고 빠른 묶음 번호 매기기를 할 수 있다.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0  # 적기 가리개(고리 꼴)

        # 배열을 미리 잡음
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
# 3. 섞은 겪음 되돌려 보기(CER)
# ---------------------------------------------------------------------------

class CombinedReplayBuffer:
    """섞은 겪음 되돌려 보기: 가장 최근 옮김을 늘 넣는다.
    
    참고: Zhang & Sutton (2017), "A Deeper Look at Experience Replay"
    
    가장 최근 옮김이 늘 묶음에 들어가 새 겪음이 곧바로 배움에
    쓰이도록 보장한다.
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

        # 아무 옮김 batch_size - 1개와 가장 최근 것 하나를 뽑음
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
# 4. 틀 쌓기 되돌려 보기 담개(아타리용)
# ---------------------------------------------------------------------------

class FrameStackReplayBuffer:
    """틀을 쌓은 살핌을 위한 기억 아끼는 되돌려 보기 담개.
    
    옮김마다 쌓은 그림 4장을 갈무리하는 대신(겹치는 자료),
    틀을 낱낱이 갈무리하고 뽑을 때 쌓음을 다시 짓는다.
    기억 아낌: 틀 4개 쌓음에서 75%쯤.
    """

    def __init__(self, capacity: int, frame_shape: Tuple[int, int] = (84, 84),
                 n_stack: int = 4):
        self.capacity = capacity
        self.n_stack = n_stack
        self.size = 0
        self.ptr = 0

        # 틀을 낱낱이 갈무리(쌓음이 아님)
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, frame: np.ndarray, action: int, reward: float, done: bool):
        """틀 하나의 옮김을 갈무리한다."""
        self.frames[self.ptr] = frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stack(self, idx: int) -> np.ndarray:
        """idx에서 끝나는 쌓은 살핌을 다시 짓는다."""
        indices = []
        for i in range(self.n_stack):
            frame_idx = (idx - i) % self.capacity
            indices.append(frame_idx)
            # 끝맺음에 닿거나 담개 시작 앞으로 가면 0으로 채움
            if self.dones[frame_idx] and i > 0:
                # 남은 틀은 쓸 수 있는 첫 틀로 채움
                indices.extend([frame_idx] * (self.n_stack - len(indices)))
                break
        indices.reverse()
        return self.frames[indices]  # (n_stack, H, W)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        # 담개 경계에 너무 가까운 자리에서는 뽑지 않음
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
# 되돌려 보기 담개 셈밝힘
# ---------------------------------------------------------------------------

def buffer_statistics(buffer, n_samples: int = 1000) -> Dict[str, float]:
    """뽑은 옮김의 셈밝힘을 셈한다."""
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
# 시연
# ---------------------------------------------------------------------------

def demo_experience_replay():
    """되돌려 보기 담개 짜기를 견준다."""
    import gymnasium as gym

    print("=" * 60)
    print("Experience Replay Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]

    # --- 담개 채우기 ---
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

    # --- 뽑기 빠르기 잣대 재기 ---
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

    # --- 담개 셈밝힘 ---
    print("\n--- Buffer Statistics ---")
    for name, buf in buffers.items():
        stats = buffer_statistics(buf)
        print(f"\n  {name}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")

    # --- 이어짐 풀기 보이기 ---
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

    # 차례대로와 아무렇게: 때에 따른 제 상관을 살핌
    seq_states = torch.FloatTensor(buf.states[:100])
    rand_states, *_ = buf.sample(100)

    # 잇닿은 뽑음 사이의 평균 짝별 거리를 셈
    seq_diffs = (seq_states[1:] - seq_states[:-1]).norm(dim=1).mean().item()
    rand_diffs = (rand_states[1:] - rand_states[:-1]).norm(dim=1).mean().item()
    print(f"  Mean consecutive distance (sequential): {seq_diffs:.4f}")
    print(f"  Mean consecutive distance (random):     {rand_diffs:.4f}")
    print(f"  Random sampling increases diversity by {rand_diffs/seq_diffs:.2f}x")

    # --- 기억 씀씀이 견주기 ---
    print("\n--- Memory Usage Estimates ---")
    for cap in [10_000, 100_000, 1_000_000]:
        deque_bytes = cap * (state_dim * 4 * 2 + 8 + 4 + 4 + 64)  # 어림값
        numpy_bytes = cap * (state_dim * 4 * 2 + 8 + 4 + 4)  # float32 배열
        print(f"  Capacity {cap:>10,d}: deque ~{deque_bytes/1e6:.1f} MB, "
              f"numpy ~{numpy_bytes/1e6:.1f} MB")

    env.close()
    print("\nExperience replay demo complete!")


if __name__ == "__main__":
    demo_experience_replay()
```

## 2. 논의

이 짜기는 기본 두 끝 줄 바탕과 효율 좋은 판의 핵심 논리를 감싼 `BasicReplayBuffer`, `EfficientReplayBuffer`, `CombinedReplayBuffer` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 쓰는 모습을 보인다. 내놓기를 살피면 웃잡 고름과 문제 짜임에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 깊은 Q 그물에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 경험 되살리기

이 짜기는 기본 두 끝 줄 바탕과 효율 좋은 판의 핵심 논리를 감싼 `BasicReplayBuffer`, `EfficientReplayBuffer`, `CombinedReplayBuffer` 갈래를 한가운데 둔다.

고갱이 갈래는 `BasicReplayBuffer`, `EfficientReplayBuffer`, `CombinedReplayBuffer`, `FrameStackReplayBuffer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
