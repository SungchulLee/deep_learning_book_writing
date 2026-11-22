# 경험 리플레이

경험 리플레이는 DQN의 핵심 구성 요소다. 에이전트가 겪은 **전이**(transition) — 상태·행동·리워드·다음 상태·종료 여부 다섯을 묶은 것 — 를 버퍼에 쌓아 두고, 트레인할 때는 그 가운데서 무작위 미니배치를 뽑는다. 연달아 겪은 전이끼리는 서로 많이 닮았는데, 무작위로 뽑으면 이 시간 상관이 끊긴다. 이 구현은 리플레이 버퍼의 네 가지 변형을 준다. 기본 덱(deque) 기반 버퍼, NumPy 배열을 미리 잡아 두는 원형 버퍼, 늘 가장 최근 전이를 배치에 넣는 결합 경험 리플레이(CER) 버퍼, 픽셀 관찰을 위한 메모리 절약형 프레임 스택 버퍼다. 이들은 단순함, 메모리 효율, 표본 다양성 사이의 트레이드오프를 보여 준다.

## 1. 코드

```python
"""
35.1.2 겪음 되돌려 보기
========================

여러 되돌려 보기 버퍼 짜기: 기본 두 끝 줄 바탕, 빠른
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
# 1. 기본 두 끝 줄 바탕 되돌려 보기 버퍼
# ---------------------------------------------------------------------------

class BasicReplayBuffer:
    """간단한 두 끝 줄 바탕 되돌려 보기 버퍼.
    
    좋은 점: 짜기 쉽고 길이가 바뀌는 마당을 다룬다.
    나쁜 점: 파이썬 물체 덧짐이 있고 큰 버퍼에서는 느리다.
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
# 2. 빠른 NumPy 바탕 되돌려 보기 버퍼
# ---------------------------------------------------------------------------

class EfficientReplayBuffer:
    """NumPy 배열 바탕 고리 형태 되돌려 보기 버퍼.
    
    빠르기를 위해 이어진 NumPy 배열로 기억을 미리 잡는다.
    파이썬 물체 덧짐을 피하고 빠른 배치 번호 매기기를 할 수 있다.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0  # 적기 마스크(고리 형태)

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
    
    가장 최근 옮김이 늘 배치에 들어가 새 겪음이 곧바로 러닝에
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
# 4. 틀 쌓기 되돌려 보기 버퍼(아타리용)
# ---------------------------------------------------------------------------

class FrameStackReplayBuffer:
    """틀을 쌓은 살핌을 위한 기억 아끼는 되돌려 보기 버퍼.
    
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
            # 끝맺음에 닿거나 버퍼 시작 앞으로 가면 0으로 채움
            if self.dones[frame_idx] and i > 0:
                # 남은 틀은 쓸 수 있는 첫 틀로 채움
                indices.extend([frame_idx] * (self.n_stack - len(indices)))
                break
        indices.reverse()
        return self.frames[indices]  # (n_stack, H, W)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        # 버퍼 경계에 너무 가까운 자리에서는 뽑지 않음
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
# 되돌려 보기 버퍼 셈밝힘
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
    """되돌려 보기 버퍼 짜기를 견준다."""
    import gymnasium as gym

    print("=" * 60)
    print("Experience Replay Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]

    # --- 버퍼 채우기 ---
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

    # --- 버퍼 셈밝힘 ---
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

이 구현은 네 버퍼 클래스 `BasicReplayBuffer`, `EfficientReplayBuffer`, `CombinedReplayBuffer`, `FrameStackReplayBuffer` 를 중심에 둔다. 넷 다 `push` 로 전이를 넣고 `sample` 로 텐서 다섯 개(상태, 행동, 리워드, 다음 상태, 종료)를 돌려주는 같은 인터페이스를 가지므로, 에이전트 코드를 바꾸지 않고 서로 갈아 끼울 수 있다.

데모 함수는 CartPole 에서 무작위 행동으로 전이 5,000개를 모아 앞의 세 버퍼에 똑같이 넣고, 네 가지를 잰다. 미니배치 1,000번을 뽑는 데 걸리는 시간, 뽑은 표본의 리워드·종료 비율 같은 통계, 순서대로 이은 상태와 무작위로 뽑은 상태의 이웃 간 평균 거리(상관이 얼마나 끊기는지), 그리고 용량별 메모리 사용량 어림값이다. 트레인은 하지 않으므로 이 데모로 알 수 있는 것은 버퍼의 속도와 표본의 성질이지 에이전트의 성능이 아니다.

`FrameStackReplayBuffer` 는 데모에서 쓰이지 않는다. 이 버퍼는 쌓은 4프레임을 전이마다 통째로 저장하는 대신 프레임을 한 장씩만 저장하고 뽑을 때 스택을 다시 만들어, 프레임 4장 스택 기준으로 메모리를 75%쯤 줄인다. 아타리처럼 관찰이 이미지인 환경에서 쓰는 방식이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 뽑기 속도 벤치마크에서 `batch_size` 를 32, 64, 256 으로 바꿔 돌려 보라. `BasicReplayBuffer` 와 `EfficientReplayBuffer` 의 속도 차이는 배치 크기에 따라 어떻게 변하리라 예상하는가? 코드의 어느 줄이 그 차이를 만드는가?

</div>

??? success "풀이"
    `BasicReplayBuffer.sample` 은 `random.sample` 로 파이썬 객체 `batch_size` 개를 고른 뒤 `zip` 으로 풀고 `np.array` 로 다시 모은다. 이 작업은 전이 하나하나를 파이썬 수준에서 건드리므로 비용이 배치 크기에 비례해 커진다. `EfficientReplayBuffer.sample` 은 미리 잡아 둔 배열에 인덱스 배열을 한 번 넣는 팬시 인덱싱이라 반복이 C 수준에서 돈다. 따라서 배치가 클수록 두 버퍼의 차이가 벌어지는 쪽을 예상할 수 있다. 실제 배수는 기계마다 다르니 세 값에서 직접 재어 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
버퍼에 전이가 $N$ 개 있고, 업데이트마다 크기 $B$ 인 배치를 뽑으며, 환경 한 스텝마다 업데이트를 한 번 한다고 하자. 방금 들어온 전이가 처음으로 배치에 뽑히기까지 기다리는 업데이트 횟수의 기댓값을 `EfficientReplayBuffer` 와 `CombinedReplayBuffer` 에서 각각 구하라. $N$ 이 그동안 변하지 않는다고 본다.

</div>

??? success "풀이"
    `EfficientReplayBuffer` 는 `np.random.randint` 로 인덱스 $B$ 개를 복원 추출한다. 한 번의 업데이트에서 특정 전이가 한 번이라도 뽑힐 확률은 $p = 1-(1-1/N)^B \approx B/N$ 이므로, 처음 뽑히기까지의 횟수는 기하분포를 따르고 기댓값은 $1/p \approx N/B$ 이다. $N=10{,}000$, $B=64$ 이면 약 156번의 업데이트를 기다린다. `CombinedReplayBuffer` 는 `last_idx` 를 배치에 늘 넣으므로 바로 다음 업데이트에서 쓰인다(기댓값 1). Zhang & Sutton 이 CER 을 제안한 까닭이 이것이다. 버퍼가 클수록 새 경험이 트레인에 늦게 반영되는데, CER 은 버퍼 크기와 상관없이 이 지연을 없앤다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`FrameStackReplayBuffer` 의 `_get_stack` 과 `sample` 을 꼼꼼히 읽고 두 가지를 따져 보라. (가) 스택이 에피소드 경계를 넘을 때 빈자리를 어느 프레임으로 채우는가? 주석이 말하는 의도와 맞는가? (나) `valid = np.arange(self.n_stack, self.size)` 는 버퍼가 아직 다 차지 않았을 때와 한 바퀴 돌아 덮어쓰기 시작한 뒤에 각각 어떤 잘못된 표본을 허용하는가?

</div>

??? success "풀이"
    (가) `dones[frame_idx]` 가 참인 프레임은 **이전 에피소드의 마지막 프레임**이다. 그런데 코드는 바로 그 `frame_idx` 를 남은 자리에 채워 넣는다. 그래서 새 에피소드 초반의 스택에 이전 에피소드의 마지막 화면이 섞인다. 주석의 의도(「쓸 수 있는 첫 프레임으로 채움」)대로 하려면 그 프레임을 넣지 말고 한 칸 뒤, 즉 현재 에피소드의 첫 프레임 `(frame_idx + 1) % capacity` 로 채워야 한다.

    (나) 다음 상태는 `_get_stack(i + 1)` 로 만든다. 버퍼가 다 차지 않았을 때 `i = size - 1` 이 뽑히면 `i + 1` 자리는 아직 쓰이지 않은 0 프레임이라 다음 상태가 검은 화면이 된다. 버퍼가 한 바퀴 돈 뒤에는 `ptr` 앞뒤가 가장 새 데이터와 가장 오래된 데이터의 경계다. `ptr - 1` 을 뽑으면 다음 상태가 가장 오래된 프레임에서 오고, `ptr` 바로 뒤 몇 칸을 뽑으면 스택이 가장 새 프레임까지 거슬러 올라간다. `valid` 는 `ptr` 을 전혀 보지 않으므로 이런 자리를 걸러 내지 못한다. 고치려면 `ptr - 1` 과 `ptr` 부터 `n_stack - 1` 칸을 후보에서 빼고, 버퍼가 다 차지 않았을 때는 `size - 1` 까지만이 아니라 `size - 2` 까지만 뽑아야 한다.

## 정리하며

**다룬 것** — 경험 리플레이

이 구현은 덱 기반, NumPy 배열 기반, 결합 경험 리플레이, 프레임 스택이라는 네 가지 리플레이 버퍼를 같은 인터페이스로 구현했다.

핵심 클래스는 `BasicReplayBuffer`, `EfficientReplayBuffer`, `CombinedReplayBuffer`, `FrameStackReplayBuffer` 이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
