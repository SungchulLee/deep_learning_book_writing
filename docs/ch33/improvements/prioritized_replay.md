# 앞섬 되돌려 보기

앞섬 되돌려 보기는 DQN 좋게 하기의 중요한 개념이다. 중요도 뽑기를 곁들인 합 나무 바탕 앞섬 되돌려 보기 버퍼를 다룬다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
33.2.3 우선순위 겪음 되돌려 보기
======================================

중요도 뽑기를 하는 합 나무 바탕 우선순위 되돌려 보기 담개.
"""

import torch
import numpy as np
from typing import Tuple, List

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 합 나무
# ---------------------------------------------------------------------------

class SumTree:
    """O(log n) 우선순위 뽑기를 위한 두 갈래 합 나무.
    
    잎은 우선순위를, 안쪽 마디는 부분합을 담는다.
    온 우선순위는 뿌리 마디의 값이다.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """잎이 바뀐 뒤 어버이 마디를 고친다."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """주어진 쌓인 값에 맞는 잎 마디를 찾는다."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def update(self, idx: int, priority: float):
        """나무 번호 idx의 잎의 우선순위를 고친다."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """주어진 우선순위로 새 칸을 더한다. 자료 번호를 돌려준다."""
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def sample(self, value: float) -> Tuple[int, float, int]:
        """쌓인 값으로 잎을 뽑는다. (tree_idx, priority, data_idx)를 돌려준다."""
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
# 우선순위 되돌려 보기 담개
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """합 나무 뽑기와 중요도 뽑기 바로잡기를 하는 우선순위 겪음 되돌려 보기."""

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

        # 자료 갈무리
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
        """옮김을 더한다. td_error가 없으면 가장 큰 우선순위를 쓴다."""
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
        """우선순위로 묶음을 뽑는다. (s, a, r, ns, d, weights, tree_indices)를 돌려준다."""
        self.step_count += 1
        tree_indices = []
        data_indices = []

        # 켜 나눠 뽑기: 전체 우선순위를 같은 토막으로 나눔
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx, priority, data_idx = self.tree.sample(value)
            tree_indices.append(tree_idx)
            data_indices.append(data_idx)

        data_indices = np.array(data_indices)

        # 중요도 뽑기 무게
        beta = self.beta
        min_prob = self.tree.min_priority / (self.tree.total + 1e-8)
        max_weight = (min_prob * self.tree.size + 1e-8) ** (-beta)

        priorities = np.array([self.tree.tree[ti] for ti in tree_indices])
        probs = priorities / (self.tree.total + 1e-8)
        weights = (probs * self.tree.size + 1e-8) ** (-beta)
        weights = weights / (max_weight + 1e-8)  # 정규화

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
        """새 때 차이 어긋남을 셈한 뒤 우선순위를 고친다."""
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_prioritized_replay():
    import gymnasium as gym
    import torch.nn as nn

    print("=" * 60)
    print("Prioritized Experience Replay Demo")
    print("=" * 60)

    # --- 합 나무 기본 ---
    print("\n--- Sum Tree ---")
    tree = SumTree(8)
    priorities = [3.0, 1.0, 5.0, 2.0, 4.0]
    for p in priorities:
        tree.add(p)
    print(f"  Priorities: {priorities}")
    print(f"  Total: {tree.total}")
    print(f"  Max: {tree.max_priority}")

    # 뽑기 분포
    counts = np.zeros(5)
    for _ in range(10000):
        v = np.random.uniform(0, tree.total)
        _, _, di = tree.sample(v)
        counts[di] += 1
    counts /= counts.sum()
    expected = np.array(priorities) / sum(priorities)
    print(f"  Expected probs: {expected.round(3)}")
    print(f"  Sampled probs:  {counts.round(3)}")

    # --- CartPole에서 우선순위 되돌려 보기 ---
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

    # --- 중요도 뽑기 무게 살피기 ---
    print("\n--- IS Weight Analysis ---")
    if len(per_buf) >= 64:
        _, _, _, _, _, weights, _ = per_buf.sample(64)
        print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Weight mean: {weights.mean():.4f}")
        print(f"  Weight std: {weights.std():.4f}")
        print(f"  Current β: {per_buf.beta:.4f}")

    print("\nPrioritized replay demo complete!")


if __name__ == "__main__":
    demo_prioritized_replay()```

## 2. 논의

이 짜기는 앞섬 되돌려 보기의 핵심 논리를 감싼 `SumTree`, `PrioritizedReplayBuffer` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
    이 얼개 고르기는 DQN 좋게 하기에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 앞섬 되돌려 보기

이 짜기는 앞섬 되돌려 보기의 핵심 논리를 감싼 `SumTree`, `PrioritizedReplayBuffer` 갈래를 한가운데 둔다.

고갱이 갈래는 `SumTree`, `PrioritizedReplayBuffer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
