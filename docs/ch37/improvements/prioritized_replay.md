# 우선순위 경험 리플레이

우선순위 경험 리플레이(PER)는 DQN 개선의 중요한 개념이다. 일반 리플레이는 버퍼의 모든 전이를 같은 확률로 뽑지만, PER 은 TD 오차가 큰 전이 — 네트워크가 아직 잘 못 맞히는, 그래서 배울 것이 많은 전이 — 를 더 자주 뽑는다. 전이 $i$ 의 우선순위를 $p_i = (|\delta_i| + \epsilon)^\alpha$ 로 두고 확률 $P(i) = p_i / \sum_k p_k$ 로 뽑는다. 이렇게 뽑으면 데이터 분포가 바뀌어 업데이트가 치우치므로, 중요도 샘플링 가중치 $w_i = (N \cdot P(i))^{-\beta}$ 로 그 치우침을 바로잡는다. 이 쪽은 이 버퍼를 $O(\log N)$ 에 뽑을 수 있게 해 주는 합 트리(sum tree)와 함께 구현한다.

## 1. 코드

```python
"""
35.2.3 우선순위 겪음 되돌려 보기
======================================

중요도 뽑기를 하는 합 나무 바탕 우선순위 되돌려 보기 버퍼.
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
    """O(log n) 우선순위 뽑기를 위한 두 클래스 합 나무.
    
    잎은 우선순위를, 안쪽 마디는 부분합을 담는다.
    온 우선순위는 뿌리 마디의 값이다.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """잎이 바뀐 뒤 어버이 절를 고친다."""
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
# 우선순위 되돌려 보기 버퍼
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
        """우선순위로 배치를 뽑는다. (s, a, r, ns, d, weights, tree_indices)를 돌려준다."""
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
    demo_prioritized_replay()
```

## 2. 논의

이 구현은 두 클래스로 되어 있다. `SumTree` 는 잎에 우선순위를, 안쪽 노드에 두 자식의 합을 담는 완전 이진 트리다. 뿌리 값이 전체 합이고, $[0, \text{total})$ 에서 고른 값 하나로 뿌리에서 잎까지 내려가면 우선순위에 비례하는 확률로 잎 하나가 뽑힌다. 갱신과 뽑기가 모두 트리 높이 $O(\log N)$ 만큼 걸린다. `PrioritizedReplayBuffer` 는 이 트리 위에 전이 데이터를 저장하고, 전체 합을 배치 크기만큼의 구간으로 나눠 구간마다 하나씩 뽑는 층화 샘플링과 $\beta$ 를 0.4 에서 1 로 늘려 가는 중요도 가중치를 더한다.

데모 함수는 두 부분이다. 먼저 용량 8 인 합 트리에 우선순위 다섯 개를 넣고 10,000 번 뽑아, 뽑힌 비율이 우선순위 비율과 맞는지 확인한다. 다음으로 CartPole 에서 PER 로 DQN 을 200 에피소드 트레인하며 50 에피소드마다 평균 리워드와 $\beta$ 를 찍고, 끝에서 한 배치의 중요도 가중치 범위·평균·표준편차를 본다. 트레인 부분에는 시드가 고정되어 있지 않아 돌릴 때마다 수가 달라지며, PER 이 없는 경우와 견주지도 않으므로 이 데모로 PER 의 이득을 말할 수는 없다.

`push` 에는 고칠 곳이 하나 있다. TD 오차를 모를 때 새 전이에 「가장 큰 우선순위」를 주려는 의도인데, 트리에 저장된 값은 이미 $\alpha$ 제곱을 한 $p_i$ 이므로 `max_priority ** self.alpha` 는 $\alpha$ 를 한 번 더 씌운 값이 된다. 원 논문처럼 하려면 `self.tree.max_priority` 를 그대로 써야 한다(연습문제 2).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 `SumTree(8)` 에 우선순위 `[3, 1, 5, 2, 4]` 를 넣었을 때 배열 `tree` 의 15칸을 모두 적어라. `tree.sample(9.0)` 은 어느 데이터 인덱스를 돌려주는가? 뿌리에서 잎까지 내려가는 경로를 적어라.

</div>

??? success "풀이"
    잎은 인덱스 7~14 이고 `[3, 1, 5, 2, 4, 0, 0, 0]` 이다. 그 위 노드는 인덱스 3 = 3+1 = 4, 인덱스 4 = 5+2 = 7, 인덱스 5 = 4+0 = 4, 인덱스 6 = 0 이고, 인덱스 1 = 4+7 = 11, 인덱스 2 = 4+0 = 4, 뿌리(인덱스 0) = 15 다. 따라서 `tree = [15, 11, 4, 4, 7, 4, 0, 3, 1, 5, 2, 4, 0, 0, 0]` 이다. `_retrieve(0, 9)`: 왼쪽 자식(인덱스 1)의 값 11 이 9 이상이므로 왼쪽으로 간다. 인덱스 1 의 왼쪽 자식(인덱스 3)은 4 < 9 라서 오른쪽(인덱스 4)으로 가며 값을 $9 - 4 = 5$ 로 줄인다. 인덱스 4 의 왼쪽 자식(인덱스 9)은 5 이고 $5 \le 5$ 라서 왼쪽으로 간다. 인덱스 9 는 잎이므로 데이터 인덱스 $9 - 8 + 1 = 2$, 우선순위 5 인 전이가 뽑힌다. 누적 구간으로 보면 데이터 0 은 $[0, 3]$, 1 은 $(3, 4]$, 2 는 $(4, 9]$ 이므로 맞다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`push` 는 TD 오차가 없을 때 `self.tree.max_priority ** self.alpha` 를 우선순위로 쓴다. 저장된 우선순위가 이미 $(|\delta|+\epsilon)^\alpha$ 라는 점을 생각하면, 이 식은 저장된 최댓값이 1 보다 클 때와 작을 때 각각 어떤 값을 주는가? 「새 전이를 적어도 한 번은 뽑히게 한다」는 의도와 맞는가?

</div>

??? success "풀이"
    저장된 최댓값을 $m$ 이라 하면 새 전이의 우선순위는 $m^\alpha$ ($\alpha = 0.6$) 이다. $m > 1$ 이면 $m^\alpha < m$ 이라 새 전이가 버퍼에서 가장 높은 우선순위를 받지 못한다. $m < 1$ 이면 $m^\alpha > m$ 이라 오히려 최댓값보다 커지고, 그 값이 다시 최댓값이 되어 다음 전이에서 또 $\alpha$ 제곱이 씌워진다. 그러면 TD 오차와 상관없이 우선순위가 1 쪽으로 끌려간다. 어느 쪽이든 「지금까지의 최대 우선순위를 그대로 주어 새 전이가 곧 뽑히게 한다」는 원 논문의 의도와 어긋난다. `priority = self.tree.max_priority if self.tree.size > 0 else 1.0` 으로 고치면 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
중요도 가중치 $w_i = (N \cdot P(i))^{-\beta} / \max_j w_j$ 에서 $\beta = 1$ 이면 업데이트의 기댓값이 균일 샘플링과 같아짐을 보여라. 왜 처음부터 $\beta = 1$ 로 두지 않고 0.4 에서 늘려 가는가? 데모의 `beta_steps=100000` 으로 200 에피소드 동안 $\beta$ 가 1 에 닿을 수 있는지도 따져 보라.

</div>

??? success "풀이"
    정규화 상수 $\max_j w_j$ 를 빼고 보면, 전이 $i$ 의 로스 그래디언트를 $g_i$ 라 할 때 $\mathbb{E}_{i \sim P}[w_i g_i] = \sum_i P(i)\,(N P(i))^{-1} g_i = \frac{1}{N}\sum_i g_i$ 로 균일 샘플링의 기댓값과 같다. 상수로 나누는 것은 러닝레이트를 바꾸는 것과 같아 방향을 바꾸지 않는다. 트레인 초기에는 타깃 자체가 크게 움직여 업데이트가 어차피 치우쳐 있으므로, 완전한 보정보다 TD 오차가 큰 전이를 빨리 배우는 쪽이 이득이다. 수렴에 가까워질수록 치우침 없는 추정이 중요해지므로 $\beta$ 를 1 로 늘린다. 데모에서 `step_count` 는 `sample` 을 부를 때마다 1 씩 오르고, 버퍼가 500 개가 된 뒤 환경 스텝마다 한 번 부른다. CartPole 에피소드는 최대 500 스텝이라 200 에피소드는 많아야 100,000 스텝이고, 처음 500 스텝은 샘플링하지 않는다. 따라서 `step_count` 는 100,000 에 닿지 못하고 $\beta$ 는 끝까지 1 보다 작다. 데모의 마지막 줄 `Current β` 가 그 값을 보여 준다.

## 정리하며

**다룬 것** — 우선순위 경험 리플레이

이 구현은 $O(\log N)$ 샘플링을 해 주는 `SumTree` 와, 그 위에 층화 샘플링과 중요도 가중치를 더한 `PrioritizedReplayBuffer` 를 중심에 둔다.

핵심 클래스는 `SumTree`, `PrioritizedReplayBuffer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
