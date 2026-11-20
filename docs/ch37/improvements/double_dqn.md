# Double DQN

Double DQN은 액션 선택과 액션 평가를 떼어 놓아 일반 DQN에 원래 있는 구조적 과대추정 바이어스를 다룬다. 일반 DQN에서는 같은 네트워크가 다음 최선 액션을 고르고 평가까지 해 가치 추정이 낙관 쪽으로 치우쳐 학습을 해칠 수 있다. Double DQN은 온라인 네트워크로 액션을 고르고 타깃 네트워크로 평가해 더 맞는 Q 값 추정을 내며 보통 같거나 나은 성능을 더 안정된 학습으로 이룬다.

## 1. 코드

```python
"""
35.2.1 겹 DQN
==================

보통 DQN과 견주는 겹 DQN 짜기로
넘겨 잡는 치우침이 줄어듦을 보인다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Dict
import random

# ========================================================================
# 메인
# ========================================================================

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
# 겹 DQN과 보통 DQN의 타깃 셈
# ---------------------------------------------------------------------------

def dqn_target(online_net: nn.Module, target_net: nn.Module,
               next_states: torch.Tensor, rewards: torch.Tensor,
               dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """여느 DQN 타깃: y = r + γ max_a' Q_target(s', a')"""
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        return rewards + (1 - dones) * gamma * next_q


def double_dqn_target(online_net: nn.Module, target_net: nn.Module,
                      next_states: torch.Tensor, rewards: torch.Tensor,
                      dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """이중 DQN 타깃: y = r + γ Q_target(s', argmax_a' Q_online(s', a'))"""
    with torch.no_grad():
        # 온라인 네트워크가 액션을 고름
        best_actions = online_net(next_states).argmax(dim=1)
        # 타깃 네트워크가 값을 매김
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        return rewards + (1 - dones) * gamma * next_q


# ---------------------------------------------------------------------------
# 겹 깃발을 지닌 통합 DQN 에이전트
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """보통 DQN과 겹 DQN을 모두 받치는 DQN 에이전트."""

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

        # Q 값 적기
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
# 넘겨 잡기 살피기
# ---------------------------------------------------------------------------

def measure_overestimation(env_name='CartPole-v1', episodes=200):
    """DQN과 겹 DQN의 Q 값 어림을 견준다."""
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
# 시연
# ---------------------------------------------------------------------------

def demo_double_dqn():
    print("=" * 60)
    print("Double DQN Demo")
    print("=" * 60)

    results = measure_overestimation('CartPole-v1', episodes=250)

    # 비교
    print("\n--- Summary ---")
    for name, data in results.items():
        r = data['rewards']
        q = data['q_values']
        print(f"  {name}:")
        print(f"    Final avg reward: {np.mean(r[-50:]):.1f}")
        print(f"    Q-value range: [{min(q):.2f}, {max(q):.2f}]")
        print(f"    Q-value mean: {np.mean(q):.2f}")

    # 차이가 Q 값의 잣대에 있음을 보임
    if all(r['q_values'] for r in results.values()):
        dqn_q = np.mean(results['Standard DQN']['q_values'][-200:])
        ddqn_q = np.mean(results['Double DQN']['q_values'][-200:])
        print(f"\n  Overestimation reduction: {dqn_q - ddqn_q:.2f} "
              f"({(1 - ddqn_q/dqn_q)*100:.1f}% lower Q-values)")

    print("\nDouble DQN demo complete!")


if __name__ == "__main__":
    demo_double_dqn()
```

## 2. 논의

이 구현은 Double DQN의 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크에 실제로 적용하는 모습을 보인다. 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 DQN 개선 연구에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — Double DQN

이 구현은 Double DQN의 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent` 클래스를 중심에 둔다.

핵심 클래스는 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
