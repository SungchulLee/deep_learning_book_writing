# 두 겹 DQN

두 겹 DQN은 움직임 고르기와 움직임 따지기를 떼어 놓아 여느 DQN에 본디 있는 몸에 밴 지나친 어림 치우침을 다룬다. 여느 DQN에서는 같은 그물이 다음 최선 움직임을 고르고 따지기까지 해 값 어림이 낙관으로 치우쳐 배움을 해칠 수 있다. 두 겹 DQN은 온라인 그물로 움직임을 고르고 과녁 그물로 따져 더 맞는 Q 값 어림을 내며 보통 같거나 나은 성능을 더 안정된 익히기로 이룬다.

## 1. 코드

```python
"""
33.2.1 겹 DQN
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
# 겹 DQN과 보통 DQN의 과녁 셈
# ---------------------------------------------------------------------------

def dqn_target(online_net: nn.Module, target_net: nn.Module,
               next_states: torch.Tensor, rewards: torch.Tensor,
               dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """여느 DQN 과녁: y = r + γ max_a' Q_target(s', a')"""
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        return rewards + (1 - dones) * gamma * next_q


def double_dqn_target(online_net: nn.Module, target_net: nn.Module,
                      next_states: torch.Tensor, rewards: torch.Tensor,
                      dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """이중 DQN 과녁: y = r + γ Q_target(s', argmax_a' Q_online(s', a'))"""
    with torch.no_grad():
        # 온라인 그물이 움직임을 고름
        best_actions = online_net(next_states).argmax(dim=1)
        # 과녁 그물이 값을 매김
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        return rewards + (1 - dones) * gamma * next_q


# ---------------------------------------------------------------------------
# 겹 깃발을 지닌 통합 DQN 부림꾼
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """보통 DQN과 겹 DQN을 모두 받치는 DQN 부림꾼."""

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
    demo_double_dqn()```

## 2. 논의

이 짜기는 두 겹 DQN의 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 두 겹 DQN

이 짜기는 두 겹 DQN의 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent` 갈래를 한가운데 둔다.

고갱이 갈래는 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
