# Double DQN

Double DQN은 액션 선택과 액션 평가를 떼어 놓아 일반 DQN에 구조적으로 들어 있는 과대추정 바이어스를 줄인다. 일반 DQN의 타깃 $r + \gamma\max_{a'} Q_{\theta^-}(s', a')$ 에서는 같은 네트워크가 다음 최선 액션을 고르고 그 값을 평가까지 한다. Q 값 추정에 잡음이 있으면 최댓값을 고르는 일 자체가 우연히 크게 추정된 액션을 고르기 쉬워, 가치 추정이 낙관 쪽으로 치우친다. Double DQN은 온라인 네트워크로 액션을 고르고 타깃 네트워크로 그 액션을 평가해, 한쪽의 우연한 과대추정이 다른 쪽에서 그대로 되풀이되지 않게 한다. 그 결과 Q 값 추정이 더 정확해지고, 보통 같거나 나은 성능을 더 안정된 트레인으로 이룬다.

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

두 방법의 차이는 타깃을 계산하는 두 함수 `dqn_target` 과 `double_dqn_target` 에 다 들어 있다. 앞의 것은 `target_net(next_states).max(dim=1)` 한 줄이고, 뒤의 것은 `online_net` 의 `argmax` 로 액션을 고른 뒤 `target_net` 의 출력에서 그 액션의 값을 `gather` 로 꺼낸다. `DoubleDQNAgent` 는 `double` 플래그 하나로 둘을 갈아 끼우므로, 나머지(네트워크, 버퍼, 후버 로스, 타깃 갱신 주기 200)는 똑같다.

데모 함수는 CartPole 에서 두 에이전트를 같은 시드 42 로 250 에피소드씩 트레인하고, 마지막 50 에피소드 리워드와 업데이트마다 기록한 배치 평균 Q 값을 견준다. 마지막 줄의 「Overestimation reduction」은 두 에이전트의 최근 200 업데이트 평균 Q 값의 차이다. 이 수를 읽을 때 두 가지를 조심해야 한다. 첫째, Q 값이 **낮다**는 것과 **정확하다**는 것은 다르다. 과대추정을 재려면 Q 값을 실제로 받은 할인 리턴과 견주어야 하는데(CartPole 에서 $\gamma = 0.99$ 이면 참값은 100 을 넘을 수 없다), 이 데모는 그 비교를 하지 않는다. 둘째, 두 에이전트는 서로 다른 폴리시를 학습해 서로 다른 상태를 방문하므로, 같은 상태에서의 Q 값을 견준 것이 아니다. 시드 하나로 한 번 돌린 결과이기도 하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
어떤 상태에서 두 액션의 참 Q 값이 모두 0 이고, 추정값이 서로 독립인 잡음 $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, 1)$ 만큼 틀린다고 하자. 일반 DQN 식의 $\max_a \hat Q(s, a)$ 의 기댓값을 구하라. 같은 잡음을 가진 독립된 두 번째 추정 $\hat Q'$ 로 평가하는 Double 식 $\hat Q'(s, \arg\max_a \hat Q(s, a))$ 의 기댓값은 얼마인가?

</div>

??? success "풀이"
    $\epsilon_1 - \epsilon_2 \sim \mathcal{N}(0, 2)$ 이고 $\max(\epsilon_1, \epsilon_2) = \tfrac12(\epsilon_1 + \epsilon_2) + \tfrac12|\epsilon_1 - \epsilon_2|$ 이다. $\mathbb{E}|Z| = \sigma\sqrt{2/\pi}$ 이므로 $\mathbb{E}[\max] = \tfrac12\sqrt{2}\sqrt{2/\pi} = 1/\sqrt{\pi} \approx 0.564$ 다. 참값이 0 인데 평균 0.56 만큼 크게 추정된다. 이것이 $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$ 라는 젠센 부등식이 말하는 과대추정이다. Double 식에서는 고른 액션이 $\hat Q$ 의 잡음에만 달려 있고, 그 액션을 평가하는 $\hat Q'$ 의 잡음은 그것과 독립이므로 기댓값이 정확히 0 이다. 실제 Double DQN 에서는 온라인 네트워크와 타깃 네트워크가 완전히 독립이 아니어서(타깃은 온라인의 옛 복사본이다) 과대추정이 0 이 되지는 않고 줄어들 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`dqn_target` 과 `double_dqn_target` 의 계산량을 견주어라. 업데이트 한 번에 순전파를 몇 번 하는가? 새 파라메터나 메모리가 더 드는가?

</div>

??? success "풀이"
    업데이트 한 번에 일반 DQN 은 현재 상태에 대한 온라인 네트워크 순전파 1번(그래디언트 포함)과 다음 상태에 대한 타깃 네트워크 순전파 1번을 한다. Double DQN 은 여기에 다음 상태에 대한 온라인 네트워크 순전파가 1번 더 붙는다. `torch.no_grad()` 안이라 역전파와 활성값 저장은 필요 없다. 새 파라메터는 없다. 타깃 네트워크는 일반 DQN 에도 이미 있기 때문이다. 그래서 Double DQN 은 코드 몇 줄과 순전파 한 번으로 얻는 개선이라 거의 모든 DQN 변형에 기본으로 들어간다. 다음 상태와 현재 상태를 한 배치로 이어 붙여 온라인 네트워크를 한 번만 부르면 이 비용도 더 줄일 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 쪽의 데모는 두 에이전트의 평균 Q 값만 견준다. 과대추정을 제대로 재는 실험을 설계하고 구현하라. 무엇과 무엇을 견주어야 하며, 시드는 몇 개가 필요한가?

</div>

??? success "풀이"
    과대추정은 **추정값 − 참값**이다. 트레인 중 일정 간격마다 트레인을 멈추고 그리디 폴리시로 에피소드를 여러 번 돌린다. 각 에피소드의 방문 상태 $s_t$ 마다 네트워크가 내는 $\max_a Q(s_t, a)$ 와, 그 뒤로 실제로 받은 할인 리턴 $\sum_k \gamma^k r_{t+k}$ 를 기록해 둘의 평균 차이를 낸다. 이것이 van Hasselt 등이 아타리에서 쓴 방식이다. 이렇게 하면 두 에이전트가 방문하는 상태가 달라도 각자의 과대추정을 공정하게 잴 수 있다. CartPole 은 500 스텝에서 잘리므로 끝에 가까운 상태는 실제 리턴이 짧게 잘려 참값이 작게 나온다. 그러니 에피소드 앞부분의 상태만 쓰거나 잘림을 따로 처리해야 한다. 시드는 적어도 5개를 돌려, 두 방법의 평균 차이가 시드에 따른 표준편차보다 큰지 확인한다.

## 정리하며

**다룬 것** — Double DQN

이 구현은 일반 DQN 과 Double DQN 의 타깃 계산을 두 함수로 나누고, `double` 플래그 하나로 갈아 끼우는 `DoubleDQNAgent` 로 두 방법을 같은 조건에서 견준다.

핵심 클래스는 `ReplayBuffer`, `QNetwork`, `DoubleDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
