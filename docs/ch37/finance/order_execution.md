# 주문 실행

주문 실행은 강화학습의 금융 응용에서 중요한 개념이다. 시뮬레이션한 시장 충격과 함께 하는 DQN 기반 최적 주문 실행을 다룬다. 이 구현은 복잡한 핵심 알고리즘과 자료 구조를 직접 다뤄 보게 하며 이론적 기초와 실제 적용 시 고려할 점을 함께 보여 준다.

## 1. 코드

```python
"""
35.7.1 주문 처리
========================

저자 충격을 흉내 낸 DQN 바탕 가장 좋은 주문 처리.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import random
np.random.seed(42)

# 쪽 전체가 한 프로그램이다. 모듈 수준에서 한 번 박아 두어야
# 아래 수를 다시 얻을 수 있다.
torch.manual_seed(42)

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 주문 처리 둘레
# ---------------------------------------------------------------------------

class OrderExecutionEnv:
    """저자 충격이 있는 흉내 주문 처리 둘레.
    
    에이전트는 T기간에 걸쳐 주식 Q주를 처분해야 한다.
    """

    def __init__(self, total_shares: int = 10000, n_periods: int = 20,
                 initial_price: float = 100.0, volatility: float = 0.02,
                 temp_impact: float = 0.1, perm_impact: float = 0.01,
                 spread: float = 0.01, n_actions: int = 11):
        self.Q = total_shares
        self.T = n_periods
        self.p0 = initial_price
        self.sigma = volatility
        self.eta = temp_impact      # 잠깐 충격 계수
        self.gamma_imp = perm_impact  # 남는 충격 계수
        self.spread = spread
        self.n_actions = n_actions  # 조각낸 몫: 0, 0.1, 0.2, ..., 1.0

        self.state_dim = 5  # (inventory_frac, time_frac, price_return, volume, volatility)
        self.action_dim = n_actions

    def reset(self) -> np.ndarray:
        self.inventory = self.Q
        self.time_step = 0
        self.price = self.p0
        self.arrival_price = self.p0
        self.total_cost = 0.0
        self.execution_log = []
        self.volume_history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        inv_frac = self.inventory / self.Q
        time_frac = self.time_step / self.T
        price_ret = (self.price - self.arrival_price) / self.arrival_price
        volume = np.random.lognormal(0, 0.3)  # 아무 거래량
        vol = self.sigma * np.sqrt(1 + 0.5 * np.random.randn())
        self.volume_history.append(volume)
        return np.array([inv_frac, time_frac, price_ret, volume, vol], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # 액션을 남은 재고의 몫으로 바꿈
        fraction = action / (self.n_actions - 1)

        # 사고팔 주식 수를 정함
        if self.time_step >= self.T - 1:
            # 끝나는 때에 강제로 처분
            n_shares = self.inventory
        else:
            n_shares = int(fraction * self.inventory)
            n_shares = max(0, min(n_shares, self.inventory))

        # 저자 충격
        volume = max(self.volume_history[-1] * self.Q * 0.1, 1)
        participation = n_shares / volume if volume > 0 else 0

        temp_cost = self.eta * participation * self.price * n_shares
        perm_shift = self.gamma_imp * n_shares / self.Q * self.price
        spread_cost = self.spread * self.price * n_shares * 0.5

        # 처리
        exec_price = self.price + temp_cost / max(n_shares, 1) + self.spread * 0.5
        total_exec_cost = temp_cost + spread_cost

        # 값에 남는 충격
        self.price -= perm_shift

        # 아무 값 흐름
        self.price *= np.exp(self.sigma * np.random.randn())

        # 상태 고치기
        self.inventory -= n_shares
        self.time_step += 1
        self.total_cost += total_exec_cost

        # 리워드: 음의 처리 비용(짜기 모자람)
        reward = -total_exec_cost / (self.Q * self.arrival_price) * 1000  # 잣대

        # 재고를 안고 있는 벌점(다급함)
        if self.inventory > 0:
            holding_penalty = -0.1 * (self.inventory / self.Q) ** 2
            reward += holding_penalty

        done = (self.time_step >= self.T) or (self.inventory <= 0)

        self.execution_log.append({
            'step': self.time_step, 'shares': n_shares, 'price': exec_price,
            'cost': total_exec_cost, 'remaining': self.inventory,
        })

        info = {'exec_cost': total_exec_cost, 'shares_traded': n_shares,
                'inventory': self.inventory, 'participation': participation}

        return self._get_state(), reward, done, info


# ---------------------------------------------------------------------------
# TWAP과 VWAP 베이스라인
# ---------------------------------------------------------------------------

def twap_policy(env: OrderExecutionEnv) -> List[Dict]:
    """때 무게 평균 값: 기간마다 같은 주식 수."""
    state = env.reset()
    shares_per_period = env.Q // env.T
    total_reward = 0
    done = False
    while not done:
        n = min(shares_per_period, env.inventory)
        action = int(round(n / max(env.inventory, 1) * (env.n_actions - 1)))
        action = max(0, min(action, env.n_actions - 1))
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward, env.execution_log


# ---------------------------------------------------------------------------
# 주문 처리를 위한 DQN 에이전트
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class OrderExecDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.05, eps_decay=5000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)

        self.buf = deque(maxlen=buf_cap)
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
        self.buf.append((s, a, r, ns, float(d)))

    def update(self):
        if len(self.buf) < 500:
            return
        batch = random.sample(self.buf, self.batch_size)
        s = torch.FloatTensor([t[0] for t in batch])
        a = torch.LongTensor([t[1] for t in batch])
        r = torch.FloatTensor([t[2] for t in batch])
        ns = torch.FloatTensor([t[3] for t in batch])
        d = torch.FloatTensor([t[4] for t in batch])

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            nq = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            tgt = r + (1 - d) * self.gamma * nq
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


def demo_order_execution():
    print("=" * 60)
    print("Order Execution Demo")
    print("=" * 60)

    env = OrderExecutionEnv(total_shares=10000, n_periods=20)
    print(f"\nProblem: Liquidate {env.Q} shares over {env.T} periods")
    print(f"State dim: {env.state_dim}, Actions: {env.action_dim}")

    # TWAP 베이스라인
    print("\n--- TWAP Baseline (10 trials) ---")
    twap_rewards = []
    for _ in range(10):
        r, _ = twap_policy(env)
        twap_rewards.append(r)
    print(f"  TWAP reward: {np.mean(twap_rewards):.2f} ± {np.std(twap_rewards):.2f}")

    # DQN 익히기
    print("\n--- DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    agent = OrderExecDQN(env.state_dim, env.action_dim, lr=1e-3, eps_decay=3000)

    rewards_hist = []
    for ep in range(500):
        s = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.store(s, a, r, ns, done)
            agent.update()
            s = ns; total += r
        rewards_hist.append(total)
        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}: avg100={np.mean(rewards_hist[-100:]):.2f}")

    # DQN 값 매기기
    print("\n--- DQN Evaluation (50 trials) ---")
    dqn_rewards = []
    for _ in range(50):
        s = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s, training=False)
            s, r, done, _ = env.step(a)
            total += r
        dqn_rewards.append(total)
    print(f"  DQN reward: {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
    print(f"  TWAP reward: {np.mean(twap_rewards):.2f} ± {np.std(twap_rewards):.2f}")
    improvement = (np.mean(dqn_rewards) - np.mean(twap_rewards))
    print(f"  Improvement over TWAP: {improvement:+.2f}")

    print("\nOrder execution demo complete!")


if __name__ == "__main__":
    demo_order_execution()
```

**출력:**

```
============================================================
Order Execution Demo
============================================================

Problem: Liquidate 10000 shares over 20 periods
State dim: 5, Actions: 11

--- TWAP Baseline (10 trials) ---
  TWAP reward: -1026.02 ± 334.01

--- DQN Training ---
  Episode 100: avg100=-582.75
  Episode 200: avg100=-527.30
  Episode 300: avg100=-763.37
  Episode 400: avg100=-843.98
  Episode 500: avg100=-909.86

--- DQN Evaluation (50 trials) ---
  DQN reward: -1065.51 ± 268.03
  TWAP reward: -1026.02 ± 334.01
  Improvement over TWAP: -39.49

Order execution demo complete!
```

## 2. 논의

이 구현은 주문 실행의 핵심 논리를 감싼 `OrderExecutionEnv`, `QNet`, `OrderExecDQN` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 핵심 동작을 잘 드러내는 합성 데이터에서 이 구성 요소들의 실제 쓰임을 보인다. 출력을 살펴보면 하이퍼파라미터 선택과 문제 설정에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
    이 구조 선택은 강화학습의 금융 응용에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 주문 실행

이 구현은 주문 실행의 핵심 논리를 감싼 `OrderExecutionEnv`, `QNet`, `OrderExecDQN` 클래스를 중심에 둔다.

핵심 클래스는 `OrderExecutionEnv`, `QNet`, `OrderExecDQN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
