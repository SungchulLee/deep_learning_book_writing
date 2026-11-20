# 주문 실행

**최적 주문 실행**은 큰 주문을 정해진 기간 안에 여러 조각으로 나누어 처리하면서 비용을 가장 작게 하는 문제다. 한꺼번에 많이 팔면 **시장 충격**(market impact) — 내 주문 때문에 가격이 불리하게 움직이는 것 — 이 커지고, 너무 천천히 팔면 남은 재고가 가격 변동에 오래 노출된다. 이 쪽은 시장 충격을 흉내 낸 시뮬레이션 환경에서 DQN 에이전트를 학습시키고, 기간마다 같은 양을 파는 **TWAP**(time-weighted average price) 전략과 견준다.

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

구현은 세 부분이다. `OrderExecutionEnv` 는 1만 주를 20 기간에 걸쳐 파는 환경이고, 액션은 **남은 재고**의 0%, 10%, …, 100% 를 파는 11가지다. 일시적 충격 비용은 `eta * participation * price * n_shares` 로, 참여율(거래량 대비 내 주문의 비율)이 주문량에 비례하므로 **한 기간에 파는 양의 제곱**에 비례한다. 그래서 나누어 팔수록 유리하다. `QNet` 과 `OrderExecDQN` 은 리플레이 버퍼와 타깃 네트워크를 쓰는 DQN 이며, 타깃을 계산할 때 온라인 네트워크로 액션을 고르고 타깃 네트워크로 그 값을 매기는 Double DQN 방식을 쓴다.

출력은 정직하게 읽어야 한다. 학습 중 100 에피소드 평균 리턴은 200 에피소드 무렵 $-527.30$ 으로 가장 좋다가 ε 이 줄어드는 동안 $-909.86$ 까지 나빠진다. 탐색을 줄일수록 나빠진 셈이다. 평가에서 DQN 은 $-1065.51$ 로 TWAP 의 $-1026.02$ 보다 39.49 낮고, 그 차이는 두 표준편차(268, 334)보다 훨씬 작다. **이 실행에서 DQN 은 베이스라인을 넘지 못했다.**

베이스라인도 이름대로 동작하지 않는다는 점에 주의하라. `twap_policy` 는 첫 기간에 액션을 `round(500 / 10000 * 10) = round(0.5)` 로 잡는데, 파이썬의 `round` 는 0.5 를 짝수 쪽인 0 으로 반올림한다. 그러면 아무것도 팔지 않아 다음 기간에도 같은 계산이 되풀이되고, 결국 마지막 기간에 1만 주를 한꺼번에 강제 처분한다. 이때 참여율이 약 10 이 되어 리턴이 약 $-1000$ 이 되는데, 출력의 $-1026.02 \pm 334.01$ 이 바로 그 크기다. 출력의 「TWAP」은 실제로는 「끝까지 기다렸다 한꺼번에 팔기」이고, DQN 의 평가 리턴이 그와 비슷하다는 것은 DQN 도 대부분의 재고를 끝에 처분하고 있을 가능성을 시사한다.

이 환경은 교육용으로 크게 단순화되어 있다. 실제 주문 실행에서는 호가창의 깊이, 체결 확률, 가격의 자기상관을 함께 모델링한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`twap_policy` 가 기간마다 정확히 500주씩 팔도록 고쳐라. 액션이 「남은 재고의 10% 단위」라는 제약 때문에 정확한 TWAP 을 11가지 액션으로 표현할 수 있는가? 표현할 수 없다면 어떻게 우회하겠는가?

</div>

??? success "풀이"
    $k$ 번째 기간(0부터)에 남은 재고는 $(20-k)\times 500$ 주이므로 500주를 팔려면 비율 $1/(20-k)$ 이 필요하다. 이것이 0.1 의 배수가 되는 것은 $k=10$ (1/10) 같은 몇 기간뿐이라, 11가지 액션으로는 TWAP 을 정확히 표현할 수 없다. 베이스라인은 액션을 거치지 않고 `n_shares` 를 직접 500 으로 정하는 별도 경로로 실행하거나, 액션을 「처음 주문량 $Q$ 의 몇 %」로 다시 정의한다. 가격 변동을 무시하고 거래량을 평균값으로 두면 참여율이 기간마다 약 0.5 라 기간당 리턴이 약 $-2.5$, 20 기간 합이 약 $-50$ 이고 스프레드와 재고 벌점을 더해도 $-60$ 안팎이다. 한꺼번에 처분하는 경우의 약 $-1000$ 보다 열 배 넘게 낫다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
일시적 충격 비용이 한 기간에 파는 양 $n$ 의 제곱에 비례한다고 하자($c\,n^2$). 가격 변동과 재고 벌점을 무시하면, $Q$ 주를 $T$ 기간에 나누어 팔 때 총비용을 가장 작게 하는 분할은 무엇인가? 재고 벌점(가격 위험)을 더하면 그 답이 어느 쪽으로 바뀌는가?

</div>

??? success "풀이"
    $\sum_t n_t = Q$ 제약에서 $\sum_t c\,n_t^2$ 을 최소화하면 라그랑주 조건 $2c\,n_t=\lambda$ 에서 모든 $n_t$ 가 같아진다. 즉 $n_t=Q/T$ 인 TWAP 이 최적이고 총비용은 $cQ^2/T$ 다. 한꺼번에 팔 때의 $cQ^2$ 보다 $T$ 배 작다. 재고를 오래 들고 있는 데 벌점이나 가격 위험이 붙으면 앞쪽에서 더 많이 파는 것이 유리해져 최적 분할이 앞으로 쏠린다. 이것이 알름그렌-크리스 모델의 결론으로, 위험 회피가 클수록 처분 곡선이 가파르게 앞쪽으로 휜다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
출력에서 학습 중 평균 리턴은 200 에피소드 무렵 가장 좋고 그 뒤로 나빠진다. 원인을 가려내는 실험을 설계하라. 적어도 (가) 평가 에피소드마다 기간별 판매량을 기록해 DQN 이 무엇을 배웠는지 보기, (나) 리워드의 크기 확인하기 두 가지를 포함하라.

</div>

??? success "풀이"
    (가) `execution_log` 의 `shares` 를 기간별로 평균하면 DQN 이 재고를 끝까지 들고 있다가 마지막에 처분하는지 바로 보인다. (나) 리워드를 보면 한꺼번에 처분하는 마지막 기간에 $-1000$ 규모의 값이 한 번 나오고, 나머지 기간의 값은 $-0.1$ 규모다. 이 큰 값은 마지막 기간의 강제 처분에서 나오는데, 그 기간에는 어떤 액션을 고르든 같은 일이 일어나므로 Q 값의 목표가 액션과 상관없이 크게 흔들린다. 또 상태의 `volume` 은 다음 기간의 체결에 쓰일 거래량이지만 `vol` 은 순수한 잡음이라 학습을 방해한다. 리워드를 스케일링하거나 클리핑하고 `vol` 을 상태에서 빼 보며 학습이 안정되는지 확인한다. 원인마다 시드를 바꾸어 여러 번 돌려 차이가 시드에 따른 흔들림보다 큰지 본다.

## 정리하며

**다룬 것** — 주문 실행

시장 충격 비용이 주문량의 제곱에 비례하면 나누어 팔수록 유리하고, 가격 위험까지 따지면 앞쪽에 더 많이 판다. 이 쪽의 DQN 은 그 구조를 배우지 못했고 베이스라인도 반올림 때문에 한꺼번에 처분하고 있었다 — 금융 강화학습에서는 에이전트보다 먼저 베이스라인과 리워드의 크기를 확인해야 한다.

핵심 클래스는 `OrderExecutionEnv`, `QNet`, `OrderExecDQN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
