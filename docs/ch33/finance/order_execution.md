# 주문 실행

주문 실행은 힘 북돋우는 배움의 금융 쓰임새에서 중요한 개념이다. 흉내 낸 시장 충격과 함께 하는 DQN 바탕 가장 좋은 주문 실행을 다룬다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
33.7.1 주문 처리
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

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 주문 처리 둘레
# ---------------------------------------------------------------------------

class OrderExecutionEnv:
    """저자 충격이 있는 흉내 주문 처리 둘레.
    
    부림꾼은 T기간에 걸쳐 주식 Q주를 처분해야 한다.
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
        # 움직임을 남은 재고의 몫으로 바꿈
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

        # 보상: 음의 처리 비용(짜기 모자람)
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
# TWAP과 VWAP 밑금
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
# 주문 처리를 위한 DQN 부림꾼
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

    # TWAP 밑금
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
    demo_order_execution()```

## 논의

이 짜기는 주문 실행의 핵심 논리를 감싼 `OrderExecutionEnv`, `QNet`, `OrderExecDQN` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 핵심 움직임을 도드라지게 하는 만든 자료에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
    이 얼개 고르기는 힘 북돋우는 배움의 금융 쓰임새에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.
