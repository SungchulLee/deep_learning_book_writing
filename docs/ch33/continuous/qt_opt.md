# QT-Opt

QT-Opt은 단순한 argmax 대신 교차 엔트로피 방법으로 움직임을 가장 좋게 하여 Q 배우기를 이어진 움직임 자리로 넓힌다. 이어진 움직임 자리는 모두 늘어놓을 수 없으므로 QT-Opt은 가우스 분포에서 움직임 후보를 되풀이해 뽑고 Q 그물로 따진 뒤 잘한 표본에 분포를 다시 맞춘다. 이 길은 큰 규모의 로봇 집기에 성공으로 쓰여 값 바탕 방법이 이어진 다스리기에서 방침 기울기 방법과 겨룰 수 있음을 보였다.

## 코드

```python
"""
33.4.2 QT-Opt
===============

어긋 엔트로피 방법(CEM)을 쓴 이어진 움직임의 Q 배움.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple
import random

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# Q 그물(상태-움직임 들임)
# ---------------------------------------------------------------------------

class ContinuousQNetwork(nn.Module):
    """(상태, 움직임)을 들임으로 받는 Q 그물 → 스칼라 Q 값."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 움직임 다듬기를 위한 어긋 엔트로피 방법
# ---------------------------------------------------------------------------

class CEM:
    """이어진 움직임 다듬기를 위한 어긋 엔트로피 방법."""

    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 n_samples: int = 64, n_elite: int = 6, n_iterations: int = 3):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations

    def optimize(self, q_network: nn.Module, state: torch.Tensor) -> np.ndarray:
        """어긋 엔트로피 방법으로 Q(state, action)을 가장 크게 하는 움직임을 찾는다.
        
        인수:
            q_network: Q 그물
            state: 상태 텐서 하나, 꼴 (state_dim,) 또는 (1, state_dim)
            
        반환값:
            찾아낸 가장 좋은 움직임, 꼴 (action_dim,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 분포 첫값 매기기
        mu = np.zeros(self.action_dim)
        sigma = np.ones(self.action_dim)

        best_action = mu.copy()
        best_q = -float('inf')

        for _ in range(self.n_iterations):
            # 가우스에서 움직임 뽑기
            actions = np.random.normal(mu, sigma, size=(self.n_samples, self.action_dim))
            actions = np.clip(actions, self.action_low, self.action_high)

            # Q 값 매기기
            actions_t = torch.FloatTensor(actions)
            states_t = state.expand(self.n_samples, -1)
            with torch.no_grad():
                q_values = q_network(states_t, actions_t).numpy()

            # 뛰어난 뽑음 고르기
            elite_idx = np.argsort(q_values)[-self.n_elite:]
            elite_actions = actions[elite_idx]

            # 통틀어 가장 좋은 것 좇기
            if q_values[elite_idx[-1]] > best_q:
                best_q = q_values[elite_idx[-1]]
                best_action = actions[elite_idx[-1]].copy()

            # 분포 다시 맞추기
            mu = elite_actions.mean(axis=0)
            sigma = elite_actions.std(axis=0) + 1e-6

        return best_action

    def optimize_batch(self, q_network: nn.Module, states: torch.Tensor) -> torch.Tensor:
        """상태 묶음의 움직임을 다듬는다."""
        actions = []
        for i in range(states.shape[0]):
            a = self.optimize(q_network, states[i])
            actions.append(a)
        return torch.FloatTensor(np.array(actions))


# ---------------------------------------------------------------------------
# 되돌려 보기 담개
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd, ad):
        self.cap=cap;self.sz=0;self.p=0
        self.s=np.zeros((cap,sd),np.float32);self.a=np.zeros((cap,ad),np.float32)
        self.r=np.zeros(cap,np.float32);self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r;self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.FloatTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


# ---------------------------------------------------------------------------
# QT-Opt 부림꾼
# ---------------------------------------------------------------------------

class QTOptAgent:
    """QT-Opt: 이어진 움직임을 위한 Q 배움과 어긋 엔트로피 방법."""

    def __init__(self, state_dim, action_dim, action_low, action_high,
                 lr=1e-3, gamma=0.99, tau=0.005, batch_size=128,
                 buf_cap=100000, cem_samples=64, cem_elite=6, cem_iters=3,
                 noise_std=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_low = action_low
        self.action_high = action_high
        self.noise_std = noise_std

        self.online = ContinuousQNetwork(state_dim, action_dim)
        self.target = ContinuousQNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buf_cap, state_dim, action_dim)

        self.cem = CEM(action_dim, action_low, action_high,
                       cem_samples, cem_elite, cem_iters)

    def act(self, state, training=True):
        s = torch.FloatTensor(state)
        if training:
            # 살펴보기 시끄러움을 넣어 어긋 엔트로피 방법 쓰기
            action = self.cem.optimize(self.online, s)
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, self.action_low, self.action_high)
        else:
            action = self.cem.optimize(self.online, s)
        return action

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < self.batch_size:
            return 0.0
        s, a, r, ns, d = self.buf.sample(self.batch_size)

        # 지금 Q
        q = self.online(s, a)

        # 과녁: r + γ max_a' Q_target(s', a')
        # 다음 상태마다 max_a'을 찾으려 어긋 엔트로피 방법 쓰기
        with torch.no_grad():
            best_next_a = self.cem.optimize_batch(self.target, ns)
            next_q = self.target(ns, best_next_a)
            targets = r + (1 - d) * self.gamma * next_q

        loss = nn.functional.mse_loss(q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)
        return loss.item()


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_qt_opt():
    print("=" * 60)
    print("QT-Opt Demo")
    print("=" * 60)

    # --- 어긋 엔트로피 방법 다듬기 ---
    print("\n--- CEM Action Optimization ---")
    sd, ad = 3, 1
    q_net = ContinuousQNetwork(sd, ad)
    cem = CEM(ad, np.array([-2.0]), np.array([2.0]),
              n_samples=64, n_elite=6, n_iterations=3)

    state = torch.randn(1, sd)
    best_a = cem.optimize(q_net, state)
    print(f"  State: {state.numpy().round(3)}")
    print(f"  Best action (CEM): {best_a.round(3)}")

    # 격자 찾기로 확인
    grid = torch.linspace(-2, 2, 100).unsqueeze(1)
    with torch.no_grad():
        grid_q = q_net(state.expand(100, -1), grid).numpy()
    grid_best = grid[np.argmax(grid_q)].item()
    print(f"  Best action (grid): {grid_best:.3f}")
    print(f"  CEM vs grid gap: {abs(best_a[0] - grid_best):.4f}")

    # --- Pendulum에서 익히기 ---
    print("\n--- QT-Opt Training on Pendulum-v1 ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    env = gym.make('Pendulum-v1')
    agent = QTOptAgent(3, 1, np.array([-2.0]), np.array([2.0]),
                       lr=1e-3, noise_std=0.3,
                       cem_samples=32, cem_elite=4, cem_iters=2)
    rewards = []

    for ep in range(150):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, done)
            agent.update()
            s = ns; total += r
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards[-50:]):.1f}")

    env.close()
    print(f"\n  Final avg(50): {np.mean(rewards[-50:]):.1f}")
    print("\nQT-Opt demo complete!")


if __name__ == "__main__":
    demo_qt_opt()```

## 논의

이 짜기는 QT-Opt의 핵심 논리를 감싼 `ContinuousQNetwork`, `CEM`, `ReplayBuffer` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
    이 얼개 고르기는 이어진 움직임 자리에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.
