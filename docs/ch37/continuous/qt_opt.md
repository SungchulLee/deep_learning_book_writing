# QT-Opt

QT-Opt은 단순한 argmax 대신 교차 엔트로피 방법으로 액션을 최적화하여 Q 학습을 연속 액션 공간으로 넓힌다. 연속 액션 공간은 모두 늘어놓을 수 없으므로 QT-Opt은 가우시안 분포에서 액션 후보를 반복해 샘플링하고 Q 네트워크로 평가한 뒤 상위 표본에 분포를 다시 맞춘다. 이 길은 대규모 로봇 파지에 성공적으로 쓰여 밸류 기반 방법이 연속 제어에서 폴리시 경사 방법과 경쟁할 수 있음을 보였다.

## 1. 코드

```python
"""
35.4.2 QT-Opt
===============

어긋 엔트로피 방법(CEM)을 쓴 이어진 액션의 Q 러닝.
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
# Q 네트워크(상태-액션 들임)
# ---------------------------------------------------------------------------

class ContinuousQNetwork(nn.Module):
    """(상태, 액션)을 들임으로 받는 Q 네트워크 → 스칼라 Q 값."""

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
# 액션 다듬기를 위한 어긋 엔트로피 방법
# ---------------------------------------------------------------------------

class CEM:
    """이어진 액션 다듬기를 위한 어긋 엔트로피 방법."""

    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 n_samples: int = 64, n_elite: int = 6, n_iterations: int = 3):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations

    def optimize(self, q_network: nn.Module, state: torch.Tensor) -> np.ndarray:
        """어긋 엔트로피 방법으로 Q(state, action)을 가장 크게 하는 액션을 찾는다.
        
        인수:
            q_network: Q 네트워크
            state: 상태 텐서 하나, 형태 (state_dim,) 또는 (1, state_dim)
            
        반환값:
            찾아낸 가장 좋은 액션, 형태 (action_dim,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 분포 첫값 매기기
        mu = np.zeros(self.action_dim)
        sigma = np.ones(self.action_dim)

        best_action = mu.copy()
        best_q = -float('inf')

        for _ in range(self.n_iterations):
            # 가우스에서 액션 뽑기
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
        """상태 배치의 액션을 다듬는다."""
        actions = []
        for i in range(states.shape[0]):
            a = self.optimize(q_network, states[i])
            actions.append(a)
        return torch.FloatTensor(np.array(actions))


# ---------------------------------------------------------------------------
# 되돌려 보기 버퍼
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
# QT-Opt 에이전트
# ---------------------------------------------------------------------------

class QTOptAgent:
    """QT-Opt: 이어진 액션을 위한 Q 러닝과 어긋 엔트로피 방법."""

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

        # 타깃: r + γ max_a' Q_target(s', a')
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
    demo_qt_opt()
```

## 2. 논의

이 구현은 QT-Opt의 핵심 논리를 감싼 `ContinuousQNetwork`, `CEM`, `ReplayBuffer` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

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
    이 구조 선택은 연속 액션 공간에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — QT-Opt

이 구현은 QT-Opt의 핵심 논리를 감싼 `ContinuousQNetwork`, `CEM`, `ReplayBuffer` 클래스를 중심에 둔다.

핵심 클래스는 `ContinuousQNetwork`, `CEM`, `ReplayBuffer`, `QTOptAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
