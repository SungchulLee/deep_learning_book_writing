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

이 구현은 네 클래스로 되어 있다. `ContinuousQNetwork` 는 상태와 액션을 이어 붙여 받아 스칼라 Q 값을 내는 MLP 다. 액션이 입력이므로 DQN 처럼 출력 중 최댓값을 고르는 방식으로는 $\max_a Q$ 를 구할 수 없고, 그 일을 `CEM` 이 맡는다. `CEM.optimize` 는 평균 0, 표준편차 1 인 가우스에서 액션 후보를 뽑아 액션 범위로 자르고, Q 값이 높은 상위(엘리트) 후보들의 평균과 표준편차로 분포를 다시 맞추기를 몇 번 되풀이한 뒤, 그동안 본 가장 좋은 후보를 돌려준다. `QTOptAgent` 는 행동을 고를 때(여기에 가우스 탐험 잡음을 더한다)와 TD 타깃의 $\max_{a'} Q_{\theta^-}(s', a')$ 을 구할 때 모두 CEM 을 쓰고, 타깃 네트워크는 $\tau = 0.005$ 소프트 업데이트로 따라가게 한다.

데모 함수는 두 부분이다. (1) 학습하지 않은 무작위 Q 네트워크에서 CEM 이 찾은 액션과, $[-2, 2]$ 를 100 칸으로 나눈 격자 탐색이 찾은 액션을 견준다. 이 부분은 시드를 고정하기 전이라 돌릴 때마다 값이 다르다. (2) 시드 42 로 Pendulum-v1 에서 150 에피소드를 학습하며 50 에피소드마다 평균 리턴을 찍는다. 이때 CEM 은 후보 32 개, 엘리트 4 개, 반복 2 번으로 줄여 쓴다. 그래도 업데이트 한 번마다 배치의 128 개 다음 상태에 CEM 을 하나씩 파이썬 루프로 돌리므로 매우 느리다(연습문제 1). 한 번 돌린 결과이므로 학습 곡선의 수치는 시드에 따라 달라진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 학습 설정(`cem_samples=32`, `cem_iters=2`, `batch_size=128`)에서 행동 선택 한 번과 업데이트 한 번에 Q 네트워크를 몇 개의 (상태, 액션) 쌍에 대해 평가하는가? Pendulum-v1 의 에피소드는 200 스텝이다. 150 에피소드 동안 모두 몇 번인가? 같은 일을 액션 2 개짜리 DQN 이 한다면 몇 번인가?

</div>

??? success "풀이"
    CEM 한 번은 반복마다 후보 32 개를 평가하므로 $32 \times 2 = 64$ 쌍이다. 행동 선택은 CEM 한 번이라 64 쌍, 업데이트는 다음 상태 128 개마다 CEM 을 돌리므로 $128 \times 64 = 8{,}192$ 쌍에 현재 Q 계산 128 쌍과 타깃 계산 128 쌍이 더해진다. 학습은 버퍼가 128 개가 된 뒤부터 매 스텝 일어나므로, 150 에피소드 $\times$ 200 스텝 $= 30{,}000$ 스텝 동안 업데이트 쪽만 해도 대략 $30{,}000 \times 8{,}192 \approx 2.5 \times 10^8$ 쌍이다. DQN 이라면 순전파 한 번이 모든 액션의 Q 값을 내므로 업데이트마다 배치 두 번(현재 상태, 다음 상태)이면 된다. 게다가 이 구현은 CEM 을 상태마다 따로 부르므로 파이썬 루프가 업데이트마다 128 번 돈다. 모든 상태의 후보를 `(128 × 32, ...)` 한 배치로 묶어 한 번에 평가하면 같은 계산을 훨씬 빨리 할 수 있다. QT-Opt 원 논문이 CEM 을 쓸 수 있었던 것은 액션 차원이 작고, 타깃 계산을 여러 기계에 나눠 대규모로 병렬화했기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
데모 첫 부분은 CEM 이 찾은 액션과 100 칸 격자 탐색이 찾은 액션의 차이(`CEM vs grid gap`)를 찍는다. 격자 쪽은 얼마나 정확한가? 차이가 0 이 아니면 어느 쪽이 더 나은 액션인지 어떻게 가리겠는가? `CEM.optimize` 의 첫 분포(평균 0, 표준편차 1)와 `np.clip` 이 최적 액션이 범위 끝(±2) 근처에 있을 때 어떤 영향을 주는지도 따져 보라.

</div>

??? success "풀이"
    `torch.linspace(-2, 2, 100)` 의 칸 간격은 $4/99 \approx 0.040$ 이므로, 격자의 답은 참 최대점에서 최대 0.02 쯤 떨어질 수 있다. 그러니 차이가 0.02 안쪽이면 CEM 이 틀렸다고 할 수 없고, CEM 이 격자보다 더 나은 점을 찾았을 수도 있다. 어느 쪽이 나은지는 액션 위치가 아니라 **두 액션에서의 Q 값**을 견주어 가린다. 범위 끝을 보면, 표준편차 1 인 첫 분포에서 $|a| > 2$ 인 표본은 약 4.6% 이고 이것들이 `np.clip` 으로 정확히 ±2 에 모인다. 최적이 끝에 있으면 오히려 끝값 표본이 엘리트가 되어 쉽게 찾지만, 최적이 끝 근처 안쪽(예: 1.8)에 있으면 평균 0 에서 출발한 분포가 반복 3 번(학습에서는 2 번) 만에 거기까지 좁혀 가지 못할 수 있다. 엘리트의 표준편차로 분포를 다시 맞추므로 분포가 빨리 좁아져 국소 최대에 머무를 위험도 있다. 반복 수와 후보 수를 늘리거나, 앞 스텝의 해를 다음 CEM 의 출발점으로 쓰면 나아진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`update` 는 타깃의 $\max_{a'} Q_{\theta^-}(s', a')$ 을 CEM 의 근사 최댓값으로 바꿔 쓴다. 이 근사가 TD 타깃에 어떤 방향의 바이어스를 주는지 설명하고, DQN 의 과대추정(「Double DQN」 쪽)과 어떻게 얽히는지 논하라.

</div>

??? success "풀이"
    CEM 이 돌려주는 액션 $\hat a$ 는 후보 가운데 가장 좋은 것일 뿐이므로 $Q_{\theta^-}(s', \hat a) \le \max_{a'} Q_{\theta^-}(s', a')$ 이다. 따라서 네트워크 자체에 대해서는 타깃이 늘 **아래쪽**으로 치우친다. 한편 $\max$ 를 쓰는 타깃은 Q 추정의 잡음 때문에 **위쪽**으로 치우친다(우연히 크게 추정된 액션을 고르기 쉽다). CEM 도 결국 잡음 섞인 Q 값 가운데 큰 것을 고르므로 이 과대추정을 그대로 물려받는다. 두 효과가 일부 상쇄되지만 크기가 상태마다 달라 서로 맞아떨어질 보장은 없고, CEM 은 표본에 따라 다른 답을 내므로 타깃의 분산도 커진다. CEM 의 후보 수를 늘리면 아래쪽 바이어스는 줄지만 위쪽 바이어스는 그만큼 드러난다. 그래서 QT-Opt 원 논문은 타깃 네트워크 두 개 가운데 작은 값을 쓰는 clipped double Q 로 과대추정을 따로 눌렀다. 이 구현에는 그 장치가 없다.

## 정리하며

**다룬 것** — QT-Opt

이 구현은 (상태, 액션)을 입력으로 받는 `ContinuousQNetwork` 와, 행동 선택과 TD 타깃의 최댓값을 모두 교차 엔트로피 방법으로 근사하는 `CEM`·`QTOptAgent` 를 중심에 둔다.

핵심 클래스는 `ContinuousQNetwork`, `CEM`, `ReplayBuffer`, `QTOptAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
