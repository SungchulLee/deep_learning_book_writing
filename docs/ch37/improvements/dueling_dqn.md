# Dueling DQN

Dueling DQN은 특별한 네트워크 구조로 Q 값을 상태 밸류 함수와 액션 어드밴티지 함수로 나눈다. $Q(s, a) = V(s) + A(s, a) - \text{mean}_a A(s, a)$으로 쪼개면 네트워크가 어떤 액션이 있든 상관없이 어느 상태가 값진지 배울 수 있어, 특히 많은 액션의 효과가 비슷한 둘레에서 학습이 효율적이 된다. 이 구현은 다층 뉴럴 네트워크와 컨볼루션 네트워크 모두에 듀얼링 구조를 보이고 타깃 계산에 Double DQN을 더한다.

## 1. 코드

```python
"""
35.2.2 겨루기 DQN
===================

값 클래스와 이점 클래스를 지닌 겨루기 네트워크 구조.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Tuple, List
import random

# ========================================================================
# 메인
# ========================================================================


class DuelingQNetwork(nn.Module):
    """겨루기 Q 네트워크: 값 클래스와 이점 클래스를 나눈다.
    
    Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # 함께 쓰는 피처 뽑개
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # 값 클래스: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 이점 클래스: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        # 가려낼 수 있게 평균으로 가운데 맞춤
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def value_and_advantage(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """살피기를 위해 V와 A를 따로 돌려준다."""
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


class DuelingConvQNetwork(nn.Module):
    """아타리 클래스 그림 살핌을 위한 겨루기 구조."""

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        feat = self.conv(x).view(x.size(0), -1)
        v = self.value_stream(feat)
        a = self.advantage_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# 되돌려 보기 버퍼
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd):
        self.cap=cap; self.sz=0; self.p=0
        self.s=np.zeros((cap,sd),np.float32); self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32); self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r
        self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.LongTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),
                torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


# ---------------------------------------------------------------------------
# 겨루기 구조와 겹 DQN으로 익히기
# ---------------------------------------------------------------------------

class DuelingDQNAgent:
    """겨루기 구조와 겹 DQN을 엮은 에이전트."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.01, eps_decay=5000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = DuelingQNetwork(state_dim, action_dim)
        self.target = DuelingQNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buf_cap, state_dim)

        self.step = 0
        self.updates = 0
        self.v_history = []
        self.a_history = []

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

        # V와 A 쪼개기 좇기
        with torch.no_grad():
            v, adv = self.online.value_and_advantage(s)
            self.v_history.append(v.mean().item())
            self.a_history.append(adv.abs().mean().item())

        # 겨루기 구조를 쓴 겹 DQN 타깃
        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            next_q = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            targets = r + (1 - d) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_dueling_dqn():
    print("=" * 60)
    print("Dueling DQN Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n
    env.close()

    # --- 구조 견주기 ---
    print("\n--- Architecture Comparison ---")
    standard = nn.Sequential(
        nn.Linear(sd, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    dueling = DuelingQNetwork(sd, ad)

    std_params = sum(p.numel() for p in standard.parameters())
    duel_params = sum(p.numel() for p in dueling.parameters())
    print(f"  Standard Q-Net params: {std_params:,}")
    print(f"  Dueling Q-Net params:  {duel_params:,}")

    # --- V/A 쪼개기 ---
    print("\n--- Value/Advantage Decomposition ---")
    test_states = torch.randn(5, sd)
    q_vals = dueling(test_states)
    v_vals, a_vals = dueling.value_and_advantage(test_states)

    for i in range(5):
        print(f"  State {i}: V={v_vals[i].item():.3f}, "
              f"A={a_vals[i].detach().numpy().round(3)}, "
              f"Q={q_vals[i].detach().numpy().round(3)}")

    # --- 익히기 견주기 ---
    print("\n--- Training: Dueling+Double DQN on CartPole ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    agent = DuelingDQNAgent(sd, ad)
    env = gym.make('CartPole-v1')
    rewards = []

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, float(done))
            agent.update()
            s = ns; total += r
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"  Episode {ep+1}: avg50={avg:.1f}, ε={agent.epsilon:.3f}")
    env.close()

    # --- V/A 클래스 살피기 ---
    print("\n--- V/A Stream Analysis ---")
    if agent.v_history:
        v_early = np.mean(agent.v_history[:200])
        v_late = np.mean(agent.v_history[-200:])
        a_early = np.mean(agent.a_history[:200])
        a_late = np.mean(agent.a_history[-200:])
        print(f"  V stream — early: {v_early:.3f}, late: {v_late:.3f}")
        print(f"  A stream (|A|) — early: {a_early:.3f}, late: {a_late:.3f}")
        print(f"  V/|A| ratio — early: {v_early/(a_early+1e-8):.2f}, "
              f"late: {v_late/(a_late+1e-8):.2f}")

    # --- 감음 겨루기 구조 형태 살피기 ---
    print("\n--- Conv Dueling Architecture ---")
    conv_duel = DuelingConvQNetwork(action_dim=4)
    dummy = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    q_out = conv_duel(dummy.float())
    print(f"  Input: {dummy.shape} → Output: {q_out.shape}")
    print(f"  Params: {sum(p.numel() for p in conv_duel.parameters()):,}")

    print("\nDueling DQN demo complete!")


if __name__ == "__main__":
    demo_dueling_dqn()
```

## 2. 논의

이 구현은 Dueling DQN의 핵심 논리를 감싼 `DuelingQNetwork`, `DuelingConvQNetwork`, `ReplayBuffer` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

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

**다룬 것** — Dueling DQN

이 구현은 Dueling DQN의 핵심 논리를 감싼 `DuelingQNetwork`, `DuelingConvQNetwork`, `ReplayBuffer` 클래스를 중심에 둔다.

핵심 클래스는 `DuelingQNetwork`, `DuelingConvQNetwork`, `ReplayBuffer`, `DuelingDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
