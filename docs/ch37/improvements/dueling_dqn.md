# Dueling DQN

Dueling DQN은 네트워크 구조를 바꾸어 Q 값을 상태 밸류 함수 $V(s)$ 와 액션 어드밴티지 함수 $A(s, a)$ 로 나눈다. 어드밴티지는 그 상태에서 액션 $a$ 가 평균보다 얼마나 나은지를 나타낸다. 공유 피처 위에 두 갈래의 출력 스트림을 두고 $Q(s, a) = V(s) + A(s, a) - \text{mean}_a A(s, a)$로 합치면, 네트워크는 어떤 액션을 고르든 상관없이 그 상태가 얼마나 좋은지를 $V$ 로 따로 배울 수 있다. 그래서 많은 액션의 효과가 비슷한 둘레에서 특히 트레인이 효율적이다. 이 구현은 MLP 와 컨볼루션 네트워크 모두에 듀얼링 구조를 보이고 타깃 계산에 Double DQN을 더한다.

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

이 구현의 중심은 `DuelingQNetwork` 의 `forward` 마지막 줄 `value + (advantage - advantage.mean(dim=1, keepdim=True))` 이다. `DuelingConvQNetwork` 는 같은 구조를 아타리용 컨볼루션 백본 위에 올린 것이고, `DuelingDQNAgent` 는 이 네트워크에 Double DQN 타깃과 후버 로스를 붙인 에이전트다.

데모 함수는 네 부분으로 되어 있다. (1) 같은 은닉 크기의 일반 Q 네트워크와 듀얼링 네트워크의 파라메터 수를 견준다. 듀얼링 쪽은 두 스트림이 저마다 은닉 레이어를 하나씩 가져 파라메터가 두 배쯤 된다. (2) 무작위 상태 다섯 개에 대해 $V$, $A$, $Q$ 를 찍는다. 여기서 찍는 $A$ 는 `value_and_advantage` 가 돌려주는 평균을 빼기 **전**의 값이므로 $V + A$ 가 찍힌 $Q$ 와 같지 않다. 같아지는 것은 $V + A - \text{mean}_a A$ 다. (3) CartPole 에서 시드 42 로 250 에피소드를 트레인하며 50 에피소드마다 평균 리워드를 찍고, 트레인 초기와 말기 200 업데이트의 $V$ 평균과 $|A|$ 평균을 견준다. (4) 아타리 크기 입력 $(2, 4, 84, 84)$ 을 컨볼루션 듀얼링 네트워크에 넣어 출력 형태와 파라메터 수를 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CartPole(상태 4차원, 액션 2개)에서 데모의 일반 Q 네트워크와 `DuelingQNetwork`(은닉 128)의 파라메터 수를 손으로 세어라. 무엇이 차이를 만드는가?

</div>

??? success "풀이"
    일반 네트워크는 $4\cdot128+128 = 640$, $128\cdot128+128 = 16{,}512$, $128\cdot2+2 = 258$ 을 더해 17,410 개다. 듀얼링 네트워크는 공유 피처 레이어 640 개에, 값 스트림 $16{,}512 + (128+1) = 16{,}641$ 개와 어드밴티지 스트림 $16{,}512 + 258 = 16{,}770$ 개를 더해 34,051 개다. 차이는 두 스트림이 **저마다** $128\times128$ 은닉 레이어를 가진 데서 온다. 듀얼링이 일반 네트워크보다 나은지 견주려면 이 차이를 맞춰 주어야 공정하다. 일반 네트워크의 은닉 크기를 키우거나 듀얼링의 스트림을 줄여 파라메터 수를 비슷하게 맞춘 뒤 견준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`forward` 에서 평균을 빼지 않고 $Q = V + A$ 로 합치면 무엇이 문제인가? 평균을 빼면 $V(s)$ 는 Q 값들과 어떤 관계를 갖게 되는가? 원 논문이 고려한 다른 선택인 $\max_a A$ 빼기와는 무엇이 다른가?

</div>

??? success "풀이"
    $Q = V + A$ 이면 아무 상수 $c$ 에 대해 $V + c$ 와 $A - c$ 가 똑같은 $Q$ 를 낸다. 트레인 신호는 $Q$ 에만 오므로 $V$ 와 $A$ 가 하나로 정해지지 않는다(식별 불가능). 그러면 두 스트림이 자유롭게 떠돌아 $V$ 를 상태 가치로 읽을 수 없다. 평균을 빼면 $\text{mean}_a Q(s, a) = V(s)$ 가 성립해 $V$ 가 액션에 대한 Q 값의 평균으로 정해진다. $\max_a A$ 를 빼면 $\max_a Q(s, a) = V(s)$ 가 되어 $V$ 가 그리디 폴리시의 상태 가치라는 더 깔끔한 뜻을 갖는다. 하지만 최대인 액션이 바뀔 때마다 기준이 뛰어 트레인이 덜 안정적이다. Wang 등은 실험에서 평균 빼기가 더 안정적이라고 보고했고, 이 구현도 그것을 따른다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
데모의 「V/A Stream Analysis」는 `a_history` 에 `adv.abs().mean()` 을 기록하는데, 여기서 `adv` 는 평균을 빼기 전의 어드밴티지 스트림 출력이다. 이 지표와 그것으로 만든 `V/|A| ratio` 가 무엇을 재지 **못하는지** 설명하고, 어드밴티지가 액션을 얼마나 가르는지 바르게 재도록 고쳐라.

</div>

??? success "풀이"
    연습문제 2에서 보았듯 평균을 빼기 전의 어드밴티지 스트림 출력에는 $Q$ 에 영향을 주지 않는 상수만큼의 자유도가 있다. 그 상수가 트레인 중 어디로 떠돌든 `adv.abs().mean()` 은 그 크기를 그대로 담는다. 그래서 이 지표가 커졌다고 액션 사이의 차이가 커졌다고 말할 수 없다. 비율 `V/|A|` 도 같은 문제를 갖고, $V$ 가 0 근처를 지나면 비율이 터진다. 바르게 재려면 평균을 뺀 어드밴티지 `adv - adv.mean(dim=1, keepdim=True)` 의 절댓값 평균을 쓰거나, 액션 사이의 Q 값 차이 `q.max(1) - q.min(1)` 을 기록한다. 둘 다 $Q$ 만으로 정해지는 양이라 식별 불가능한 자유도에 흔들리지 않는다. 이렇게 고친 지표로 보면, 트레인이 진행되면서 $V$ 는 상태 가치(CartPole 에서 $\gamma = 0.99$ 이면 100 이하)로 커지고 어드밴티지는 그보다 훨씬 작게 남는지를 확인할 수 있다. 이것이 듀얼링 구조가 노리는 성질이다.

## 정리하며

**다룬 것** — Dueling DQN

이 구현은 Q 값을 상태 가치 스트림과 어드밴티지 스트림으로 나누는 `DuelingQNetwork`·`DuelingConvQNetwork` 와, 거기에 Double DQN 타깃을 붙인 `DuelingDQNAgent` 를 중심에 둔다.

핵심 클래스는 `DuelingQNetwork`, `DuelingConvQNetwork`, `ReplayBuffer`, `DuelingDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
