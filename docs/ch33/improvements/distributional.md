# 분포 힘 북돋우는 배움

분포 힘 북돋우는 배움은 기댓값만이 아니라 돌아옴의 온 분포를 나타낸다. C51 알고리즘은 돌아옴 분포를 붙박인 원자 모임에 대한 갈래 분포로 나타내고, QR-DQN은 분위 되돌이 맞춤으로 돌아옴 분포의 분위 함수를 배운다. 돌아옴에 본디 있는 확률성을 담아 분포 방법은 더 풍부한 배움 신호를 주며 흔히 여느 값 바탕 길보다 나은 성능을 낸다.

## 1. 코드

```python
"""
33.2.5 분포 힘 북돋우는 배움
==========================

C51(갈래 DQN)과 QR-DQN 짜기.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


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
# C51: 갈래 DQN
# ---------------------------------------------------------------------------

class C51Network(nn.Module):
    """C51 그물: 돌아옴 알갱이 위의 확률 분포를 내놓는다."""

    def __init__(self, state_dim: int, action_dim: int, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('atoms', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """로그 확률을 돌려준다: (batch, action_dim, n_atoms)."""
        feat = self.feature(x)
        logits = self.output(feat).view(-1, self.action_dim, self.n_atoms)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """분포에서 얻은 어림 Q 값."""
        probs = self.forward(x).exp()
        return (probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(dim=2)


def c51_projection(target_net: C51Network, next_states: torch.Tensor,
                   rewards: torch.Tensor, dones: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """과녁 분포를 고정된 알갱이 받침에 쏘아 내린다."""
    batch_size = next_states.size(0)
    n_atoms = target_net.n_atoms
    atoms = target_net.atoms
    v_min = target_net.v_min
    v_max = target_net.v_max
    delta_z = target_net.delta_z

    with torch.no_grad():
        # 가장 좋은 움직임의 과녁 분포를 얻음
        next_q = target_net.q_values(next_states)
        best_actions = next_q.argmax(dim=1)
        next_log_probs = target_net(next_states)
        idx = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, n_atoms)
        next_probs = next_log_probs.exp().gather(1, idx).squeeze(1)

        # 쏘아 내리기: T_z = r + gamma * z, [v_min, v_max]로 자름
        tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * atoms.unsqueeze(0)
        tz = tz.clamp(v_min, v_max)

        # 쏘아 내림 번호 셈
        b = (tz - v_min) / delta_z
        l = b.floor().long().clamp(0, n_atoms - 1)
        u = b.ceil().long().clamp(0, n_atoms - 1)

        # 확률을 나눠 줌
        m = torch.zeros(batch_size, n_atoms, device=next_states.device)
        offset = torch.arange(batch_size, device=next_states.device).unsqueeze(1) * n_atoms

        m.view(-1).index_add_(0, (l + offset).view(-1),
                               (next_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                               (next_probs * (b - l.float())).view(-1))
    return m


def c51_loss(online_net: C51Network, target_net: C51Network,
             states, actions, rewards, next_states, dones, gamma=0.99):
    """C51 어긋 엔트로피 손실."""
    target_dist = c51_projection(target_net, next_states, rewards, dones, gamma)
    log_probs = online_net(states)
    idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, online_net.n_atoms)
    log_probs_a = log_probs.gather(1, idx).squeeze(1)
    loss = -(target_dist * log_probs_a).sum(dim=1).mean()
    return loss


# ---------------------------------------------------------------------------
# QR-DQN: 분위 되돌이 DQN
# ---------------------------------------------------------------------------

class QRDQNNetwork(nn.Module):
    """분위 되돌이 DQN: 분위 값을 내놓는다."""

    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int = 51,
                 hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        taus = (2 * torch.arange(n_quantiles).float() + 1) / (2 * n_quantiles)
        self.register_buffer('taus', taus)

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, action_dim * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """분위 값을 돌려준다: (batch, action_dim, n_quantiles)."""
        feat = self.feature(x)
        return self.output(feat).view(-1, self.action_dim, self.n_quantiles)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """기대 Q는 분위 값의 평균이다."""
        return self.forward(x).mean(dim=2)


def quantile_huber_loss(quantiles: torch.Tensor, target_quantiles: torch.Tensor,
                        taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """QR-DQN을 위한 분위 후버 손실.
    
    quantiles: (batch, n_quantiles) — 미리 잡은 값
    target_quantiles: (batch, n_quantiles) — 과녁
    taus: (n_quantiles,) — 분위 몫
    """
    n = quantiles.size(1)
    # 짝별 때 차이 어긋남: (batch, n, n)
    td_errors = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)

    # 후버 손실
    huber = torch.where(td_errors.abs() <= kappa,
                        0.5 * td_errors.pow(2),
                        kappa * (td_errors.abs() - 0.5 * kappa))

    # 비대칭 무게
    tau_weight = (taus.unsqueeze(0).unsqueeze(2) - (td_errors < 0).float()).abs()
    loss = (tau_weight * huber).sum(dim=2).mean(dim=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_distributional_rl():
    print("=" * 60)
    print("Distributional RL Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # --- C51 얼개 ---
    print("\n--- C51 Architecture ---")
    c51 = C51Network(sd, ad, n_atoms=51, v_min=-10, v_max=10)
    x = torch.randn(4, sd)
    q = c51.q_values(x)
    log_p = c51(x)
    print(f"  Q-values shape: {q.shape}")
    print(f"  Distribution shape: {log_p.shape}")
    print(f"  Q-values: {q[0].detach().numpy().round(3)}")
    print(f"  Atom support: [{c51.v_min}, {c51.v_max}], {c51.n_atoms} atoms")
    print(f"  Params: {sum(p.numel() for p in c51.parameters()):,}")

    # --- QR-DQN 얼개 ---
    print("\n--- QR-DQN Architecture ---")
    qrdqn = QRDQNNetwork(sd, ad, n_quantiles=51)
    quantiles = qrdqn(x)
    q_qr = qrdqn.q_values(x)
    print(f"  Quantile values shape: {quantiles.shape}")
    print(f"  Q-values: {q_qr[0].detach().numpy().round(3)}")
    print(f"  Quantile fractions: {qrdqn.taus[:5].numpy().round(3)}...")
    print(f"  Params: {sum(p.numel() for p in qrdqn.parameters()):,}")

    # --- CartPole에서 C51 익히기 ---
    print("\n--- C51 Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    online = C51Network(sd, ad, n_atoms=51, v_min=0, v_max=200)
    target = C51Network(sd, ad, n_atoms=51, v_min=0, v_max=200)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    buf = ReplayBuffer(50000, sd)
    rewards_hist = []
    step = 0

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            step += 1
            eps = max(0.01, 1.0 - step / 5000)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = online.q_values(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, ns, float(done))
            if len(buf) >= 500:
                st, at, rt, nst, dt = buf.sample(64)
                loss = c51_loss(online, target, st, at, rt, nst, dt, gamma=0.99)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                if step % 200 == 0:
                    target.load_state_dict(online.state_dict())
            s = ns; total += r
        rewards_hist.append(total)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards_hist[-50:]):.1f}")

    env.close()

    # --- 익힌 분포 그려 보기 ---
    print("\n--- Learned Distribution ---")
    env2 = gym.make('CartPole-v1')
    s, _ = env2.reset()
    with torch.no_grad():
        log_p = online(torch.FloatTensor(s).unsqueeze(0))
        probs = log_p.exp()[0]  # (action_dim, n_atoms)
    for a_idx in range(ad):
        dist = probs[a_idx].numpy()
        mean_val = (dist * online.atoms.numpy()).sum()
        std_val = np.sqrt(((online.atoms.numpy() - mean_val)**2 * dist).sum())
        print(f"  Action {a_idx}: mean={mean_val:.2f}, std={std_val:.2f}")
    env2.close()

    print("\nDistributional RL demo complete!")


if __name__ == "__main__":
    demo_distributional_rl()```

## 2. 논의

이 짜기는 분포 방법의 핵심 논리를 감싼 `ReplayBuffer`, `C51Network`, `QRDQNNetwork` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 분포 힘 북돋우는 배움

이 짜기는 분포 방법의 핵심 논리를 감싼 `ReplayBuffer`, `C51Network`, `QRDQNNetwork` 갈래를 한가운데 둔다.

고갱이 갈래는 `ReplayBuffer`, `C51Network`, `QRDQNNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
