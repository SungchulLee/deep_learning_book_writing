# 분포적 강화학습

분포 강화학습은 기댓값만이 아니라 리턴의 전체 분포를 나타낸다. C51 알고리즘은 리턴 분포를 고정된 원자 집합에 대한 범주형 분포로 나타내고, QR-DQN은 분위 회귀로 리턴 분포의 분위 함수를 배운다. 리턴에 원래 있는 확률성을 담아 분포 방법은 더 풍부한 학습 신호를 주며 흔히 일반 밸류 기반 길보다 나은 성능을 낸다.

## 1. 코드

```python
"""
35.2.5 분포 힘 북돋우는 러닝
==========================

C51(클래스 DQN)과 QR-DQN 짜기.
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
# C51: 클래스 DQN
# ---------------------------------------------------------------------------

class C51Network(nn.Module):
    """C51 네트워크: 돌아옴 알갱이 위의 확률 분포를 내놓는다."""

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
    """타깃 분포를 고정된 알갱이 받침에 쏘아 내린다."""
    batch_size = next_states.size(0)
    n_atoms = target_net.n_atoms
    atoms = target_net.atoms
    v_min = target_net.v_min
    v_max = target_net.v_max
    delta_z = target_net.delta_z

    with torch.no_grad():
        # 가장 좋은 액션의 타깃 분포를 얻음
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
    """C51 어긋 엔트로피 로스."""
    target_dist = c51_projection(target_net, next_states, rewards, dones, gamma)
    log_probs = online_net(states)
    idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, online_net.n_atoms)
    log_probs_a = log_probs.gather(1, idx).squeeze(1)
    loss = -(target_dist * log_probs_a).sum(dim=1).mean()
    return loss


# ---------------------------------------------------------------------------
# QR-DQN: 분위 회귀 DQN
# ---------------------------------------------------------------------------

class QRDQNNetwork(nn.Module):
    """분위 회귀 DQN: 분위 값을 내놓는다."""

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
    """QR-DQN을 위한 분위 후버 로스.
    
    quantiles: (batch, n_quantiles) — 미리 잡은 값
    target_quantiles: (batch, n_quantiles) — 타깃
    taus: (n_quantiles,) — 분위 몫
    """
    n = quantiles.size(1)
    # 짝별 때 차이 어긋남: (batch, n, n)
    td_errors = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)

    # 후버 로스
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

    # --- C51 구조 ---
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

    # --- QR-DQN 구조 ---
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
    demo_distributional_rl()
```

## 2. 논의

이 구현은 분포 방법의 핵심 논리를 감싼 `ReplayBuffer`, `C51Network`, `QRDQNNetwork` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

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

**다룬 것** — 분포 강화학습

이 구현은 분포 방법의 핵심 논리를 감싼 `ReplayBuffer`, `C51Network`, `QRDQNNetwork` 클래스를 중심에 둔다.

핵심 클래스는 `ReplayBuffer`, `C51Network`, `QRDQNNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
