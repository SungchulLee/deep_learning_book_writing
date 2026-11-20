# 분포적 강화학습

분포적 강화학습은 리턴의 기댓값 $Q(s, a)$ 하나만이 아니라 리턴의 분포 $Z(s, a)$ 전체를 배운다. C51 알고리즘은 리턴이 놓일 수 있는 범위 $[V_\min, V_\max]$ 에 고정된 값 51 개(원자, atom)를 깔고, 리턴 분포를 이 원자들 위의 범주형 분포로 나타낸다. QR-DQN 은 반대로 확률 쪽을 고정하고, 분위 회귀(quantile regression)로 정해진 분위수 $\tau_i$ 마다의 리턴 값을 배운다. 리턴에 원래 있는 무작위성까지 담으므로 분포적 방법은 더 풍부한 학습 신호를 주며, 아타리 벤치마크에서는 흔히 일반 밸류 기반 방법보다 나은 성능을 냈다.

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

C51 의 핵심은 `c51_projection` 이다. 벨만 연산 $\mathcal{T}z_j = r + \gamma z_j$ 를 원자마다 적용하면 원자들이 고정된 자리에서 벗어나므로, 옮겨진 값 하나하나의 확률을 그 양옆의 두 원자에 거리에 반비례하게 나눠 준다. 그렇게 고정된 원자 위로 되돌린 타깃 분포와 온라인 네트워크의 분포 사이의 교차 엔트로피가 `c51_loss` 다. QR-DQN 쪽은 네트워크가 분위수 51 개의 리턴 값을 바로 내고, `quantile_huber_loss` 가 분위수마다 비대칭 가중치 $|\tau - \mathbb{1}[\delta < 0]|$ 를 단 후버 로스로 그것을 배운다. Q 값은 C51 에서는 분포의 평균 $\sum_j p_j z_j$, QR-DQN 에서는 분위 값들의 평균이다.

데모 함수는 두 네트워크에 무작위 입력을 넣어 출력 형태, Q 값, 파라메터 수를 확인한 뒤(이 부분은 시드를 고정하기 전이라 값이 돌릴 때마다 다르다), C51 만 CartPole 에서 시드 42 로 250 에피소드 학습한다. QR-DQN 은 구조만 보이고 학습하지 않는다. 학습이 끝나면 첫 상태에서 두 행동의 리턴 분포 평균과 표준편차를 찍는다. CartPole 은 매 스텝 리워드가 1 이고 $\gamma = 0.99$ 라 리턴이 100 을 넘을 수 없는데, 데모는 지지 구간을 $[0, 200]$ 으로 잡았다(연습문제 2).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 학습 설정(원자 51 개, $V_\min = 0$, $V_\max = 200$, $\gamma = 0.99$)에서 원자 간격 $\Delta z$ 를 구하라. 리워드 $r = 1$, 종료 아님인 전이에서 원자 $z = 10$ 위의 확률 질량은 `c51_projection` 에 의해 어느 두 원자에 어떤 비율로 나뉘는가? 종료($d = 1$)인 전이에서는 어떻게 되는가?

</div>

??? success "풀이"
    $\Delta z = (200 - 0)/50 = 4$ 이므로 원자는 $0, 4, 8, \dots, 200$ 이다. $\mathcal{T}z = 1 + 0.99 \times 10 = 10.9$ 이고 $b = 10.9 / 4 = 2.725$ 라 $l = 2$ (원자 8), $u = 3$ (원자 12)이다. 원자 8 에는 $u - b = 0.275$, 원자 12 에는 $b - l = 0.725$ 의 비율이 간다. 검산하면 $0.275 \times 8 + 0.725 \times 12 = 10.9$ 로 평균이 보존된다. 종료 전이에서는 `(1 - dones)` 때문에 모든 원자가 $\mathcal{T}z = r = 1$ 로 모이고, $b = 0.25$ 라 질량 전체의 0.75 가 원자 0 에, 0.25 가 원자 4 에 간다. 평균은 $0.25 \times 4 = 1$ 이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CartPole 에서 $\gamma = 0.99$ 일 때 리턴이 가질 수 있는 범위를 구하고, 데모의 지지 구간 $[0, 200]$ 이 알맞은지 따져 보라. 원자 수를 그대로 두고 구간을 $[0, 100]$ 으로 줄이면 무엇이 달라지는가?

</div>

??? success "풀이"
    리워드가 매 스텝 1 이므로 리턴은 $\sum_{k=0}^{T-1}\gamma^k = (1-\gamma^T)/(1-\gamma)$ 로 1 이상 100 미만이다. 500 스텝에서 잘려도 $(1 - 0.99^{500}) \times 100 \approx 99.3$ 이다. 그러니 $[0, 200]$ 에서 위쪽 절반의 원자 25 개는 질량을 받을 일이 거의 없고, 분포를 실제로 나타내는 원자는 절반뿐이다. $[0, 100]$ 으로 줄이면 $\Delta z$ 가 4 에서 2 로 줄어 같은 원자 수로 분포를 두 배 촘촘하게 나타낼 수 있다. 다만 구간을 리턴의 최댓값에 너무 바짝 붙이면 학습 초기에 과대추정된 값이 `clamp` 로 잘려 위쪽 끝 원자에 질량이 쌓이므로, 약간의 여유를 두는 편이 안전하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`c51_projection` 에서 $b$ 가 정확히 정수가 되면 `l` 과 `u` 가 같아진다. 그때 두 `index_add_` 가 더하는 질량은 얼마인가? 데모 설정에서 이런 일이 일어날 수 있는 원자를 하나 찾고, 질량이 보존되도록 코드를 고쳐라.

</div>

??? success "풀이"
    $l = u = b$ 이면 `u.float() - b` 와 `b - l.float()` 가 둘 다 0 이라, 그 원자의 확률 질량이 어디에도 더해지지 않고 사라진다. 그러면 타깃 분포의 합이 1 보다 작아져 교차 엔트로피 로스가 엉뚱한 방향으로 그래디언트를 낸다. 데모 설정에서는 원자 $z = 100$ 이 그 예다. $r = 1$ 이면 $\mathcal{T}z = 1 + 0.99 \times 100 = 100$ 이고 $b = 25$ 로 정수다(부동소수점 반올림에 따라 25 를 아주 조금 벗어나면 이 일이 피해지기도 한다). 지지 구간 끝으로 `clamp` 된 값도 $b$ 가 정확히 0 이나 50 이 되어 같은 문제를 겪는다. 흔히 쓰는 수정은 `l == u` 인 자리에서 한쪽을 한 칸 옮기는 것이다. `l[(u > 0) & (l == u)] -= 1` 을 한 뒤 `u[(l < n_atoms - 1) & (l == u)] += 1` 을 하면, $b$ 가 정수인 원자는 $u - b = 1$ 또는 $b - l = 1$ 이 되어 질량 전체가 그 원자 하나에 온전히 간다.

## 정리하며

**다룬 것** — 분포적 강화학습

이 구현은 리턴 분포를 고정된 원자 위의 범주형 분포로 배우는 `C51Network`(투영 `c51_projection` 과 교차 엔트로피 로스)와, 분위 값을 배우는 `QRDQNNetwork`(분위 후버 로스)를 중심에 둔다.

핵심 클래스는 `ReplayBuffer`, `C51Network`, `QRDQNNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
