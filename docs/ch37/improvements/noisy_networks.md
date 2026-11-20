# Noisy Networks

Noisy Networks는 DQN 개선의 중요한 개념이다. $\varepsilon$-그리디는 정해진 확률로 무작위 행동을 섞어 탐험하지만, Noisy Networks 는 선형 레이어의 웨이트에 학습 가능한 크기의 노이즈를 더해 네트워크 자신이 탐험을 만들게 한다. 웨이트를 $w = \mu + \sigma \odot \varepsilon$ 로 두고 평균 $\mu$ 와 노이즈 크기 $\sigma$ 를 둘 다 경사 하강법으로 배우므로, 탐험이 얼마나 필요한지를 에이전트가 상태와 학습 단계에 맞춰 스스로 조절한다. 이 쪽은 그 레이어인 `NoisyLinear` 와, 그것으로 만든 Q 네트워크를 다룬다.

## 1. 코드

```python
"""
35.2.4 노이지 네트워크
======================

익힌 살펴보기를 위한 NoisyLinear 켜와 노이지 네트워크 Q 네트워크.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from typing import List, Tuple
import math

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 노이지 선형 켜(인수분해 가우스)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """인수분해 가우스 노이지 네트워크 선형 켜.
    
    y = (mu_w + sigma_w * eps_w) @ x + (mu_b + sigma_b * eps_b)
    여기서 eps는 eps_ij = f(eps_i) * f(eps_j), f(x) = sign(x)*sqrt(|x|)로 인수분해된다
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 학습 가능한 파라메터
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # 인수분해 시끄러움 버퍼(값이 아님)
        self.register_buffer('eps_in', torch.zeros(in_features))
        self.register_buffer('eps_out', torch.zeros(out_features))

        # 초기화한다
        bound = 1 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init / math.sqrt(in_features))
        self.sigma_bias.data.fill_(sigma_init / math.sqrt(out_features))

        self.reset_noise()

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        """f(x) = sign(x) * sqrt(|x|)"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """인수분해 시끄러움을 다시 뽑는다."""
        eps_i = self._factorized_noise(self.in_features)
        eps_j = self._factorized_noise(self.out_features)
        self.eps_in.copy_(eps_i)
        self.eps_out.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 인수분해 시끄러움: eps_w = eps_out^T @ eps_in
            eps_w = self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
            eps_b = self.eps_out
            weight = self.mu_weight + self.sigma_weight * eps_w
            bias = self.mu_bias + self.sigma_bias * eps_b
        else:
            # 값 매길 때는 시끄러움 없음
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)

    @property
    def noise_magnitude(self) -> float:
        """시그마 값의 평균 크기(지켜보기용)."""
        return (self.sigma_weight.abs().mean() + self.sigma_bias.abs().mean()).item() / 2


# ---------------------------------------------------------------------------
# 홀로선 가우스 노이지 선형(견주기용)
# ---------------------------------------------------------------------------

class IndependentNoisyLinear(nn.Module):
    """무게마다 홀로선 가우스 시끄러움(더 비싸다)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        self.register_buffer('eps_weight', torch.zeros(out_features, in_features))
        self.register_buffer('eps_bias', torch.zeros(out_features))

        bound = 1 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init)
        self.sigma_bias.data.fill_(sigma_init)

        self.reset_noise()

    def reset_noise(self):
        self.eps_weight.normal_()
        self.eps_bias.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# 노이지 네트워크 Q 네트워크
# ---------------------------------------------------------------------------

class NoisyQNetwork(nn.Module):
    """살펴보기를 위해 NoisyLinear 켜를 쓴 Q 네트워크."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 sigma_init: float = 0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU())
        # 마지막 켜를 노이지 켜로 바꿈
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        feat = F.relu(self.noisy1(feat))
        return self.noisy2(feat)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    @property
    def noise_magnitude(self) -> float:
        return (self.noisy1.noise_magnitude + self.noisy2.noise_magnitude) / 2


# ---------------------------------------------------------------------------
# 되돌려 보기 버퍼
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd):
        self.cap=cap;self.sz=0;self.p=0
        self.s=np.zeros((cap,sd),np.float32);self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32);self.ns=np.zeros((cap,sd),np.float32)
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
# 시연
# ---------------------------------------------------------------------------

def demo_noisy_networks():
    print("=" * 60)
    print("Noisy Networks Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # --- NoisyLinear 켜 살피기 ---
    print("\n--- NoisyLinear Layer ---")
    noisy = NoisyLinear(sd, 64)
    x = torch.randn(5, sd)

    outputs = []
    for _ in range(10):
        noisy.reset_noise()
        outputs.append(noisy(x).detach())
    outputs = torch.stack(outputs)
    print(f"  Output variance across noise samples: {outputs.var(dim=0).mean():.4f}")
    print(f"  Sigma magnitude: {noisy.noise_magnitude:.4f}")

    # --- 노이지 네트워크로 익히기(엡실론 욕심쟁이 없음) ---
    print("\n--- NoisyNet DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    online = NoisyQNetwork(sd, ad)
    target = NoisyQNetwork(sd, ad)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    buf = ReplayBuffer(50000, sd)

    rewards = []
    noise_mags = []
    step = 0

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        online.reset_noise()  # 마당마다 시끄러움 되돌리기

        while not done:
            step += 1
            # 엡실론 욕심쟁이 없음! 시끄러움이 살펴보기를 맡음
            with torch.no_grad():
                online.train()
                a = online(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, ns, float(done))

            if len(buf) >= 500:
                st, at, rt, nst, dt = buf.sample(64)
                online.reset_noise()
                target.reset_noise()
                q = online(st).gather(1, at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = target(nst).max(1)[0]
                    tgt = rt + (1 - dt) * 0.99 * nq
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()

                if step % 200 == 0:
                    target.load_state_dict(online.state_dict())

            s = ns; total += r

        rewards.append(total)
        noise_mags.append(online.noise_magnitude)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards[-50:]):.1f}, "
                  f"σ={online.noise_magnitude:.4f}")

    env.close()

    # --- 시끄러움의 흐름 ---
    print("\n--- Noise Magnitude Evolution ---")
    for i in range(0, len(noise_mags), 50):
        end = min(i + 50, len(noise_mags))
        avg_sigma = np.mean(noise_mags[i:end])
        print(f"  Episodes {i+1}-{end}: avg σ = {avg_sigma:.4f}")

    print(f"\n  Initial σ ≈ {noise_mags[0]:.4f}")
    print(f"  Final σ ≈ {noise_mags[-1]:.4f}")
    print(f"  σ reduction: {(1 - noise_mags[-1]/noise_mags[0])*100:.1f}%")

    print("\nNoisy networks demo complete!")


if __name__ == "__main__":
    demo_noisy_networks()
```

## 2. 논의

이 구현의 중심은 `NoisyLinear` 다. 노이즈를 웨이트마다 따로 뽑지 않고 **인수분해 가우스 노이즈**를 쓴다. 입력 쪽 노이즈 $\varepsilon_i$ ($p$ 개)와 출력 쪽 노이즈 $\varepsilon_j$ ($q$ 개)만 뽑고, 웨이트 노이즈를 $\varepsilon_{ij} = f(\varepsilon_i) f(\varepsilon_j)$, $f(x) = \operatorname{sign}(x)\sqrt{|x|}$ 로 만든다. `IndependentNoisyLinear` 는 웨이트마다 노이즈를 따로 뽑는 비교용 레이어다. 둘 다 `eval()` 모드에서는 노이즈를 끄고 $\mu$ 만 쓴다. `NoisyQNetwork` 는 첫 레이어는 일반 `nn.Linear` 로 두고 그 뒤 두 레이어를 `NoisyLinear` 로 바꾼 Q 네트워크다.

데모 함수는 두 부분이다. 먼저 `NoisyLinear` 하나에 같은 입력을 넣고 노이즈를 열 번 다시 뽑아 출력 분산과 $\sigma$ 의 평균 크기를 찍는다. 이 부분은 시드를 고정하기 전에 실행되므로 돌릴 때마다 값이 조금씩 다르다. 다음으로 시드 42 로 $\varepsilon$-그리디 없이 NoisyNet DQN 을 CartPole 에서 250 에피소드 학습하며, 50 에피소드마다 평균 리워드와 $\sigma$ 의 평균 크기를 찍고, 끝에서 $\sigma$ 가 처음보다 몇 % 줄었는지 보인다. $\sigma$ 가 줄어든다는 것은 에이전트가 학습이 진행되며 노이즈, 곧 탐험을 스스로 줄였다는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 `NoisyQNetwork(4, 2)`(은닉 128, `sigma_init=0.5`)에서 학습 전 `noise_magnitude` 를 손으로 계산하라. 데모가 찍는 `Initial σ` 가 이 값과 같아야 하는 까닭은 무엇인가?

</div>

??? success "풀이"
    `NoisyLinear` 는 $\sigma_w$ 를 $0.5/\sqrt{p}$, $\sigma_b$ 를 $0.5/\sqrt{q}$ 로 채운다($p$ 는 입력, $q$ 는 출력 크기). `noisy1`(128→128)은 둘 다 $0.5/\sqrt{128} \approx 0.04419$ 라 크기가 0.04419 다. `noisy2`(128→2)는 $\sigma_w \approx 0.04419$, $\sigma_b = 0.5/\sqrt2 \approx 0.35355$ 라 평균이 0.19887 이다. 네트워크의 `noise_magnitude` 는 둘의 평균인 약 0.1215 다. `Initial σ` 는 첫 에피소드가 끝난 뒤의 값인데, 버퍼가 500 개가 되기 전에는 업데이트가 없으므로 첫 에피소드(CartPole 에서 보통 수십 스텝) 동안 $\sigma$ 가 바뀌지 않는다. 그래서 초기값 0.1215 가 그대로 찍혀야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 $p$, 출력 $q$ 인 레이어에서 `NoisyLinear`(인수분해)와 `IndependentNoisyLinear`(독립)가 노이즈를 한 번 다시 뽑을 때 만드는 가우스 난수의 개수를 견주어라. 데모의 128→128 레이어에서는 각각 몇 개인가? 학습할 파라메터 수는 다른가?

</div>

??? success "풀이"
    독립 노이즈는 웨이트마다 하나, 바이어스마다 하나라 $pq + q$ 개다. 인수분해 노이즈는 입력 쪽 $p$ 개와 출력 쪽 $q$ 개만 뽑고 웨이트 노이즈는 바깥곱으로, 바이어스 노이즈는 출력 쪽 노이즈를 그대로 쓰므로 $p + q$ 개다. 128→128 이면 독립이 $16{,}384 + 128 = 16{,}512$ 개, 인수분해가 256 개로 약 64분의 1 이다. 학습할 파라메터는 둘 다 $\mu_w, \sigma_w, \mu_b, \sigma_b$ 로 $2(pq + q)$ 개이며 같다. 차이는 난수 생성 비용과 노이즈의 상관 구조뿐이다. Fortunato 등은 DQN 처럼 계산이 빠듯한 알고리즘에는 인수분해를, A3C 처럼 비동기 알고리즘에는 독립 노이즈를 권했다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
데모의 학습 루프는 에피소드를 시작할 때 `online.reset_noise()` 를 부르고 주석에 「에피소드마다 노이즈를 다시 뽑는다」는 뜻을 적었다. 그런데 버퍼가 500 개를 넘은 뒤 행동을 고를 때 쓰는 노이즈는 실제로 얼마나 자주 바뀌는가? 이것이 탐험의 성질에 어떤 차이를 만드는지 설명하라.

</div>

??? success "풀이"
    버퍼가 500 개를 넘으면 매 환경 스텝마다 업데이트 직전에 `online.reset_noise()` 를 다시 부른다. 그래서 다음 스텝의 행동은 업데이트 때 뽑은 노이즈로 고르게 되고, 행동 노이즈가 사실상 **스텝마다** 바뀐다. 에피소드 시작 때의 `reset_noise` 는 첫 500 스텝 동안만 주석대로 작동한다. 노이즈를 에피소드 동안 고정하면 에이전트는 「약간 다른 폴리시 하나」를 에피소드 끝까지 일관되게 따르므로, 여러 스텝에 걸친 탐험(깊은 탐험)이 가능하다. 스텝마다 바뀌면 행동 하나하나에 독립된 잡음을 섞는 것에 가까워져 $\varepsilon$-그리디와 비슷한 성질이 된다. 주석의 의도대로 하려면 행동 선택용 네트워크의 노이즈를 업데이트용과 따로 두거나, 업데이트가 끝난 뒤 행동 노이즈를 저장해 둔 값으로 되돌려야 한다.

## 정리하며

**다룬 것** — Noisy Networks

이 구현은 인수분해 가우스 노이즈를 쓰는 `NoisyLinear`, 비교용 `IndependentNoisyLinear`, 그리고 이 레이어로 $\varepsilon$-그리디 없이 탐험하는 `NoisyQNetwork` 를 중심에 둔다.

핵심 클래스는 `NoisyLinear`, `IndependentNoisyLinear`, `NoisyQNetwork`, `ReplayBuffer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
