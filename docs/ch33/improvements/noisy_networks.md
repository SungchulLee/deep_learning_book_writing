# 잡소리 그물

잡소리 그물은 DQN 좋게 하기의 중요한 개념이다. 배우는 살펴보기를 위한 NoisyLinear 층과 NoisyNet Q 그물을 다룬다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
33.2.4 시끄러운 그물
======================

익힌 살펴보기를 위한 NoisyLinear 켜와 시끄러운 그물 Q 그물.
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
# 시끄러운 선형 켜(인수분해 가우스)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """인수분해 가우스 시끄러운 그물 선형 켜.
    
    y = (mu_w + sigma_w * eps_w) @ x + (mu_b + sigma_b * eps_b)
    여기서 eps는 eps_ij = f(eps_i) * f(eps_j), f(x) = sign(x)*sqrt(|x|)로 인수분해된다
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 학습 가능한 매개변수
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # 인수분해 시끄러움 담개(값이 아님)
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
# 홀로선 가우스 시끄러운 선형(견주기용)
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
# 시끄러운 그물 Q 그물
# ---------------------------------------------------------------------------

class NoisyQNetwork(nn.Module):
    """살펴보기를 위해 NoisyLinear 켜를 쓴 Q 그물."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 sigma_init: float = 0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU())
        # 마지막 켜를 시끄러운 켜로 바꿈
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
# 되돌려 보기 담개
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

    # --- 시끄러운 그물로 익히기(엡실론 욕심쟁이 없음) ---
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
    demo_noisy_networks()```

## 논의

이 짜기는 잡소리 그물의 핵심 논리를 감싼 `NoisyLinear`, `IndependentNoisyLinear`, `NoisyQNetwork` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
