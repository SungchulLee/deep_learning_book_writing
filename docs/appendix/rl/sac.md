# SAC

SAC은 2018년 글 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"에서 나왔다. - 움직이는 이가 바라는 되받음 + 엔트로피(둘러보기)를 가장 크게 한다 - Q 네트워크 둘(겹 Q)로 위로 바이어스를 줄인다 - 든든함을 위해 타깃 Q 네트워크를 쓴다.

여기 구현은 SAC을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
SAC - 부드러운 움직이는 이-따지는 이(이어진 다루기)
글: "부드러운 움직이는 이-따지는 이: 폴리시 밖 가장 큰 엔트로피 깊은 북돋움 러닝" (2018)
지은이: 투오마스 하르노야 외
핵심 깨침:
  - 움직이는 이가 바라는 되받음 + 엔트로피(둘러보기)를 가장 크게 한다
  - Q 넷트웍 둘(겹 Q)로 위로 치우침을 줄인다
  - 든든함을 위해 타깃 Q 넷트웍을 쓴다

두루마리: appendix/rl/sac.py
눈여겨볼 것: 배우기 위한 본이다(넷트웍 + 핵심 타깃 셈).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class GaussianPolicy(nn.Module):
    """
    이어진 액션을 위한 확률 움직이는 이:
      a ~ N(mu(s), sigma(s))을 뽑은 뒤 흔히 tanh으로 [-1,1] 안에 눌러 넣는다.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)  # numerical stability
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        파라메터를 다시 잡은 뽑기:
          a = tanh(mu + std * eps)
        tanh 눌러 넣기를 셈에 넣은 log_prob(a)도 돌려준다(짧게 하려고 여기서는 뺐다).
        """
        mu, std = self.forward(obs)
        eps = torch.randn_like(std)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # 참 SAC 구현에서는 tanh 바로잡음을 넣어 log_prob을 셈한다.
        log_prob = None
        return a, log_prob


class QNetwork(nn.Module):
    """이어진 액션을 위한 따지는 이 Q(s,a)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(1)  # (B,)


def sac_target(q1_tgt, q2_tgt, policy, s2, r, done, gamma=0.99, alpha=0.2):
    """
    SAC 타깃을 셈한다:
      a' ~ pi(s')
      y = r + gamma*(1-done) * ( min(Q1_tgt(s',a'), Q2_tgt(s',a')) - alpha * log pi(a'|s') )

    여기서는 짧게 하려고 log pi 마디를 뺐다. 온전한 구현에서는 넣어야 한다.
    """
    with torch.no_grad():
        a2, logp2 = policy.sample(s2)
        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        # logp2이 None이면(이 배우기용 코드처럼) 엔트로피 마디를 셈에서 뺀다.
        if logp2 is None:
            backup = qmin
        else:
            backup = qmin - alpha * logp2

        y = r + gamma * (1.0 - done) * backup
    return y


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`GaussianPolicy`, `QNetwork`)를 매기고, 이들이 어울려 온전한 강화학습 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `GaussianPolicy`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GaussianPolicy`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = GaussianPolicy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — SAC

이 구현은 클래스 2개(`GaussianPolicy`, `QNetwork`)를 매기고, 이들이 어울려 온전한 강화학습 구조를 이룬다.

핵심 클래스는 `GaussianPolicy`, `QNetwork`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
