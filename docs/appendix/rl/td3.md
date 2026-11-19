# TD3

TD3은 2018년 논문 "Addressing Function Approximation Error in Actor-Critic Methods"에서 나왔다. 1) 쌍둥이 Q 네트워크(크리틱 둘 가운데 작은 값) 2) 타깃 폴리시 스무딩(타깃 액션에 잡음을 더한다) 3) 액터의 지연 갱신.

여기 구현은 TD3을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
TD3 - 쌍둥이 지연 DDPG(Twin Delayed DDPG)
논문: "액터-크리틱 방법의 함수 근사 오차 다루기" (2018)
지은이: 스콧 후지모토, 헤르케 판 호프, 데이비드 메거
핵심 아이디어:
  1) 쌍둥이 Q 네트워크(크리틱 둘 가운데 작은 값)
  2) 타깃 폴리시 스무딩(타깃 액션에 잡음을 더한다)
  3) 액터의 지연 갱신

파일: appendix/rl/td3.py
참고: 교육용 예제다(핵심 타깃 계산 + 네트워크).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class Actor(nn.Module):
    """결정론적 폴리시 a = pi(s)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # assume actions scaled to [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Q(s,a) 크리틱."""
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
        return self.net(x).squeeze(1)


def td3_target(q1_tgt, q2_tgt, actor_tgt, s2, r, done, gamma=0.99, noise_std=0.2, noise_clip=0.5):
    """
    TD3 타깃:
      a' = actor_tgt(s') + clipped_noise
      y  = r + gamma*(1-done) * min(Q1_tgt(s',a'), Q2_tgt(s',a'))

    타깃 폴리시를 스무딩하면 뾰족한 Q 봉우리에서 오는 과대추정이 줄어든다.
    """
    with torch.no_grad():
        a2 = actor_tgt(s2)

        # 클리핑된 가우스 잡음을 더한다
        noise = torch.randn_like(a2) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        a2 = (a2 + noise).clamp(-1.0, 1.0)

        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        y = r + gamma * (1.0 - done) * qmin
    return y


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`Actor`, `Critic`)를 정의하고, 이들이 함께 완전한 강화학습 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `Actor`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검증하려면 핵심 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 수 있는 실패 양상(failure mode) 둘을 들고, 각각을 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Actor`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = Actor(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — TD3

이 구현은 클래스 2개(`Actor`, `Critic`)를 정의하고, 이들이 함께 완전한 강화학습 구조를 이룬다.

핵심 클래스는 `Actor`, `Critic`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
