# DQN

DQN은 2015년 논문 "Human-level control through deep reinforcement learning"에서 나왔다. 뉴럴 네트워크로 행동 가치 함수 $Q(s,a)$를 근사한다 — 상태 $s$에서 액션 $a$를 고른 뒤 얻을 할인된 보상 합의 기댓값이다. 학습은 *타깃 네트워크*를 쓰는 TD 타깃 $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$로 한다. 타깃 네트워크는 주기적으로만 갱신하는 $Q$ 네트워크의 사본이라, 쫓아가는 목표가 매 스텝 흔들리지 않는다. 경험 리플레이(experience replay)는 지나간 전이를 버퍼에 쌓아 두고 무작위로 뽑아 학습하는 방법으로, 연속된 표본 사이의 상관을 끊는다.

여기 구현은 DQN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DQN - 심층 Q 네트워크
논문: "심층 강화학습으로 사람 수준 제어" (2015)
지은이: 볼로디미르 므니 외
핵심 아이디어:
  - 뉴럴 네트워크로 Q(s,a)을 근사한다
  - *타깃 네트워크*을 쓰는 TD 타깃으로 학습한다
  - 경험 리플레이(experience replay)로 표본 사이의 상관을 끊는다

파일: appendix/rl/dqn.py
참고: 교육용 예제다. 모델 + 리플레이 + TD 로스 계산(완전한 환경 루프는 없다).
"""

from dataclasses import dataclass
import random
from collections import deque

# ========================================================================
# 메인
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """이산 액션을 위한 단순한 MLP Q(s,a) 근사기."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        return self.net(obs)  # (B, num_actions)


@dataclass
class Transition:
    """리플레이 버퍼에 담는 경험(전이) 하나."""
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """크기가 고정된 선입선출 리플레이 버퍼."""
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        # 필드를 쌓아 배치 텐서로 만든다
        s = torch.stack([b.s for b in batch], dim=0)
        a = torch.stack([b.a for b in batch], dim=0)
        r = torch.stack([b.r for b in batch], dim=0)
        s2 = torch.stack([b.s2 for b in batch], dim=0)
        done = torch.stack([b.done for b in batch], dim=0)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def dqn_td_loss(q_net: nn.Module, target_net: nn.Module, batch, gamma: float = 0.99):
    """
    DQN의 TD 로스를 계산한다.

    전이 (s,a,r,s',done)마다:
      target = r + gamma * (1-done) * max_a' Q_target(s', a')
      loss = MSE( Q(s,a), target )

    참고:
      - a은 이산 액션의 인덱스다(형태: (B,))
      - done은 끝이면 1, 아니면 0이다
    """
    s, a, r, s2, done = batch

    # 모든 액션에 대한 현재의 Q 값: (B, A)
    q_values = q_net(s)

    # gather으로 Q(s,a)을 고른다:
    # dim=1에서 gather 하려면 a의 형태가 (B,1)이어야 한다
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)  # (B,)

    # 타깃 네트워크로 타깃을 계산한다(그래디언트 없이)
    with torch.no_grad():
        q_next = target_net(s2)                   # (B, A)
        max_q_next = q_next.max(dim=1).values     # (B,)
        target = r + gamma * (1.0 - done) * max_q_next

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    # 간단한 스모크 테스트(환경 없음)
    obs_dim, num_actions = 8, 4
    q = QNetwork(obs_dim, num_actions)
    tgt = QNetwork(obs_dim, num_actions)

    # 가짜 배치
    B = 5
    s = torch.randn(B, obs_dim)
    a = torch.randint(0, num_actions, (B,))
    r = torch.randn(B)
    s2 = torch.randn(B, obs_dim)
    done = torch.randint(0, 2, (B,), dtype=torch.float32)

    loss = dqn_td_loss(q, tgt, (s, a, r, s2, done))
    print("loss:", float(loss))
```

**출력:**

```
loss: 1.1922111511230469
```

## 2. 논의

이 구현은 클래스 3개(`QNetwork`, `Transition`, `ReplayBuffer`)를 정의하고, TD 로스를 계산하는 함수 `dqn_td_loss`와 함께 DQN 학습의 부품을 이룬다. 환경과 상호작용하는 루프, ε-탐욕 액션 선택, 타깃 네트워크를 주기적으로 복사하는 부분은 빠져 있다(독스트링이 밝힌 대로다). 출력의 로스 값은 가짜 배치로 한 번 계산한 것이라 그 크기에 뜻은 없다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 쪽의 예제 설정(`obs_dim=8`, `num_actions=4`, `hidden=128`)으로 만든 `QNetwork`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. 8 × 128 + 128 = 1,152개, 128 × 128 + 128 = 16,512개, 128 × 4 + 4 = 516개로 모두 18,180개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검증하려면 핵심 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅 시간을 크게 줄여 주고, 코드를 다른 곳에서 다시 쓸 때도 잘못된 입력을 일찍 걸러 준다.

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
`QNetwork`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = QNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — DQN

이 구현은 클래스 3개(`QNetwork`, `Transition`, `ReplayBuffer`)를 정의하고, TD 로스 함수 `dqn_td_loss`와 함께 DQN 학습의 부품을 이룬다.

핵심 클래스는 `QNetwork`, `Transition`, `ReplayBuffer`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
