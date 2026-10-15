# DQN

DQN은 2015년 글 "Human-level control through deep reinforcement learning"에서 나왔다. - 신경망로 Q(s,a)을 어림한다 - *타깃 네트워크*을 쓰는 TD 과녁으로 학습된이다 - 겪음 되짚기로 얽힘을 끊는다.

여기 짜보기는 DQN을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 걸음에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DQN - 깊은 Q 그물
글: "깊은 북돋움 배움으로 사람 수준 다루기" (2015)
지은이: 볼로디미르 므니 외
핵심 깨침:
  - 신경 그물로 Q(s,a)을 어림한다
  - *과녁 그물*을 쓰는 TD 과녁으로 익힌다
  - 겪음 되짚기로 얽힘을 끊는다

두루마리: appendix/rl/dqn.py
눈여겨볼 것: 배우기 위한 본이다. 모형 + 되짚기 + TD 잃음 셈(온전한 둘레 되돌이는 없다).
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
    """따로 떨어진 움직임을 위한 단순한 MLP Q(s,a) 어림개."""
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
    """되짚기 곳간에 담는 겪음 하나."""
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """크기가 붙박인 선입선출 되짚기 곳간."""
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        # 밭을 쌓아 배치 텐서로 만든다
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
    DQN의 TD 잃음을 셈한다.

    넘어감 (s,a,r,s',done)마다:
      target = r + gamma * (1-done) * max_a' Q_target(s', a')
      loss = MSE( Q(s,a), target )

    눈여겨볼 것:
      - a은 따로 떨어진 움직임의 번호다(꼴: (B,))
      - done은 끝이면 1, 아니면 0이다
    """
    s, a, r, s2, done = batch

    # 모든 움직임에 대한 이제의 Q 값: (B, A)
    q_values = q_net(s)

    # gather으로 Q(s,a)을 고른다:
    # dim=1에서 gather 하려면 a의 꼴이 (B,1)이어야 한다
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)  # (B,)

    # 과녁 그물로 과녁을 셈한다(기울기 없이)
    with torch.no_grad():
        q_next = target_net(s2)                   # (B, A)
        max_q_next = q_next.max(dim=1).values     # (B,)
        target = r + gamma * (1.0 - done) * max_q_next

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    # 장난감 맛보기 시험(둘레 없음)
    obs_dim, num_actions = 8, 4
    q = QNetwork(obs_dim, num_actions)
    tgt = QNetwork(obs_dim, num_actions)

    # 거짓 배치
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

이 짜보기는 갈래 3개(`QNetwork`, `Transition`, `ReplayBuffer`)를 매기고, 이들이 어울려 온전한 강화학습 구조를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `QNetwork`의 배울 수 있는 매개변수를 모두 세어라. 짐과 편향을 아울러 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 편향이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 편향이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 갈래가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 손실은 줄어드는데 따짐 손실이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`QNetwork`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = QNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DQN

이 짜보기는 갈래 3개(`QNetwork`, `Transition`, `ReplayBuffer`)를 매기고, 이들이 어울려 온전한 강화학습 구조를 이룬다.

핵심 갈래는 `QNetwork`, `Transition`, `ReplayBuffer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
