# Dueling DQN

Dueling DQN은 2016년 논문 "Dueling Network Architectures for Deep Reinforcement Learning"에서 나왔다. 상태 밸류 $V(s)$와 어드밴티지 $A(s,a)$를 따로 트레인한 뒤 $Q(s,a)$로 결합한다. $V(s)$는 그 상태 자체가 얼마나 좋은지, $A(s,a)$는 그 상태에서 액션 $a$가 다른 액션보다 얼마나 나은지를 뜻한다. 밸류가 비슷한 액션이 많을 때 도움이 된다 — 어느 액션을 골라도 비슷한 상태에서는 $V(s)$만 잘 배우면 되고, 그 트레인이 모든 액션의 Q 값에 함께 반영되기 때문이다.

여기 구현은 Dueling DQN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Dueling DQN - 밸류 스트림과 어드밴티지 스트림의 분리
논문: "심층 강화학습을 위한 듀얼링 네트워크 구조" (2016)
지은이: 쯔위 왕 외
핵심 아이디어:
  - V(s)과 A(s,a)을 따로 트레인한 뒤 Q(s,a)으로 결합한다
  - 밸류가 비슷한 액션이 많을 때 도움이 된다

결합:
  Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

파일: appendix/rl/dueling_dqn.py
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class DuelingQNetwork(nn.Module):
    """
    듀얼링 네트워크:
      공유 백본 -> 밸류 헤드 + 어드밴티지 헤드
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # 밸류 스트림은 스칼라 V(s) 하나를 낸다
        self.value = nn.Linear(hidden, 1)

        # 어드밴티지 스트림은 모든 액션에 대한 A(s,a)을 낸다
        self.adv = nn.Linear(hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)

        V = self.value(h)           # (B, 1)
        A = self.adv(h)             # (B, A)

        # 어드밴티지에서 평균을 빼 Q을 식별 가능하게 한다(안 그러면 V/A이 유일하게 정해지지 않는다)
        A_centered = A - A.mean(dim=1, keepdim=True)

        Q = V + A_centered          # (B, A)
        return Q


if __name__ == "__main__":
    net = DuelingQNetwork(obs_dim=8, num_actions=4)
    x = torch.randn(3, 8)
    q = net(x)
    print("Q:", q.shape)  # (3, 4)
```

**출력:**

```
Q: torch.Size([3, 4])
```

## 2. 논의

`DuelingQNetwork` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 쪽의 예제 설정(`obs_dim=8`, `num_actions=4`, `hidden=128`)으로 만든 `DuelingQNetwork`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. 공유 백본이 8 × 128 + 128 = 1,152개와 128 × 128 + 128 = 16,512개, 밸류 헤드가 128 + 1 = 129개, 어드밴티지 헤드가 128 × 4 + 4 = 516개로 모두 18,309개다.

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
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 트레인 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 트레인 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`DuelingQNetwork`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = DuelingQNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — Dueling DQN

`DuelingQNetwork` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다.

핵심 클래스는 `DuelingQNetwork`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
