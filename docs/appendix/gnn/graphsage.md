# GraphSAGE

GraphSAGE은 2017년 논문 "Inductive Representation Learning on Large Graphs"에서 나왔다. - 이웃을 샘플링해 집계한다 - 귀납적(inductive) 설정: 본 적 없는 노드나 그래프에도 일반화한다 - 대표적인 갱신 식: h'_i = sigma( W [h_i || AGG({h_j, j in N(i)})] ).

여기 구현은 GraphSAGE을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
GraphSAGE - 큰 그래프에서의 귀납적 표현 학습
논문: "큰 그래프에서의 귀납적 표현 학습" (2017)
지은이: 윌 해밀턴, 즈타오 잉, 유레 레스코베츠
핵심 아이디어:
  - 이웃을 샘플링해 집계한다
  - 귀납적(inductive) 설정: 본 적 없는 노드나 그래프에도 일반화한다
  - 대표적인 갱신 식: h'_i = sigma( W [h_i || AGG({h_j, j in N(i)})] )

파일: appendix/gnn/graphsage.py
참고: 밀집(dense) 인접 행렬에 평균 집계를 쓰는, 교육용 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class MeanAggregator(nn.Module):
    """이웃 피처의 평균을 계산한다."""
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # A: (N, N) 인접 행렬(0/1). 원하면 자기 루프(self-loop)를 포함한 것으로 본다
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid divide-by-zero
        neigh_mean = (A @ X) / deg
        return neigh_mean


class GraphSAGELayer(nn.Module):
    """
    평균 집계를 쓰는 GraphSAGE 레이어 하나.

    h'_i = sigma( W [h_i || mean_{j in N(i)} h_j] )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.agg = MeanAggregator()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        neigh = self.agg(X, A)                    # (N, in_dim)
        h_cat = torch.cat([X, neigh], dim=1)      # (N, 2*in_dim)
        return self.lin(h_cat)                    # (N, out_dim)


class GraphSAGE(nn.Module):
    """노드 분류를 위한 두 레이어 GraphSAGE."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.sage1(X, A))
        logits = self.sage2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 5, 8, 4
    X = torch.randn(N, Fin)

    # 자기 루프를 넣어 이웃 평균에 자기 노드도 들어가게 한다(흔한 기법)
    A = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GraphSAGE(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (5, 4)
```

**출력:**

```
logits: torch.Size([5, 4])
```

## 2. 논의

이 구현은 클래스 3개(`MeanAggregator`, `GraphSAGELayer`, `GraphSAGE`)를 정의하고, 이들이 함께 완전한 그래프 뉴럴 네트워크 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `MeanAggregator`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

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
`MeanAggregator`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = MeanAggregator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — GraphSAGE

이 구현은 클래스 3개(`MeanAggregator`, `GraphSAGELayer`, `GraphSAGE`)를 정의하고, 이들이 함께 완전한 그래프 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `MeanAggregator`, `GraphSAGELayer`, `GraphSAGE`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
