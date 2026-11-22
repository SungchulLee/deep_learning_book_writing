# GIN

GIN은 2019년 논문 "How Powerful are Graph Neural Networks?"에서 나왔다. 합 집계(sum aggregation)와 MLP를 쓰며, 갱신식은 $h'_i = \mathrm{MLP}\left((1 + \epsilon)\, h_i + \sum_{j \in N(i)} h_j\right)$이다. (몇 가지 가정 아래) 바이스파일러-레만(WL) 검사만큼 표현력이 강하다고 밝혀졌다. WL 검사는 두 그래프가 동형(같은 구조)인지 가리는 고전적인 방법으로, 노드마다 이웃의 라벨을 모아 새 라벨을 붙이기를 되풀이한다. 평균이나 최댓값과 달리 합은 이웃의 개수까지 구별하므로, 합 집계가 이 검사의 힘을 따라갈 수 있다.

여기 구현은 GIN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
GIN - 그래프 동형 네트워크
논문: "How Powerful are Graph Neural Networks?" (2019)
저자: Keyulu Xu 외
핵심 아이디어:
  - 합 집계 + MLP를 쓴다
  - 갱신: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )
  - (몇 가지 가정 아래) 바이스파일러-레만 검사만큼 표현력이 강하다고 밝혀졌다

파일: appendix/gnn/gin.py
참고: 밀집 인접 행렬을 쓰는 트레인용 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class MLP(nn.Module):
    """GIN 안에서 쓰는 작은 MLP."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GINLayer(nn.Module):
    """
    GIN 레이어 하나:
      h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )

    eps는 고정하거나 트레인시킬 수 있다. 여기서는 트레인시킨다.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = MLP(in_dim, out_dim, hidden_dim=2 * out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 합 집계에는 A가 에지를 담고 있으면 A @ X를 쓸 수 있다(셀프 루프는 선택 사항이다)
        neigh_sum = A @ X  # (N, in_dim)
        out = (1.0 + self.eps) * X + neigh_sum
        return self.mlp(out)


class GIN(nn.Module):
    """노드 분류를 위한 두 레이어 GIN."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gin1 = GINLayer(in_dim, hidden_dim)
        self.gin2 = GINLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.gin1(X, A))
        logits = self.gin2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 6, 3
    X = torch.randn(N, Fin)

    # GIN에서는 (1+eps)*X가 자기 자신의 기여를 주므로 셀프 루프 없는 인접 행렬이라도 괜찮다
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GIN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
```

**출력:**

```
logits: torch.Size([4, 3])
```

## 2. 논의

이 구현은 클래스 3개(`MLP`, `GINLayer`, `GIN`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 쪽의 예제 설정(`in_dim=6`, `hidden_dim=16`)에서 첫 레이어 `gin1` 안에 든 `MLP`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. `GINLayer`가 은닉 차원을 `2 * out_dim` = 32로 주므로, 이 `MLP`는 6 × 32 + 32 = 224개와 32 × 16 + 16 = 528개로 모두 752개다. `gin1` 전체를 세면 여기에 `eps` 한 개가 더해진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주 함수나 클래스에 추가하라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅 시간을 크게 줄여 주고, 코드를 다른 곳에서 다시 쓸 때도 잘못된 입력을 일찍 걸러 준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 저마다 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 진단한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 트레인 로스는 줄어드는데 검증 로스가 오르면 과적합으로 진단한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 트레인 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MLP`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — GIN

이 구현은 클래스 3개(`MLP`, `GINLayer`, `GIN`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `MLP`, `GINLayer`, `GIN`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
