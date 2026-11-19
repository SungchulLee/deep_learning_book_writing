# GCN

GCN은 2017년 논문 "Semi-Supervised Classification with Graph Convolutional Networks"에서 나왔다. 노드 피처는 (정규화된) 이웃 피처를 집계해 갱신하며, 정규화된 인접 행렬 D^{-1/2} (A + I) D^{-1/2}을 쓴다.

여기 구현은 GCN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
GCN - 그래프 컨볼루션 네트워크
논문: "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
저자: Thomas Kipf, Max Welling
핵심 아이디어:
  - 노드 피처는 (정규화된) 이웃 피처를 집계해 갱신한다
  - 정규화된 인접 행렬을 쓴다:  D^{-1/2} (A + I) D^{-1/2}

파일: appendix/gnn/gcn.py
참고: 알아보기 쉽도록 밀집 인접 행렬을 쓰는 학습용 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    정규화된 인접 행렬을 계산한다:  D^{-1/2} (A + I) D^{-1/2}

    A: (N, N) 인접 행렬(0/1이거나 가중값)
    반환값:
      A_norm: (N, N)
    """
    N = A.size(0)

    # 셀프 루프를 더한다: A_hat = A + I
    A_hat = A + torch.eye(N, device=A.device)

    # 차수(degree) 행렬: D_hat[i] = sum_j A_hat[i, j]
    deg = A_hat.sum(dim=1)  # (N,)

    # D^{-1/2}: 0으로 나누지 않도록 주의한다
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 정규화: D^{-1/2} A_hat D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


class GCNLayer(nn.Module):
    """
    GCN 레이어 하나:
      H^{(l+1)} = sigma( A_norm H^{(l)} W )

    여기서:
      - H^{(l)}은 노드 피처 행렬 (N, Fin)
      - W는 학습 가능한 웨이트 (Fin, Fout)
      - A_norm은 정규화된 인접 행렬 (N, N)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # 피처에 웨이트를 곱한 뒤 그래프 구조를 따라 전파한다
        return A_norm @ self.lin(X)  # (N, out_dim)


class GCN(nn.Module):
    """
    노드 분류를 위한 단순한 두 레이어 GCN.

    입력:
      X: (N, Fin) 노드 피처
      A: (N, N) 인접 행렬
    출력:
      logits: (N, num_classes)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_norm = normalize_adjacency(A)
        H = F.relu(self.gcn1(X, A_norm))
        logits = self.gcn2(H, A_norm)
        return logits


if __name__ == "__main__":
    # 노드 4개짜리 장난감 예제
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)

    # 단순한 무방향 인접 행렬
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GCN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
```

**출력:**

```
logits: torch.Size([4, 3])
```

## 2. 논의

이 구현은 클래스 2개(`GCNLayer`, `GCN`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `GCNLayer`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주 함수나 클래스에 추가하라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 저마다 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 진단한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 과적합으로 진단한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GCNLayer`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = GCNLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — GCN

이 구현은 클래스 2개(`GCNLayer`, `GCN`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `GCNLayer`, `GCN`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
