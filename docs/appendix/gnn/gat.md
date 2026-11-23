# GAT

GAT는 2018년 논문 "Graph Attention Networks"에서 나왔다. GCN의 고정된 정규화 대신 이웃에 대한 어텐션 웨이트를 트레인하며, 노드 $i$는 피처에서 계산한 웨이트 $\alpha_{ij}$로 이웃 $j$를 가중합해 집계한다. $\alpha_{ij}$는 두 노드의 변환된 피처로 점수 $e_{ij}$를 매긴 뒤 노드 $i$의 이웃에 걸쳐 소프트맥스를 취한 값이라, 이웃마다 반영되는 비중이 데이터에 따라 달라진다.

여기 구현은 GAT를 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
GAT - 그래프 어텐션 네트워크
논문: "Graph Attention Networks" (2018)
저자: Petar Veličković 외
핵심 아이디어:
  - 고정된 정규화 대신 이웃에 대한 어텐션 웨이트를 트레인한다
  - 노드 i에서는 피처에서 트레인한 alpha_{ij}로 이웃 j를 집계한다

파일: appendix/gnn/gat.py
참고: 밀집 인접 행렬과 단일 헤드 어텐션을 쓰는 트레인용 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class GATLayer(nn.Module):
    """
    단일 헤드 GAT 레이어(밀집 인접 행렬).

    단계:
      1) 선형 변환: Wh_i = W x_i
      2) 어텐션 점수: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
      3) 에지가 없는 곳을 마스킹한다
      4) alpha_{ij} = softmax_j(e_{ij})
      5) h'_i = sum_j alpha_{ij} Wh_j
    """
    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # a를 두 부분으로 나눈다(a^T [Wh_i || Wh_j]와 같다)
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: (N, Fin), A: (N, N) 인접 행렬 (0/1)
        H = self.W(X)  # (N, Fout)

        # 어텐션 로짓을 효율적으로 계산한다:
        # e_{ij} = LeakyReLU( a_l(H_i) + a_r(H_j) )
        e_l = self.attn_l(H)  # (N, 1)
        e_r = self.attn_r(H)  # (N, 1)
        e = e_l + e_r.T        # (N, N)
        e = self.leaky_relu(e)

        # 이웃이 아닌 곳을 마스킹한다:
        # 아주 작은 음수를 써서 에지가 없는 곳의 소프트맥스가 0에 가깝게 한다
        mask = (A == 0)
        e = e.masked_fill(mask, float("-inf"))

        # 이웃 j에 걸쳐 정규화한다
        alpha = F.softmax(e, dim=1)  # (N, N)

        # 이웃 피처의 가중합
        H_out = alpha @ H  # (N, Fout)
        return H_out


class GAT(nn.Module):
    """노드 분류를 위한 두 레이어 GAT(단일 헤드)."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.elu(self.gat1(X, A))
        logits = self.gat2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)
    A = torch.tensor([
        [1, 1, 1, 0],  # include self-loop in GAT for stability
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GAT(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
```

**출력:**

```
logits: torch.Size([4, 3])
```

## 2. 논의

이 구현은 클래스 2개(`GATLayer`, `GAT`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`GATLayer(in_dim=8, out_dim=16)`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. `GATLayer`의 선형 레이어 셋은 모두 `bias=False`라 웨이트만 있으므로, `W`가 8 × 16 = 128개, `attn_l`과 `attn_r`이 16개씩으로 모두 160개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(이웃 피처 `H`와 곱하기 앞)에 드롭아웃 레이어를 추가하라. 트레인하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 트레인하는 동안 어텐션 웨이트의 일부를 무작위로 0으로 만들어, 모델이 특정 이웃 노드와의 관계에만 의존하는 것을 막는다(GAT 논문도 정규화된 어텐션 웨이트에 드롭아웃을 건다). 그래서 모델이 어텐션을 더 고루 분산하고 더 강건한 표현을 학습하게 되는데, 이는 일반 드롭아웃이 뉴런끼리의 공적응(co-adaptation)을 막는 것과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 구하라. 이것이 긴 시퀀스에 Longformer나 Linformer 같은 구조를 쓰게 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    일반 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택된 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 밸류를 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GATLayer`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = GATLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — GAT

이 구현은 클래스 2개(`GATLayer`, `GAT`)를 정의하고, 이들이 함께 전체 그래프 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `GATLayer`, `GAT`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
