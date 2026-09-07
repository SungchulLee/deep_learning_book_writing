# GAT

GAT은 2018년 글 "Graph Attention Networks"에서 나왔다. - 붙박인 잣대 맞추기 대신 이웃에 대한 눈길 짐을 배운다 - 마디 i에서는 결에서 배운 alpha_{ij}으로 이웃 j을 모은다.

여기 짜보기는 GAT을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
GAT - 그림 눈길 그물
글: "그림 눈길 그물" (2018)
지은이: 페타르 벨리치코비치 외
고갱이 깨침:
  - 붙박인 잣대 맞추기 대신 이웃에 대한 눈길 짐을 배운다
  - 마디 i에서는 결에서 배운 alpha_{ij}으로 이웃 j을 모은다

두루마리: appendix/gnn/gat.py
눈여겨볼 것: 빽빽한 이웃 행렬과 머리 하나짜리 눈길을 쓰는, 배우기 위한 짜보기다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class GATLayer(nn.Module):
    """
    머리 하나짜리 GAT 켜(빽빽한 이웃 행렬).

    걸음:
      1) 선형 바꿈: Wh_i = W x_i
      2) 눈길 점수: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
      3) 이음 없는 곳을 가린다
      4) alpha_{ij} = softmax_j(e_{ij})
      5) h'_i = sum_j alpha_{ij} Wh_j
    """
    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # a을 두 조각으로 가른다(a^T [Wh_i || Wh_j]와 같다)
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: (N, Fin), A: (N, N) 이웃 행렬 (0/1)
        H = self.W(X)  # (N, Fout)

        # 눈길 로짓을 잘 들게 셈한다:
        # e_{ij} = LeakyReLU( a_l(H_i) + a_r(H_j) )
        e_l = self.attn_l(H)  # (N, 1)
        e_r = self.attn_r(H)  # (N, 1)
        e = e_l + e_r.T        # (N, N)
        e = self.leaky_relu(e)

        # 이웃이 아닌 곳을 가린다:
        # 아주 작은 음수를 써서 이음 없는 곳의 소프트맥스가 0에 가깝게 한다
        mask = (A == 0)
        e = e.masked_fill(mask, float("-inf"))

        # 이웃 j에 걸쳐 잣대를 맞춘다
        alpha = F.softmax(e, dim=1)  # (N, N)

        # 이웃 결의 짐 실은 합
        H_out = alpha @ H  # (N, Fout)
        return H_out


class GAT(nn.Module):
    """마디 가름을 위한 두 켜 GAT(머리 하나)."""
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
    print("logits:", logits.shape)  # (4, 3)```

## 논의

이 짜보기는 갈래 2개(`GATLayer`, `GAT`)를 매기고, 이들이 어울려 온전한 그림 신경 그물 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `GATLayer`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율 0.1을 써라. 눈길 드롭아웃이 다독임에 왜 도움이 되는지 밝혀라.

??? success "익힘 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 얼마를 아무렇게나 0으로 만들어, 모형이 낱말끼리의 어떤 사이에만 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 드러냄을 배우게 되는데, 이는 여느 드롭아웃이 신경 낱자리끼리 함께 길드는 것을 막는 것과 같다.

---

**익힘 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "익힘 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer은 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`GATLayer`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = GATLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
