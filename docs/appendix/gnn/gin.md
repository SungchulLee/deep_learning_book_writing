# GIN

GIN은 2019년 글 "How Powerful are Graph Neural Networks?"에서 나왔다. - 더하기 모으기 + MLP을 쓴다 - 고침: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j ) - (몇 가지 가정 아래) 바이스파일러-레만 시험만큼 힘세다고 밝혀졌다.

여기 구현은 GIN을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
GIN - 그림 같은꼴 넷트웍
글: "그림 뉴럴 넷트웍은 얼마나 힘센가?" (2019)
지은이: 커율루 쉬 외
핵심 깨침:
  - 더하기 모으기 + MLP을 쓴다
  - 고침: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )
  - (몇 가지 가정 아래) 바이스파일러-레만 시험만큼 힘세다고 밝혀졌다

두루마리: appendix/gnn/gin.py
눈여겨볼 것: 빽빽한 이웃 행렬을 쓰는, 배우기 위한 구현이다.
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
    GIN 켜 하나:
      h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )

    eps은 붙박아 두거나 배우게 할 수 있다. 여기서는 배우게 한다.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = MLP(in_dim, out_dim, hidden_dim=2 * out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 더하기 모으기에는 A이 이음을 담고 있으면 A @ X을 쓸 수 있다(제 고리는 골라 뺀다)
        neigh_sum = A @ X  # (N, in_dim)
        out = (1.0 + self.eps) * X + neigh_sum
        return self.mlp(out)


class GIN(nn.Module):
    """마디 가름을 위한 두 켜 GIN."""
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

    # GIN에서는 (1+eps)*X이 제 몫을 주므로 제 고리 없는 이웃 행렬이라도 괜찮다
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

이 구현은 클래스 3개(`MLP`, `GINLayer`, `GIN`)를 매기고, 이들이 어울려 전체 이미지 뉴럴 넷트웍 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `MLP`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 어그멘테이션)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MLP`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — GIN

이 구현은 클래스 3개(`MLP`, `GINLayer`, `GIN`)를 매기고, 이들이 어울려 전체 이미지 뉴럴 넷트웍 구조를 이룬다.

핵심 클래스는 `MLP`, `GINLayer`, `GIN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
