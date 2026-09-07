# GCN

GCN was introduced in the 2017 paper "Semi-Supervised Classification with Graph Convolutional Networks." - Node features are updated by aggregating (normalized) neighbor features   - Uses normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}.

This implementation provides a concise, educational reference for GCN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
"""
GCN - Graph Convolutional Network
Paper: "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
Authors: Thomas N. Kipf, Max Welling
Key idea:
  - Node features are updated by aggregating (normalized) neighbor features
  - Uses normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

File: appendix/gnn/gcn.py
Note: Educational implementation using dense adjacency for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

    A: (N, N) adjacency matrix (0/1 or weighted)
    Returns:
      A_norm: (N, N)
    """
    N = A.size(0)

    # Add self-loops: A_hat = A + I
    A_hat = A + torch.eye(N, device=A.device)

    # Degree matrix: D_hat[i] = sum_j A_hat[i, j]
    deg = A_hat.sum(dim=1)  # (N,)

    # D^{-1/2}: careful about division by zero
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: D^{-1/2} A_hat D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


class GCNLayer(nn.Module):
    """
    One GCN layer:
      H^{(l+1)} = sigma( A_norm H^{(l)} W )

    Where:
      - H^{(l)} is node feature matrix (N, Fin)
      - W is learnable weight (Fin, Fout)
      - A_norm is normalized adjacency (N, N)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # Multiply features by weight, then propagate via graph structure
        return A_norm @ self.lin(X)  # (N, out_dim)


class GCN(nn.Module):
    """
    A simple 2-layer GCN for node classification.

    Inputs:
      X: (N, Fin) node features
      A: (N, N) adjacency
    Output:
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
    # Toy example with 4 nodes
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)

    # Simple undirected adjacency
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GCN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)```

## 논의

이 짜보기는 갈래 2개(`GCNLayer`, `GCN`)를 매기고, 이들이 어울려 온전한 그림 신경 그물 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `GCNLayer`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
`GCNLayer`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = GCNLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
