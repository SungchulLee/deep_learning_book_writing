# MPNN

MPNN은 2017년 글 "Neural Message Passing for Quantum Chemistry"에서 나왔다. - *알림* 함수와 *고침* 함수를 따로 둔다 - T 스텝 동안 거듭 퍼뜨린다: m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij}) h_i^{t+1} = U(h_i^t, m_i^{t+1}).

여기 구현은 MPNN을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
MPNN - 메시지 패싱 뉴럴 넷트웍(두루 쓰는 틀)
글: "양자 화학을 위한 뉴럴 메시지 패싱" (2017)
지은이: 저스틴 길머 외
핵심 깨침:
  - *알림* 함수와 *고침* 함수를 따로 둔다
  - T 스텝 동안 거듭 퍼뜨린다:
      m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij})
      h_i^{t+1} = U(h_i^t, m_i^{t+1})

두루마리: appendix/gnn/mpnn.py
눈여겨볼 것: 배우기 위한 구현이며 이런 것을 쓴다:
  - 빽빽한 이웃 행렬
  - 골라 쓰는 이음 결 행렬 E (N, N, E_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class MessageFn(nn.Module):
    """
    알림 함수 M(h_i, h_j, e_ij).
    여기서는 이어 붙인 뒤 MLP에 넣는다.
    """
    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(inplace=True),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_i, h_j, e_ij):
        # h_i, h_j: (msg_dim?) 이지만 여기서는 node_dim
        # e_ij: (edge_dim)
        x = torch.cat([h_i, h_j, e_ij], dim=-1)
        return self.net(x)


class UpdateFn(nn.Module):
    """
    고침 함수 U(h_i, m_i).
    여기서는 GRUCell 꼴 고침을 쓴다(단순하고 흔하다).
    """
    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=msg_dim, hidden_size=node_dim)

    def forward(self, h_i, m_i):
        # m_i: (node_dim?) 여기서는 msg_dim
        return self.gru(m_i, h_i)


class MPNN(nn.Module):
    """
    두루 쓰는 메시지 패싱 넷트웍.

    들임:
      X: (N, node_dim) 마디 결
      A: (N, N) 이웃 행렬 (0/1)
      E: (N, N, edge_dim) 이음 결(골라 쓴다. None이면 0을 쓴다)

    날임:
      H: (N, node_dim) T 스텝 뒤 고쳐진 마디 리프리젠테이션
    """
    def __init__(self, node_dim: int, edge_dim: int = 0, msg_dim: int = 64, T: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.msg_dim = msg_dim
        self.T = T

        self.message = MessageFn(node_dim, edge_dim, msg_dim)
        self.update = UpdateFn(node_dim, msg_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor, E: torch.Tensor | None = None):
        N = X.size(0)
        device = X.device

        # 이음 결이 주어지지 않으면 이음의 결을 0으로 본다
        if E is None:
            E = torch.zeros(N, N, self.edge_dim, device=device)

        H = X  # current node states (N, node_dim)

        # 메시지 패싱을 T 번 돈다
        for _ in range(self.T):
            messages = []

            # 마디 i마다 이웃 j에서 온 알림을 모은다
            for i in range(N):
                m_i_list = []

                for j in range(N):
                    # 이음이 있을 때만 알림을 보낸다 (A[i, j] == 1)
                    if A[i, j] > 0:
                        m_ij = self.message(H[i], H[j], E[i, j])  # (msg_dim,)
                        m_i_list.append(m_ij)

                # 더하기 모으기(MPNN에서 흔하다)
                if len(m_i_list) == 0:
                    m_i = torch.zeros(self.msg_dim, device=device)
                else:
                    m_i = torch.stack(m_i_list, dim=0).sum(dim=0)

                messages.append(m_i)

            M = torch.stack(messages, dim=0)  # (N, msg_dim)

            # 고침 함수로 마디 상태를 고친다
            H = self.update(H, M)  # (N, node_dim)

        return H


if __name__ == "__main__":
    N, node_dim, edge_dim = 4, 8, 3
    X = torch.randn(N, node_dim)

    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    E = torch.randn(N, N, edge_dim)  # random edge features (only used where A=1)

    model = MPNN(node_dim=node_dim, edge_dim=edge_dim, msg_dim=16, T=2)
    H = model(X, A, E)
    print("H:", H.shape)  # (4, 8)
```

**출력:**

```
H: torch.Size([4, 8])
```

## 2. 논의

이 구현은 클래스 3개(`MessageFn`, `UpdateFn`, `MPNN`)를 매기고, 이들이 어울려 전체 이미지 뉴럴 넷트웍 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `MessageFn`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
숨은 크기 $h$과 입력 크기 $x$이 같을 때 LSTM 칸과 GRU 칸의 파라메터 수를 비교하여라. 어느 쪽이 더 적고 왜 그런가?

</div>

??? success "연습문제 3 풀이"
    LSTM에는 문이 넷(입력, 잊음, 칸, 출력) 있고 저마다 입력과 은닉 상태의 짐 행렬을 지니므로 파라메터는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 문이 셋(되돌림, 고침, 새것) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU은 문이 넷이 아니라 셋이고 칸 상태와 은닉 상태를 하나로 묶으므로 LSTM의 75%이다. 참으로는 파라메터가 적어도 GRU이 LSTM과 엇비슷하게 잘 듣는 일이 잦다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MessageFn`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MessageFn(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — MPNN

이 구현은 클래스 3개(`MessageFn`, `UpdateFn`, `MPNN`)를 매기고, 이들이 어울려 전체 이미지 뉴럴 넷트웍 구조를 이룬다.

핵심 클래스는 `MessageFn`, `UpdateFn`, `MPNN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
