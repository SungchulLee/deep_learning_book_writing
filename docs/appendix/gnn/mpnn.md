# MPNN

MPNN은 2017년 논문 "Neural Message Passing for Quantum Chemistry"에서 나왔다. 여러 그래프 뉴럴 네트워크를 한 틀로 묶은 것으로, *메시지* 함수 $M$과 *갱신* 함수 $U$를 따로 둔다. 노드마다 이웃이 보낸 메시지를 모으고 그것으로 자기 상태를 갱신하는 일을 $T$ 스텝 동안 되풀이한다.

$$
m_i^{t+1} = \sum_{j \in N(i)} M(h_i^t, h_j^t, e_{ij}), \qquad h_i^{t+1} = U(h_i^t, m_i^{t+1})
$$

$e_{ij}$는 에지 피처다(분자라면 결합의 종류). GCN·GAT·GIN은 $M$과 $U$를 저마다 다르게 고른 경우로 볼 수 있다.

여기 구현은 MPNN을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
MPNN - 메시지 패싱 뉴럴 네트워크(일반적인 틀)
논문: "양자 화학을 위한 뉴럴 메시지 패싱" (2017)
지은이: 저스틴 길머 외
핵심 아이디어:
  - *메시지* 함수와 *갱신* 함수를 따로 둔다
  - T 스텝 동안 반복해 전파한다:
      m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij})
      h_i^{t+1} = U(h_i^t, m_i^{t+1})

파일: appendix/gnn/mpnn.py
참고: 교육용 구현이며 다음을 쓴다:
  - 밀집(dense) 인접 행렬
  - 선택적인 엣지 피처 행렬 E (N, N, E_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class MessageFn(nn.Module):
    """
    메시지 함수 M(h_i, h_j, e_ij).
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
    갱신 함수 U(h_i, m_i).
    여기서는 GRUCell 방식의 갱신을 쓴다(단순하고 흔하다).
    """
    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=msg_dim, hidden_size=node_dim)

    def forward(self, h_i, m_i):
        # m_i: (node_dim?) 여기서는 msg_dim
        return self.gru(m_i, h_i)


class MPNN(nn.Module):
    """
    일반적인 메시지 패싱 네트워크.

    입력:
      X: (N, node_dim) 노드 피처
      A: (N, N) 인접 행렬 (0/1)
      E: (N, N, edge_dim) 엣지 피처(선택 사항. None이면 0을 쓴다)

    출력:
      H: (N, node_dim) T 스텝 뒤 갱신된 노드 표현
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

        # 엣지 피처가 주어지지 않으면 0으로 둔다
        if E is None:
            E = torch.zeros(N, N, self.edge_dim, device=device)

        H = X  # current node states (N, node_dim)

        # 메시지 패싱을 T 번 돈다
        for _ in range(self.T):
            messages = []

            # 노드 i마다 이웃 j에서 온 메시지를 모은다
            for i in range(N):
                m_i_list = []

                for j in range(N):
                    # 엣지가 있을 때만 메시지를 보낸다 (A[i, j] == 1)
                    if A[i, j] > 0:
                        m_ij = self.message(H[i], H[j], E[i, j])  # (msg_dim,)
                        m_i_list.append(m_ij)

                # 합 집계(MPNN에서 흔하다)
                if len(m_i_list) == 0:
                    m_i = torch.zeros(self.msg_dim, device=device)
                else:
                    m_i = torch.stack(m_i_list, dim=0).sum(dim=0)

                messages.append(m_i)

            M = torch.stack(messages, dim=0)  # (N, msg_dim)

            # 갱신 함수로 노드 상태를 갱신한다
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

이 구현은 클래스 3개(`MessageFn`, `UpdateFn`, `MPNN`)를 정의하고, 이들이 함께 완전한 그래프 뉴럴 네트워크 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 쪽의 예제 설정(`node_dim=8`, `edge_dim=3`, `msg_dim=16`)으로 만든 `MessageFn`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. 첫 선형 레이어는 2 × 8 + 3 = 19차원을 받으므로 19 × 16 + 16 = 320개, 둘째는 16 × 16 + 16 = 272개로 모두 592개다.

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
은닉 크기 $h$와 입력 크기 $x$가 같을 때 LSTM 셀과 GRU 셀의 파라메터 수를 비교하여라. 어느 쪽이 더 적고 왜 그런가?

</div>

??? success "풀이"
    LSTM에는 게이트가 넷(입력, 망각, 셀, 출력) 있고 각각 입력과 은닉 상태의 웨이트 행렬을 지니므로 파라메터는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 게이트가 셋(리셋, 업데이트, 후보) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU는 게이트가 넷이 아니라 셋이고 셀 상태와 은닉 상태를 하나로 합치므로 LSTM의 75%이다. 이 쪽의 `UpdateFn`이 쓰는 것도 `nn.GRUCell`이다. 실제로는 파라메터가 적어도 GRU가 LSTM과 비슷한 성능을 내는 일이 잦다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MessageFn`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = MessageFn(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — MPNN

이 구현은 클래스 3개(`MessageFn`, `UpdateFn`, `MPNN`)를 정의하고, 이들이 함께 완전한 그래프 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `MessageFn`, `UpdateFn`, `MPNN`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
