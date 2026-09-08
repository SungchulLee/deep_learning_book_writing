# ChebNet

ChebNet은 그래프 겹말기 연산의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
29.3.3: ChebNet
체비쇼프 다항식 스펙트럼 그래프 겹말기.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# ========================================================================
# 메인
# ========================================================================


class ChebConv(nn.Module):
    """체비쇼프 스펙트럼 그래프 겹말기 층."""

    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, lambda_max=2.0):
        n = x.shape[0]
        # 성긴 라플라스 짓기
        L = self._compute_laplacian(edge_index, n)
        L_tilde = (2.0 / lambda_max) * L - torch.eye(n, device=x.device)

        # 체비쇼프 되돌이
        Tx_0 = x
        out = self.lins[0](Tx_0)

        if self.K > 1:
            Tx_1 = L_tilde @ x
            out = out + self.lins[1](Tx_1)

        for k in range(2, self.K):
            Tx_2 = 2 * L_tilde @ Tx_1 - Tx_0
            out = out + self.lins[k](Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out + self.bias

    def _compute_laplacian(self, edge_index, n):
        A = torch.zeros(n, n, device=edge_index.device)
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        D = torch.diag(A.sum(dim=1))
        return D - A


class ChebNet(nn.Module):
    """마디 가름을 위한 여러 층 ChebNet."""

    def __init__(self, in_ch, hidden_ch, out_ch, K=3, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_ch, hidden_ch, K))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_ch, hidden_ch, K))
        self.convs.append(ChebConv(hidden_ch, out_ch, K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_chebyshev_polynomials():
    """체비쇼프 다항식을 그린다."""
    print("=" * 60)
    print("Chebyshev Polynomials")
    print("=" * 60)

    x = np.linspace(-1, 1, 200)
    T = [np.ones_like(x), x.copy()]
    for k in range(2, 6):
        T.append(2 * x * T[-1] - T[-2])

    for k in range(6):
        print(f"  T_{k}: max={np.max(np.abs(T[k])):.4f}, "
              f"zeros={np.sum(np.abs(np.diff(np.sign(T[k]))) > 0)}")


def demo_chebnet_node_classification():
    """가라테 동아리에서 마디를 가르는 ChebNet."""
    print("\n" + "=" * 60)
    print("ChebNet Node Classification")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.eye(n, dtype=torch.float)
    y = torch.tensor([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                       for i in range(n)], dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[::2] = True

    for K in [1, 2, 3, 5]:
        torch.manual_seed(42)
        model = ChebNet(n, 16, 2, K=K, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
        print(f"  K={K}: Test Accuracy = {test_acc:.4f}")


def demo_filter_localization():
    """K이 거르개가 한곳에 모이는 정도를 어떻게 다스리는지 보인다."""
    print("\n" + "=" * 60)
    print("Filter Localization (K-hop)")
    print("=" * 60)

    G = nx.path_graph(20)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    n = 20
    lmax = np.max(np.linalg.eigvalsh(L))
    L_tilde = 2 * L / lmax - np.eye(n)

    # 가운데 마디에 충격
    impulse = np.zeros(n)
    impulse[10] = 1.0

    for K in [1, 2, 3, 5, 10]:
        T0 = impulse.copy()
        response = T0.copy()
        if K > 1:
            T1 = L_tilde @ impulse
            response += T1
        T_prev, T_curr = T0, T1 if K > 1 else T0
        for k in range(2, K):
            T_next = 2 * L_tilde @ T_curr - T_prev
            response += T_next
            T_prev, T_curr = T_curr, T_next

        nonzero = np.where(np.abs(response) > 1e-10)[0]
        spread = nonzero[-1] - nonzero[0] + 1 if len(nonzero) > 0 else 0
        print(f"  K={K:2d}: spread={spread} nodes, "
              f"support=[{nonzero[0] if len(nonzero)>0 else '-'}, "
              f"{nonzero[-1] if len(nonzero)>0 else '-'}]")


if __name__ == "__main__":
    demo_chebyshev_polynomials()
    demo_chebnet_node_classification()
    demo_filter_localization()```

## 2. 논의

이 짜기는 ChebNet의 핵심 논리를 감싼 `ChebConv`, `ChebNet` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수들은 잘 알려진 그래프 자료 묶음에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 그래프 겹말기 연산에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — ChebNet

이 짜기는 ChebNet의 핵심 논리를 감싼 `ChebConv`, `ChebNet` 갈래를 한가운데 둔다.

고갱이 갈래는 `ChebConv`, `ChebNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
