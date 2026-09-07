# 그래프 겹말기 신경망(GCN)

2017년 키프와 벨링이 내놓은 그래프 겹말기 신경망(GCN)은 그래프 얼개 자료에서 반쯤 스승 있는 배움을 하는 바탕 얼개이다. GCN은 맞섬 고르게 맞추기로 이웃을 모아 고른 격자의 겹말기 개념을 고르지 않은 그래프에 맞춘다. 이 바닥부터 짜기는 스펙트럼에서 실마리를 얻은 층마다의 퍼뜨리기 규칙 $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$의 얼개를 드러내고 마디 가름과 금융 업종 헤아리기에 쓰는 모습을 보인다.

## 코드

```python
"""
29.3.4: 그림 엮음 그물(GCN)
Kipf & Welling(2017)을 맨바닥부터 짜기.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# ========================================================================
# 메인
# ========================================================================


class GCNConvManual(nn.Module):
    """맨바닥부터 짠 GCN 겹말기 층."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.shape[0]

        # 스스로 이음 더하기
        loop = torch.arange(num_nodes, device=edge_index.device)
        loop = loop.unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, loop], dim=1)

        # 차수 셈하기
        src, dst = ei[0], ei[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))

        # 맞섬 고르게 맞추기: D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # 특징 바꾸기
        x_transformed = x @ self.weight

        # 고르게 맞추며 쪽지 건네기
        messages = x_transformed[src] * norm.unsqueeze(1)
        out = torch.zeros(num_nodes, x_transformed.shape[1], device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """마디 가름을 위한 여러 층 GCN."""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvManual(in_ch, hidden_ch))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvManual(hidden_ch, hidden_ch))
        self.convs.append(GCNConvManual(hidden_ch, out_ch))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_gcn_layer():
    """GCN 층 하나의 셈을 보여 준다."""
    print("=" * 60)
    print("GCN Layer Computation")
    print("=" * 60)

    torch.manual_seed(42)
    # 단순한 마디 4개 그래프
    edge_index = torch.tensor([[0,1,1,2,2,3,0,2],[1,0,2,1,3,2,2,0]], dtype=torch.long)
    x = torch.tensor([[1,0],[0,1],[1,1],[0,0]], dtype=torch.float)

    layer = GCNConvManual(2, 4)
    out = layer(x, edge_index)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output:\n{out.detach()}")


def demo_gcn_karate():
    """가라테 동아리에서 마디를 가르는 GCN."""
    print("\n" + "=" * 60)
    print("GCN: Karate Club Node Classification")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1
                       for i in range(n)], dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[[0, 1, 2, 3, 33, 32, 31, 30]] = True

    model = GCN(n, 16, 2, num_layers=2)
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
        acc = (pred == y).float().mean()
        test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


def demo_gcn_normalization_effect():
    """맞섬 고르게 맞추기의 효과를 보인다."""
    print("\n" + "=" * 60)
    print("Normalization Effect")
    print("=" * 60)

    # 별 그래프: 가운데 마디 0의 차수는 9, 나머지는 1
    G = nx.star_graph(9)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = 10

    # 스스로 이음 더하기
    A_tilde = A + np.eye(n)
    D_tilde = np.diag(A_tilde.sum(axis=1))

    # 고르게 맞추지 않음
    signal = np.zeros(n)
    signal[1] = 1.0  # 잎에 있는 신호

    aggregated_raw = A_tilde @ signal
    print(f"Raw aggregation (center node 0): {aggregated_raw[0]:.4f}")
    print(f"Raw aggregation (leaf node 1): {aggregated_raw[1]:.4f}")

    # 맞섬 고르게 맞추기
    D_inv_sqrt = np.diag(np.diag(D_tilde) ** -0.5)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    aggregated_norm = A_hat @ signal
    print(f"\nNormalized aggregation (center): {aggregated_norm[0]:.4f}")
    print(f"Normalized aggregation (leaf 1): {aggregated_norm[1]:.4f}")
    print("Normalization prevents the high-degree center from dominating")


def demo_gcn_financial():
    """업종을 헤아리는 금융 그물 위의 GCN."""
    print("\n" + "=" * 60)
    print("GCN: Financial Sector Prediction")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)
    n = 20
    n_sectors = 3
    sectors = np.random.randint(0, n_sectors, n)

    # 특징 만들기
    x = torch.randn(n, 8)
    for i in range(n):
        x[i, :3] += torch.tensor([sectors[i] * 0.5, 0, 0])

    # 얽힘 변(같은 업종일수록 잦다)
    src, dst = [], []
    for i in range(n):
        for j in range(i+1, n):
            p = 0.6 if sectors[i] == sectors[j] else 0.15
            if np.random.random() < p:
                src.extend([i, j]); dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor(sectors, dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:10] = True

    model = GCN(8, 16, n_sectors, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(1)
        test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    demo_gcn_layer()
    demo_gcn_karate()
    demo_gcn_normalization_effect()
    demo_gcn_financial()```

## 논의

이 짜기는 그래프 겹말기 신경망(GCN)의 핵심 논리를 감싼 `GCNConvManual`, `GCN` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
