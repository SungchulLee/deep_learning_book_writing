# 그래프 합성곱 신경망(GCN)

[GCN](https://arxiv.org/abs/1609.02907)은 자기 특징과 이웃 특징을 차수로 정규화해 결합한다. 이 코드는 노드별 선형 투영, 엣지별 가중 메시지와 수신 노드 합산을 직접 구현한다. 수식의 대칭 무방향 그래프 가정과 실제 edge_index의 방향·중복 처리를 맞추어 읽어야 한다.

## 1. 코드와 보존된 출력

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
    demo_gcn_financial()
```

```
============================================================
GCN Layer Computation
============================================================
Input: torch.Size([4, 2]) -> Output: torch.Size([4, 4])
Output:
tensor([[-0.0049, -0.0004,  0.0152, -0.0025],
        [-0.0049, -0.0004,  0.0152, -0.0025],
        [-0.0042, -0.0003,  0.0132, -0.0022],
        [-0.0028, -0.0002,  0.0086, -0.0014]])

============================================================
GCN: Karate Club Node Classification
============================================================
Overall accuracy: 0.9412
Test accuracy: 0.9231

============================================================
Normalization Effect
============================================================
Raw aggregation (center node 0): 1.0000
Raw aggregation (leaf node 1): 1.0000

Normalized aggregation (center): 0.2236
Normalized aggregation (leaf 1): 0.5000
Normalization prevents the high-degree center from dominating

============================================================
GCN: Financial Sector Prediction
============================================================
Test accuracy: 0.5000
```

출력은 기존 실행 예시다. 이번 편집에서는 학습·속도 측정이나 새로운 검증을 수행하지 않는다.

## 2. 자기 루프와 정규화

무방향 비음수 그래프에서 Ã=A+I, D̃ᵢᵢ=ΣⱼÃᵢⱼ로 두면 기본 전파는 다음과 같다.

$$H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW+b)$$

이 구현의 단일 GCNConvManual은 선형 집계와 bias까지만 반환한다. ReLU·Dropout은 바깥 GCN의 중간층에서 적용하며 마지막 층은 logits다. bias도 정규화 행렬을 곱한 뒤 더한다.

매 호출에서 각 노드의 자기 루프를 추가한다. 기존 입력의 루프를 검사·제거하지 않아 이미 루프가 있으면 중복으로 계산한다. 일반 중복 엣지도 각 항이 별도로 합산·차수에 반영된다. 입력을 단순 그래프로 가정할지 다중 메시지로 볼지 먼저 정해야 한다.

## 3. 텐서와 정보 전달 방향

X는 n×d_in, W는 d_in×d_out이다. x_transformed=XW를 한 번 계산하고 source의 행을 엣지별로 가져온다. 각 메시지에 1/√(d̃_src d̃_dst)를 곱한 뒤 **dst로 scatter_add**한다.

차수는 dst를 세어 계산한다. 양방향으로 저장한 무방향 그래프에서는 표준 대칭 차수와 같다. 임의의 방향 그래프에서는 양 끝점에 모두 진입 차수를 쓰는 다른 연산이며, 무방향 라플라스에서 유도한 표준 GCN 식과 자동으로 같은 해석을 갖지 않는다.

num_nodes는 X 행 수와 일관되어야 하고 모든 인덱스는 범위 안의 정수여야 한다. x·edge_index·가중치의 장치를 맞추고, 새0 텐서가 기본 dtype으로 생성되는 점도 혼합·배정밀도 사용 전에 확인한다. 함수가 임의 dtype과 장치를 자동으로 정렬해 주지는 않는다.

## 4. 정규화 예제의 수치

중심0과 잎9개인 별 그래프에 자기 루프를 더하면 중심 차수는10, 잎은2다. 잎1에만1인 impulse를 넣을 때 원시 합은 중심·잎1에서 각각1이다. 정규화 뒤에는 중심1/√20≈0.2236, 잎1은1/2다.

이는 차수에 따른 스케일 조정의 예이며, 어떤 특징·가중치에서도 중심 노드가 항상 작아지거나 학습이 안정하다는 보장은 아니다. 대칭 정규화 행렬은 일반적으로 행합1이 아니므로 확률 평균과도 구분한다.

## 5. Karate Club 노드 분류

34개 노드를 one-hot 특징으로 나타내고 엣지는 양방향으로 저장한다. NetworkX 원본 엣지의 weight는 사용하지 않고 끝점만 가져오므로 **비가중 그래프**다. 레이블은 두 club이며8개 노드의 라벨로200회 학습하고 나머지26개를 테스트한다.

전체 구조와 노드 ID가 알려진 transductive 설정이다. Overall accuracy는 학습 노드까지 포함한34개 정확도라 테스트 정확도와 다르다. 별도 검증 마스크와 여러 분할·시드의 변동은 제시하지 않는다. one-hot 입력이 새 노드의 일반적 특징 함수라는 뜻도 아니다.

기본34→16→2의 파라미터 수는(34×16+16)+(16×2+2)=594개다. num_layers=1이어도 생성자가 첫·마지막 층을 둘 다 넣어 실제1층이 되지 않는다. 계층 수 인수와 실제 ModuleList 길이를 확인해야 한다.

## 6. 업종 예제의 생성 조건

금융 예제는 실제 시장 데이터가 아니라20개 노드의 합성 데이터다. 무작위로 정한3개 업종 레이블을 사용해 첫 특징 좌표를 이동시키고, 같은 업종끼리의 엣지 확률을0.6·다른 업종은0.15로 정한다. 즉 특징과 그래프에 **정답을 반영한 동질성**이 설계되어 있다.

앞10개를 학습하고 뒤10개를 테스트한다. 출력 정확도는 이 작은 합성 설정의 값으로 실제 업종 예측·수익률·위험 성능을 입증하지 않는다. 정답으로 그래프를 만드는 합성 과정과 실제 예측 시점에 관측 가능한 관계로 만드는 그래프를 구분해야 한다.

## 7. 계산과 깊이

노드 투영은 O(nd_in d_out), 메시지·합산은 O((m+n)d_out) 정도다. 이 구현은 엣지별 변환 메시지를 저장하고 매 forward에서 루프·차수를 다시 계산한다. 그래프를 고정한 경우 전처리 캐시와 연산 fusion은 별도 개선이다.

깊이를 늘리면 더 멀리 전달하지만 반복 평활화와 과압축, 최적화 문제가 생길 수 있다. 정규화나 자기 루프가 이를 모두 해결하지 않는다. 구조·특징이 비동질적인 작업에서는 이웃 평균형 가정이 맞는지도 검증해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
엣지0–1만 있는 두 노드에서 자기 루프와 W=I, b=0인 GCN 집계를 구하라. X의 두 행은(1,0),(0,1)이다.

</div>

??? success "연습문제 1 풀이"

    루프 뒤 두 노드 차수는2이고 정규화 행렬은 모든 항이1/2인2×2 행렬이다. 두 출력 행 모두(0.5,0.5)다. 자기 루프가 있어도 원래 두 특징을 손실 없이 보존하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
별 그래프 정규화 데모에서 잎1의 impulse가 중심과 잎1에 주는 값을 계산하라.

</div>

??? success "연습문제 2 풀이"

    중심 차수10·잎 차수2이므로 중심으로 가는 계수는1/√20≈0.2236, 잎1의 자기 계수는1/2다. 원시 A+I 집계에서는 둘 다1이다. 이 한 예제가 모든 특징에서 중심의 영향이 항상 작다는 보장은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Karate와 합성 업종 데모의 테스트가 검증하는 범위를 비교하라.

</div>

??? success "연습문제 3 풀이"

    Karate는 전체34노드 구조·one-hot이 보이는 transductive8라벨 학습·26테스트다. 업종은 레이블로 특징 이동과 같은 업종의 연결 확률을 설계한 합성20노드에서10개를 테스트한다. 둘 다 새로운 실제 그래프·노드에 대한 보편 성능이나 실제 금융 예측을 입증하지 않는다. 설정을 비교하려면 분할·특징·예산과 반복 변동을 맞춰야 한다.



## 정리하며

GCN은 자기 루프 뒤 새 차수로 정규화한 메시지를 수신 노드에 합한다. 방향·중복·루프·dtype 조건과 단일 계층의 비선형 위치를 확인한다. 작은 transductive·합성 예제의 정확도를 일반화 성능으로 확대하지 않는다.
