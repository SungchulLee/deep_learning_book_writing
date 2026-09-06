# 그래프 눈길 신경망(GAT)

그래프 눈길 신경망(GAT)은 그래프 신경망의 쪽지 건네기에 배울 수 있는 눈길 계수를 들여와 마디마다 이웃의 특징에 따라 다른 무게를 주게 한다. 마디 차수에서 얻은 붙박인 고르게 맞추기 무게를 쓰는 그래프 겹말기 신경망과 달리, GAT은 모든 변에 걸쳐 함께 쓰는 눈길 얼개로 자료에 매인 눈길 점수를 그때그때 셈한다. 이 여러 갈래 눈길 방식은 모델이 서로 다른 나타냄 아래 공간의 앎에 함께 눈길을 주게 하여 뒤섞인 그래프에서 나타냄 힘을 크게 높인다.

## 코드

```python
"""
Chapter 29.3.6: Graph Attention Network (GAT)
그래프 위의 여러 머리 눈길.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# ========================================================================
# 메인
# ========================================================================


class GATConvManual(nn.Module):
    """여러 머리 눈길을 쓰는 GAT 겹말기 층."""

    def __init__(self, in_ch, out_ch, heads=1, concat=True,
                 dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_ch = out_ch
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_ch, heads * out_ch, bias=False)
        self.att = nn.Parameter(torch.randn(heads, 2 * out_ch))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = nn.Parameter(torch.zeros(heads * out_ch if concat else out_ch))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # 스스로 이음 더하기
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        # 선형 바꿈
        h = self.W(x).view(n, self.heads, self.out_ch)

        # 주의 점수
        h_src = h[src]  # [E, heads, out_ch]
        h_dst = h[dst]
        cat = torch.cat([h_src, h_dst], dim=-1)  # [E, heads, 2*out_ch]
        e = (cat * self.att.unsqueeze(0)).sum(dim=-1)  # [E, heads]
        e = self.leaky_relu(e)

        # 도착 마디마다 소프트맥스
        e_max = torch.full((n, self.heads), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax')
        alpha = torch.exp(e - e_max[dst])
        alpha_sum = torch.zeros(n, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-10)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 무게 매긴 모으기
        weighted = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.out_ch, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)

        if self.concat:
            out = out.view(n, self.heads * self.out_ch)
        else:
            out = out.mean(dim=1)

        return out + self.bias


class GAT(nn.Module):
    """여러 층 GAT."""

    def __init__(self, in_ch, hidden_ch, out_ch, heads=4, num_layers=2, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConvManual(in_ch, hidden_ch, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConvManual(hidden_ch * heads, hidden_ch, heads=heads, concat=True, dropout=dropout))
        self.convs.append(GATConvManual(hidden_ch * heads, out_ch, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_gat():
    """가라테 동아리에서의 GAT."""
    print("=" * 60)
    print("GAT: Karate Club Node Classification")
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
    train_mask[[0,1,2,3,33,32,31,30]] = True

    for heads in [1, 4, 8]:
        torch.manual_seed(42)
        model = GAT(n, 8, 2, heads=heads, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(1)
            test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
        print(f"  heads={heads}: Test Acc = {test_acc:.4f}")


def demo_attention_visualization():
    """눈길 계수를 그린다."""
    print("\n" + "=" * 60)
    print("Attention Coefficient Analysis")
    print("=" * 60)

    torch.manual_seed(42)
    n = 6
    edge_index = torch.tensor([
        [0,1,0,2,1,2,1,3,2,4,3,5,4,5],
        [1,0,2,0,2,1,3,1,4,2,5,3,5,4]], dtype=torch.long)
    x = torch.randn(n, 4)

    layer = GATConvManual(4, 4, heads=1, concat=False)
    layer.eval()

    with torch.no_grad():
        # 눈길을 손으로 뽑아내기
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n)
        src_all = torch.cat([src, loop])
        dst_all = torch.cat([dst, loop])

        h = layer.W(x).view(n, 1, 4)
        h_src = h[src_all]
        h_dst = h[dst_all]
        cat = torch.cat([h_src, h_dst], dim=-1)
        e = (cat * layer.att.unsqueeze(0)).sum(dim=-1)
        e = layer.leaky_relu(e)

        e_max = torch.full((n, 1), float('-inf'))
        e_max.scatter_reduce_(0, dst_all.unsqueeze(1).expand_as(e), e, reduce='amax')
        alpha = torch.exp(e - e_max[dst_all])
        alpha_sum = torch.zeros(n, 1)
        alpha_sum.scatter_add_(0, dst_all.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst_all].clamp(min=1e-10)

    print("Edge attention coefficients:")
    for i in range(edge_index.shape[1]):
        print(f"  {src[i].item()} -> {dst[i].item()}: {alpha[i, 0].item():.4f}")


if __name__ == "__main__":
    demo_gat()
    demo_attention_visualization()```

## 논의

이 짜기는 그래프 눈길 신경망(GAT)의 핵심 논리를 감싼 `GATConvManual`, `GAT` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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
