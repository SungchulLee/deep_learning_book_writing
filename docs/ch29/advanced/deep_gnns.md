# 깊은 그래프 신경망

그래프 신경망은 셈틀 보기나 자연어 처리의 짝과 달리 층을 많이 쌓으면 성능이 나빠진다. 깊은 그래프 신경망은 남은 이음, 변 떨구기, 층 고르게 맞추기 같은 재주로 이 어려움을 다룬다. 이 얼개들은 마디 나타냄이 가릴 수 없는 벡터로 무너지지 않고도 앎이 여러 쪽지 건네기 걸음을 지나 퍼지게 한다.

## 1. 코드

```python
"""
29.4.1: 깊은 그래프 신경망 - 나머지 이음, 변 떨구기, 고르게 맞추기.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

# ========================================================================
# 메인
# ========================================================================

class GCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_all = torch.cat([src, loop]); dst_all = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_all, torch.ones(dst_all.shape[0], device=x.device))
        norm = (deg[src_all] * deg[dst_all]).pow(-0.5)
        norm[norm == float('inf')] = 0
        h = self.lin(x)
        msg = h[src_all] * norm.unsqueeze(1)
        out = torch.zeros(n, h.shape[1], device=x.device)
        out.scatter_add_(0, dst_all.unsqueeze(1).expand_as(msg), msg)
        return out

class ResGCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=8, dropout=0.5):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList([GCNLayer(hidden_ch, hidden_ch) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_ch) for _ in range(num_layers)])
        self.output_lin = nn.Linear(hidden_ch, out_ch)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, edge_index); x = norm(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        return self.output_lin(x)

def drop_edge(edge_index, p=0.2):
    mask = torch.rand(edge_index.shape[1]) > p
    return edge_index[:, mask]

def demo_deep_gcn():
    print("=" * 60); print("Deep GCN with Residual Connections"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True
    for nl in [2, 4, 8, 16]:
        torch.manual_seed(42)
        model = ResGCN(n, 16, 2, num_layers=nl); opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(model(x, ei)[tm], y[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei).argmax(1)[~tm] == y[~tm]).float().mean()
        print(f"  {nl:2d} layers: Test Acc = {acc:.4f}")

if __name__ == "__main__":
    demo_deep_gcn()
```

## 2. 논의

깊은 그래프 신경망을 세울 때의 한가운데 어려움은 지나친 매끄러워짐이다. 쪽지 건네기 층이 늘어날수록 마디 나타냄이 비슷한 값으로 모여, 뒤따르는 일에 필요한 가름 앎이 사라진다. 여기 보인 `ResGCN` 모델은 층마다 내놓기를 그 들임과 더하는 남은(건너뛰는) 이음을 더해 이를 곧바로 맞선다. 이는 어떤 그래프 겹말기 층이 거의 고른 내놓기를 내더라도 본디 특징이 지름길로 지켜지게 한다.

층 고르게 맞추기(`nn.LayerNorm`)는 층마다 숨은 나타냄의 분포를 안정시켜 이를 보완한다. 고르게 맞추지 않으면 되풀이되는 쪽지 건네기가 특징의 크기를 걷잡을 수 없이 키우거나 줄일 수 있다. 깨움을 아무렇게나 0으로 만들어 익히기에 규칙을 세우는 떨구기와 합치면, 이 재주들은 층이 8개, 16개 넘는 그래프 신경망을 익히는 튼튼한 차례가 된다.

`drop_edge` 도구는 익히는 동안 변을 아무렇게나 지우는 자료 늘리기 재주인 변 떨구기를 짠다. 앞먹임마다 그래프의 이어짐을 줄여, 변 떨구기는 앎이 퍼지고 마디가 서로 가릴 수 없게 되는 빠르기를 늦춘다. 남은 이음, 고르게 맞추기, 떨구기, 변 떨구기가 함께 깊은 그래프 신경망이 나타냄 품질을 지키면서 더 높은 차수의 이웃 얼개를 담게 하며, 이는 가라테 클럽 그래프에서 층 수를 달리해도 시험 정확도가 안정된 것으로 드러난다.

## 연습문제

**연습문제 1.**
보여 주기를 돌려 층 2개, 4개, 8개, 16개의 시험 정확도를 적어라. 그다음 `ResGCN`에서 남은 이음을 없애고(`x = x + res`을 그냥 `x = x`으로) 16층의 정확도를 견주어라. 어떤 일이 왜 일어나는가?

??? success "연습문제 1 풀이"
    16층에서 남은 이음이 없으면 정확도가 크게 떨어진다(흔히 50% 언저리, 곧 아무렇게나 찍는 수준). 남은 이음은 본디 신호를 지킨다: $x^{(l+1)} = f(x^{(l)}) + x^{(l)}$. 그것이 없으면 되풀이되는 모으기가 지나친 매끄러워짐으로 모든 마디 특징을 한곳으로 모은다. 남은 이음이 있으면 신경망이 앞선 나타냄에 여전히 닿을 수 있어 깊이 16에서도 정확도를 지킨다.

---

**연습문제 2.**
맞섬 고르게 맞추기 $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$의 스펙트럼 반지름이 많아야 1인 까닭과 이것이 깊은 신경망의 지나친 매끄러워짐과 어떻게 이어지는지 수학으로 밝혀라.

??? success "연습문제 2 풀이"
    행렬 $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$은 스스로 이음을 더한 이웃 행렬을 두 겹 확률 행렬처럼 고르게 맞춘 것이다. $D^{-1/2}(A+I)D^{-1/2}$의 줄 합이 가둬져 있으므로 고윳값 $\lambda_i$은 $|\lambda_i| \le 1$을 만족한다. 되풀이 곱 $\hat{A}^L x$은 스펙트럼을 줄인다. 가장 큰 것(상수 고유 벡터에 해당하며 1이다)을 뺀 모든 고윳값이 $\lambda_i^L \to 0$으로 지수로 사그라진다. 곧 층이 많아지면 $\hat{A}^L x$이 차수 분포에 비례하는 계수 1 행렬로 모여 마디마다의 앎을 모두 지운다. 이 스펙트럼 줄어듦이 지나친 매끄러워짐의 수학 원인이다.

---

**연습문제 3.**
익히기 걸음마다 확률 $p = 0.3$으로 변을 아무렇게나 떨구는 `DropEdge` 변형 익히기 되풀이를 짜라. 변 떨구기가 있을 때와 없을 때 16층 `ResGCN`의 성능을 견주어라. 도움이 되는가? 어떤 조건에서 해로울 수 있는가?

??? success "연습문제 3 풀이"
    ```python
    model = ResGCN(n, 16, 2, num_layers=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        ei_dropped = drop_edge(ei, p=0.3)  # 변의 30%를 떨군다
        F.cross_entropy(model(x, ei_dropped)[tm], y[tm]).backward()
        opt.step()
    ```
    변 떨구기는 마디 특징이 섞이는 빠르기를 늦추어 보통 깊은 그래프 신경망의 성능을 낫게 한다. 그러나 변을 지우면 조각이 끊기는 아주 성긴 그래프이거나 그래프 얼개가 그 일에 결정적일 때는 해로울 수 있다. (비교적 빽빽한) 가라테 클럽 그래프에서는 $p=0.3$의 변 떨구기가 16층에서 정확도를 지키거나 조금 낫게 한다.

## 정리하며

**다룬 것** — 깊은 그래프 신경망

깊은 그래프 신경망을 세울 때의 한가운데 어려움은 지나친 매끄러워짐이다.

고갱이 갈래는 `GCNLayer`, `ResGCN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
