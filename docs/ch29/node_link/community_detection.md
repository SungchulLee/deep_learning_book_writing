# 그래프 신경망으로 하는 무리 찾기

그래프 신경망으로 하는 무리 찾기는 마디와 이음 켜 일의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""29.6.4: 그래프 신경망으로 하는 무리 찾기."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from sklearn.cluster import KMeans

# ========================================================================
# 메인
# ========================================================================

class GNNCommunity(nn.Module):
    """무리 찾기를 위한 그래프 신경망 담개."""
    def __init__(self, in_ch, hidden_ch, embed_ch):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, embed_ch)
    def forward(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        for lin in [self.lin1, self.lin2]:
            h = lin(x); msg = h[sa]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        return x

def demo():
    print("=" * 60); print("Community Detection"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    sizes = [25, 25, 25]; probs = [[0.3,0.02,0.02],[0.02,0.3,0.02],[0.02,0.02,0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    n = G.number_of_nodes(); true_labels = [0]*25+[1]*25+[2]*25
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)

    # 이음 헤아리기 손실로 익히기
    model = GNNCommunity(n, 32, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(100):
        opt.zero_grad(); z = model(x, ei)
        pos_ei = torch.tensor([[e[0] for e in edges[:50]],[e[1] for e in edges[:50]]], dtype=torch.long)
        pos_s = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=-1)
        neg_ei = torch.stack([torch.randint(0,n,(50,)), torch.randint(0,n,(50,))])
        neg_s = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=-1)
        loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z = model(x, ei).numpy()
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(z)
    pred = km.labels_
    # NMI 비슷한 맞음(가장 좋은 자리바꿈)
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in pred]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc = max(best_acc, acc)
    print(f"  GNN + KMeans accuracy: {best_acc:.4f}")

    # 스펙트럼 뭉치기와 견주기
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(1)); L = D - A
    _, U = np.linalg.eigh(L)
    km2 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(U[:, 1:4])
    best_acc2 = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in km2.labels_]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc2 = max(best_acc2, acc)
    print(f"  Spectral clustering accuracy: {best_acc2:.4f}")

if __name__ == "__main__":
    demo()```

## 2. 논의

이 짜기는 그래프 신경망 무리 찾기의 핵심 논리를 감싼 `GNNCommunity` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 잘 알려진 그래프 자료 묶음에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
    이 얼개 고르기는 마디와 이음 켜 일에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 그래프 신경망으로 하는 무리 찾기

이 짜기는 그래프 신경망 무리 찾기의 핵심 논리를 감싼 `GNNCommunity` 갈래를 한가운데 둔다.

고갱이 갈래는 `GNNCommunity`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
