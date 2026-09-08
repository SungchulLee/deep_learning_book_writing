# 마디 박아 넣기

마디 박아 넣기는 마디와 이음 켜 일의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""29.6.3: 마디 박아 넣기 - DeepWalk과 Node2Vec."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================

def random_walk(G, start, length):
    walk = [start]
    for _ in range(length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if not neighbors: break
        walk.append(np.random.choice(neighbors))
    return walk

def node2vec_walk(G, start, length, p=1.0, q=1.0):
    walk = [start]
    if len(list(G.neighbors(start))) == 0: return walk
    walk.append(np.random.choice(list(G.neighbors(start))))
    for _ in range(length - 2):
        cur = walk[-1]; prev = walk[-2]
        neighbors = list(G.neighbors(cur))
        if not neighbors: break
        probs = []
        for n in neighbors:
            if n == prev: probs.append(1.0/p)
            elif G.has_edge(n, prev): probs.append(1.0)
            else: probs.append(1.0/q)
        probs = np.array(probs); probs /= probs.sum()
        walk.append(np.random.choice(neighbors, p=probs))
    return walk

class SkipGramEmbedding(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, embed_dim)
        self.context = nn.Embedding(n_nodes, embed_dim)
    def forward(self, center, context, neg):
        c = self.embed(center); ctx = self.context(context); n = self.context(neg)
        pos_score = (c * ctx).sum(dim=-1)
        neg_score = (c.unsqueeze(1) * n).sum(dim=-1)
        return -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

def demo():
    print("=" * 60); print("Node Embedding"); print("=" * 60)
    np.random.seed(42); torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    # 걸음 만들기
    walks = []
    for _ in range(10):
        for node in range(n):
            walks.append(node2vec_walk(G, node, length=10, p=1, q=0.5))
    # 익히기 짝 짓기
    window = 3; centers, contexts = [], []
    for walk in walks:
        for i, c in enumerate(walk):
            for j in range(max(0,i-window), min(len(walk),i+window+1)):
                if i != j: centers.append(c); contexts.append(walk[j])
    model = SkipGramEmbedding(n, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    centers_t = torch.tensor(centers); contexts_t = torch.tensor(contexts)
    for epoch in range(50):
        idx = torch.randperm(len(centers))[:512]
        c, ctx = centers_t[idx], contexts_t[idx]
        neg = torch.randint(0, n, (512, 5))
        loss = model(c, ctx, neg)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    # 박아 넣기 살피기
    emb = model.embed.weight.detach()
    y = [0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)]
    c0 = emb[torch.tensor([i for i,l in enumerate(y) if l==0])].mean(0)
    c1 = emb[torch.tensor([i for i,l in enumerate(y) if l==1])].mean(0)
    print(f"  Cluster distance: {(c0-c1).norm():.4f}")

if __name__ == "__main__":
    demo()
```

## 2. 논의

이 짜기는 마디 박아 넣기의 핵심 논리를 감싼 `SkipGramEmbedding` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 마디 박아 넣기

이 짜기는 마디 박아 넣기의 핵심 논리를 감싼 `SkipGramEmbedding` 갈래를 한가운데 둔다.

고갱이 갈래는 `SkipGramEmbedding`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
